// quant.go holds the streaming data passes for the new quantization
// targets (int4, int8_convrot, mxfp4, and nvfp4). Each pass reads the
// source tensor in bounded chunks and writes the converted bytes to w,
// so peak memory is O(chunk) regardless of tensor size.
package stconv

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

// streamInt4Data reads a tensor in bounded chunks, quantizes each element
// to int4 with the precomputed per-tensor scale (round-to-nearest-even),
// packs two nibbles per byte, and writes the result to w. An odd element
// count ends with a trailing zero padding nibble (see packNibbles).
func streamInt4Data(r io.ReaderAt, w io.Writer, offset int64, srcDType DType, numElems int64, scale float32, chunkElems int) error {
	// Two elements pack into one byte, so every interior chunk must hold an
	// even element count to emit whole bytes; only the final chunk may be
	// odd, and its trailing zero nibble is the required tensor-level pad.
	if chunkElems%2 == 1 {
		chunkElems++
	}
	elemSize, err := srcDType.ByteSize()
	if err != nil {
		return err
	}
	inBuf := make([]byte, chunkElems*elemSize)
	nibBuf := make([]uint8, chunkElems)

	var done int64
	for done < numElems {
		n := int64(chunkElems)
		if numElems-done < n {
			n = numElems - done
		}
		chunkIn := inBuf[:n*int64(elemSize)]
		if _, err := r.ReadAt(chunkIn, offset+done*int64(elemSize)); err != nil {
			return err
		}
		floats, err := toFloat32Slice(chunkIn, srcDType)
		if err != nil {
			return err
		}
		for i, f := range floats {
			nibBuf[i] = uint8(f32ToInt4RNE(f, scale) & 0x0F)
		}
		if _, err := w.Write(packNibbles(nibBuf[:n])); err != nil {
			return err
		}
		done += n
	}
	return nil
}

// mxfp4Block is the MXFP4 block scale size: one E8M0 scale per 32
// elements.
const mxfp4Block = 32

// streamMxFP4 converts a tensor to OCP MXFP4: the elements are grouped
// into blocks of 32, each block gets an E8M0 scale (e8m0Encode of the
// block max, ignoring NaN/Inf - the same convention as the int8 max-abs
// scan) and each element is quantized to E2M1 (round-to-nearest-even) at
// that scale, two elements packed per byte.
//
// The conversion takes two chunked passes over the source, matching the
// header layout (the ".block_scale" sibling precedes the owner):
//
//	pass 1 (w at the block-scale region): per block, decode to f32, take
//	the block max, write e8m0Encode(blockMax).
//	pass 2 (data): per block, RECOMPUTE the block max and code (recompute,
//	never buffer), s := e8m0Scale(code) (code 0 -> s 0), q_i :=
//	f32ToE2M1(x_i / s) (s == 0 -> 0), packNibbles, write.
//
// The effective read chunk is a whole number of 32-element blocks: the
// caller's chunkElems is rounded up to the next multiple of 32 (a value
// below one block becomes a single block), so an interior read never
// splits a block - only the tensor's final block may be partial, and its
// packNibbles zero-pads the trailing nibble. The only non-chunk state is
// the decoded 32-element f32 block buffer and one scalar (block max /
// code), both bounded constants.
func streamMxFP4(r io.ReaderAt, w io.Writer, offset int64, srcDType DType, numElems int64, chunkElems int) error {
	elemSize, err := srcDType.ByteSize()
	if err != nil {
		return err
	}

	// Round the user's chunk up to whole blocks (see the comment above): a
	// value below one block becomes a single block, otherwise the next
	// multiple of 32.
	chunk := int64(mxfp4Block)
	if c := int64(chunkElems); c > chunk {
		chunk = (c + mxfp4Block - 1) / mxfp4Block * mxfp4Block
	}
	rawBuf := make([]byte, chunk*int64(elemSize))

	// blockMax is the max |v| over vals, ignoring NaN/Inf (same convention
	// as the int8 max-abs scan: those are clamped at quantize time).
	blockMax := func(vals []float32) float32 {
		var m float32
		for _, v := range vals {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				continue
			}
			if v < 0 {
				v = -v
			}
			if v > m {
				m = v
			}
		}
		return m
	}

	// forEachBlock reads the source in `chunk`-element chunks and invokes
	// fn on each 32-element block (decoded to f32), passing its element
	// count (<= mxfp4Block; only the tensor's final block is partial). It
	// is used by both passes, so pass 2 re-reads and recomputes rather
	// than reusing pass 1's values.
	forEachBlock := func(fn func(vals []float32) error) error {
		var start int64
		for start < numElems {
			n := chunk
			if numElems-start < n {
				n = numElems - start
			}
			chunkRaw := rawBuf[:n*int64(elemSize)]
			if _, err := r.ReadAt(chunkRaw, offset+start*int64(elemSize)); err != nil {
				return err
			}
			fl, err := toFloat32Slice(chunkRaw, srcDType)
			if err != nil {
				return err
			}
			for i := 0; i < len(fl); i += mxfp4Block {
				cnt := mxfp4Block
				if len(fl)-i < cnt {
					cnt = len(fl) - i
				}
				if err := fn(fl[i : i+cnt]); err != nil {
					return err
				}
			}
			start += n
		}
		return nil
	}

	// Pass 1 (scales): one E8M0 byte per block, written to w before the
	// data (the ".block_scale" sibling precedes the owner in the file).
	var scaleByte [1]byte
	if err := forEachBlock(func(vals []float32) error {
		scaleByte[0] = e8m0Encode(blockMax(vals))
		_, err := w.Write(scaleByte[:])
		return err
	}); err != nil {
		return err
	}

	// Pass 2 (data): recompute each block's max and code, quantize the
	// elements to E2M1 at the decoded scale, and pack two per byte.
	if err := forEachBlock(func(vals []float32) error {
		s := e8m0Scale(e8m0Encode(blockMax(vals)))
		qs := make([]uint8, len(vals))
		for i, v := range vals {
			if s == 0 {
				qs[i] = 0
			} else {
				qs[i] = f32ToE2M1(v / s)
			}
		}
		_, err := w.Write(packNibbles(qs))
		return err
	}); err != nil {
		return err
	}

	return nil
}

// nvfp4Block is the NVFP4 block scale size: one E4M3 scale per 16
// elements.
const nvfp4Block = 16

// streamNVFP4 converts a tensor to NVIDIA NVFP4 and returns the
// per-tensor global scale alpha:
//
//	alpha = M / (6*448)   (M = max |x|, NaN/Inf ignored - the same
//	convention as the int8 max-abs scan; M == 0 -> alpha == 0)
//
// Each 16-element block gets an E4M3 (round-to-nearest-even) scale
// s = RNE_e4m3(blockMax/(6*alpha)), and every element is quantized to
// E2M1 (RNE) at the step alpha*s: q = RNE_e2m1(x/(alpha*s))
// (alpha*s == 0 -> 0), two elements packed per byte. Dequantization is
// x_hat = q*s*alpha.
//
// Since blockMax <= M, blockMax/(6*alpha) <= 448 always, so the E4M3
// scale never saturates past 448 (0x7E is the largest code emitted) and
// the 0x7F NaN pattern is structurally unreachable (pinned in the tests).
//
// The conversion takes three chunked passes over the source, matching
// the header layout (the ".global_scale" and ".block_scale" siblings
// both precede the owner):
//
//	pass 0: global max M.
//	pass 1 (w at the global-scale region): write alpha as a 4-byte LE
//	F32, then per block: RECOMPUTE blockMax and write
//	f32ToE4M3RNE(blockMax/(6*alpha)) (alpha == 0 -> 0).
//	pass 2 (data): per block: RECOMPUTE blockMax and the scale byte,
//	s := decode(byte), q_i := f32ToE2M1(x_i/(alpha*s)) (alpha*s == 0
//	-> 0), packNibbles, write.
//
// The effective read chunk is a whole number of 16-element blocks: the
// caller's chunkElems is rounded up to the next multiple of 16 (a value
// below one block becomes a single block), so an interior read never
// splits a block - only the tensor's final block may be partial, and its
// packNibbles zero-pads the trailing nibble. The only non-chunk state is
// the decoded 16-element f32 block buffer and the scalars M, alpha, and
// the block max - all bounded constants.
func streamNVFP4(r io.ReaderAt, w io.Writer, offset int64, srcDType DType, numElems int64, chunkElems int) (float32, error) {
	elemSize, err := srcDType.ByteSize()
	if err != nil {
		return 0, err
	}

	// Pass 0: global max M (a plain chunked scan; no block alignment
	// needed).
	M, err := streamComputeMaxAbsScale(r, offset, srcDType, numElems, chunkElems)
	if err != nil {
		return 0, err
	}
	var alpha float32
	if M != 0 {
		alpha = M / (6 * 448)
	}

	// Round the user's chunk up to whole blocks (see the comment above):
	// a value below one block becomes a single block, otherwise the next
	// multiple of 16.
	chunk := int64(nvfp4Block)
	if c := int64(chunkElems); c > chunk {
		chunk = (c + nvfp4Block - 1) / nvfp4Block * nvfp4Block
	}
	rawBuf := make([]byte, chunk*int64(elemSize))

	// blockMax is the max |v| over vals, ignoring NaN/Inf (same
	// convention as the int8 max-abs scan: those are clamped at
	// quantize time).
	blockMax := func(vals []float32) float32 {
		var m float32
		for _, v := range vals {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				continue
			}
			if v < 0 {
				v = -v
			}
			if v > m {
				m = v
			}
		}
		return m
	}

	// forEachBlock reads the source in `chunk`-element chunks and invokes
	// fn on each 16-element block (decoded to f32), passing its element
	// count (<= nvfp4Block; only the tensor's final block is partial). It
	// is used by passes 1 and 2, so each pass re-reads and recomputes
	// rather than reusing the previous pass's values.
	forEachBlock := func(fn func(vals []float32) error) error {
		var start int64
		for start < numElems {
			n := chunk
			if numElems-start < n {
				n = numElems - start
			}
			chunkRaw := rawBuf[:n*int64(elemSize)]
			if _, err := r.ReadAt(chunkRaw, offset+start*int64(elemSize)); err != nil {
				return err
			}
			fl, err := toFloat32Slice(chunkRaw, srcDType)
			if err != nil {
				return err
			}
			for i := 0; i < len(fl); i += nvfp4Block {
				cnt := nvfp4Block
				if len(fl)-i < cnt {
					cnt = len(fl) - i
				}
				if err := fn(fl[i : i+cnt]); err != nil {
					return err
				}
			}
			start += n
		}
		return nil
	}

	// scaleByte is the block's stored E4M3 scale. With alpha == 0 the
	// whole tensor is zero (blockMax is 0 everywhere and
	// blockMax/(6*alpha) would be 0/0), so every scale is 0.
	scaleByte := func(vals []float32) uint8 {
		if alpha == 0 {
			return 0
		}
		return f32ToE4M3RNE(blockMax(vals) / (6 * alpha))
	}

	// Pass 1: the 4-byte LE F32 global scale, then one E4M3 byte per
	// block, written to w before the data (both scale siblings precede
	// the owner in the file).
	var gs [4]byte
	binary.LittleEndian.PutUint32(gs[:], math.Float32bits(alpha))
	if _, err := w.Write(gs[:]); err != nil {
		return 0, err
	}
	var sb [1]byte
	if err := forEachBlock(func(vals []float32) error {
		sb[0] = scaleByte(vals)
		_, err := w.Write(sb[:])
		return err
	}); err != nil {
		return 0, err
	}

	// Pass 2 (data): recompute each block's scale byte, decode it, and
	// quantize the elements to E2M1 at the step alpha*s, packing two per
	// byte.
	if err := forEachBlock(func(vals []float32) error {
		step := alpha * f8E4M3ToF32(scaleByte(vals))
		qs := make([]uint8, len(vals))
		for i, v := range vals {
			if step == 0 {
				qs[i] = 0
			} else {
				qs[i] = f32ToE2M1(v / step)
			}
		}
		_, err := w.Write(packNibbles(qs))
		return err
	}); err != nil {
		return 0, err
	}

	return alpha, nil
}

// convrotGroup is the ConvRot rotation group size: 256 consecutive
// elements in flat row-major order, rotated by the 256x256 Sylvester
// Hadamard (see hadamard256).
const convrotGroup = 256

// hadamard256 rotates buf in place by the orthonormal regular Hadamard
// transform y = H_256·x / 16 (Sylvester construction). It is the 8-stage
// fast Walsh-Hadamard butterfly: for each stride s in 1,2,4,...,128,
// every block of 2s elements is folded with (a,b) := (a+b, a-b). The
// butterfly computes exactly the Sylvester Hadamard product in natural
// order (no bit-reversal permutation), and the final /16 = 1/sqrt(256)
// makes the transform orthonormal (H·Hᵀ = 256·I, so (H/16)·(H/16)ᵀ = I).
// Every operation is an f32 add, subtract, or divide-by-a-power-of-two -
// no rounding hazard for representable values - so the transform is
// deterministic: the same input always yields bit-identical output, which
// is what makes the multi-pass streaming conversion below reproducible.
func hadamard256(buf []float32) {
	for s := 1; s < convrotGroup; s <<= 1 {
		for i := 0; i < convrotGroup; i += 2 * s {
			for j := 0; j < s; j++ {
				a, b := buf[i+j], buf[i+j+s]
				buf[i+j], buf[i+j+s] = a+b, a-b
			}
		}
	}
	for i := range buf {
		buf[i] /= 16
	}
}

// streamConvRot converts a tensor to int8 ConvRot (see
// int8_convrot_guide.md): every group of 256 consecutive elements in flat
// row-major order is rotated by the orthonormal Hadamard (hadamard256),
// and each row (the first dimension; a 1-D tensor is one row) is
// quantized symmetrically with scale = rowMax/127 over the ROTATED
// values, using f32ToInt8 (half-away rounding, like the plain int8
// target). This is a value-changing conversion: the bytes written are the
// rotated weights, which a loader must pair with the matching activation
// rotation at inference time.
//
// The conversion takes three chunked passes over the source (scales, row
// rescan, quantize) and buffers nothing that grows with the tensor: the
// only non-chunk state is the 256-element group buffer (1KB of f32) and
// the current row's max - both bounded constants. Rotation groups are the
// atomic read unit (a group is always read whole and rotated in place),
// so chunkElems does not further subdivide the reads; it is kept for API
// consistency with the other streaming passes.
//
// w must be positioned at the ".scale" region: pass 1 writes the per-row
// scales in row order, then pass 2 writes the quantized data - matching
// the header layout, which emits the row-scale sibling before the owner.
// numElems must be a multiple of 256 (planTensor skips anything else).
func streamConvRot(r io.ReaderAt, w io.Writer, offset int64, srcDType DType, shape []int64, numElems int64, chunkElems int) (int, error) {
	_ = chunkElems // see the comment above: group reads are the atomic unit
	elemSize, err := srcDType.ByteSize()
	if err != nil {
		return 0, err
	}

	// c is the row width in elements; the row of element i is i/c. A 1-D
	// tensor is a single row of the whole tensor.
	var c int64
	if len(shape) >= 2 {
		c = 1
		for _, d := range shape[1:] {
			c *= d
		}
	} else {
		c = numElems // a 1-D tensor is one row of the whole tensor
	}
	if c <= 0 || numElems%c != 0 {
		return 0, fmt.Errorf("convrot: %d elements not divisible into rows of width %d", numElems, c)
	}
	rows := numElems / c

	// One group read/rotate: read 256 consecutive elements, decode, and
	// rotate in place. rawBuf and group are the only non-chunk state.
	rawBuf := make([]byte, convrotGroup*elemSize)
	group := make([]float32, convrotGroup)
	readGroup := func(g int64) error {
		if _, err := r.ReadAt(rawBuf, offset+g*convrotGroup*int64(elemSize)); err != nil {
			return err
		}
		f, err := toFloat32Slice(rawBuf, srcDType)
		if err != nil {
			return err
		}
		copy(group, f)
		hadamard256(group)
		return nil
	}

	// rowMaxOf updates m with |v|, ignoring NaN/Inf (same convention as
	// the int8 max-abs scan: they are clamped at quantization time).
	rowMaxOf := func(m, v float32) float32 {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return m
		}
		if v < 0 {
			v = -v
		}
		if v > m {
			return v
		}
		return m
	}

	var scaleBuf [4]byte
	writeScale := func(scale float32) error {
		binary.LittleEndian.PutUint32(scaleBuf[:], math.Float32bits(scale))
		_, err := w.Write(scaleBuf[:])
		return err
	}

	// Pass 1 (scales): scan the groups in flat order, tracking the
	// current row's max over the rotated values. Groups may straddle row
	// boundaries when c % 256 != 0, so the row is tracked per element;
	// each row's scale is written as soon as its last element is seen.
	curRow := int64(0)
	var rowMax float32
	for g := int64(0); g*convrotGroup < numElems; g++ {
		if err := readGroup(g); err != nil {
			return 0, err
		}
		for j := 0; j < convrotGroup; j++ {
			if rr := (g*convrotGroup + int64(j)) / c; rr > curRow {
				if err := writeScale(int8Scale(rowMax)); err != nil {
					return 0, err
				}
				curRow = rr
				rowMax = 0
			}
			rowMax = rowMaxOf(rowMax, group[j])
		}
	}
	if err := writeScale(int8Scale(rowMax)); err != nil {
		return 0, err
	}

	// Pass 2 (data): for each row, (a) rescan the groups overlapping the
	// row to recompute its max - recompute, never buffer - then (b)
	// rescan again and quantize, writing the I8 bytes in element order.
	// Each write is at most one group (256 bytes), so nothing here grows
	// with the row width or the tensor.
	var qbuf [convrotGroup]byte
	for rr := int64(0); rr < rows; rr++ {
		start, end := rr*c, (rr+1)*c
		g0, g1 := start/convrotGroup, (end-1)/convrotGroup
		// groupRange is the element range of group g that falls inside
		// [start, end), as indices within the group buffer.
		groupRange := func(g int64) (int, int) {
			s, e := g*convrotGroup, (g+1)*convrotGroup
			if s < start {
				s = start
			}
			if e > end {
				e = end
			}
			return int(s - g*convrotGroup), int(e - g*convrotGroup)
		}

		var m float32
		for g := g0; g <= g1; g++ {
			if err := readGroup(g); err != nil {
				return 0, err
			}
			l, h := groupRange(g)
			for j := l; j < h; j++ {
				m = rowMaxOf(m, group[j])
			}
		}
		scale := int8Scale(m)

		for g := g0; g <= g1; g++ {
			if err := readGroup(g); err != nil {
				return 0, err
			}
			l, h := groupRange(g)
			for j := l; j < h; j++ {
				qbuf[j-l] = byte(f32ToInt8(group[j], scale))
			}
			if _, err := w.Write(qbuf[:h-l]); err != nil {
				return 0, err
			}
		}
	}
	return int(rows), nil
}
