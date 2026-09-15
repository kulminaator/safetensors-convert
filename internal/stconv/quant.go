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
//
// The per-chunk quantize+pack is fanned out across cores over whole packed
// bytes (mapContiguous on the byte count): worker ranges are even element
// boundaries, so a packed byte never straddles two workers, and each worker
// packs its bytes into a disjoint sub-range of packedBuf. The final odd
// tensor-level pad nibble stays in the last byte, which is one worker's to
// write in full - every byte of the written prefix is fully assigned by
// exactly one worker (the pad nibble is an explicit zero), so the reused
// packedBuf cannot leak stale nibbles, the same guarantee
// packNibblesInto's zero prefixing gives.
//
// The chunk loop double-buffers the raw input (see
// streamComputeMaxAbsScale): the first chunk is read synchronously, each
// later chunk is prefetched into a second chunk-sized buffer while the
// previous chunk is quantized, so the next chunk's read latency overlaps
// this chunk's CPU work.
func streamInt4Data(r io.ReaderAt, w io.Writer, offset int64, srcDType DType, numElems int64, scale float32, chunkElems int, sc *passScratch, prog *Progress) error {
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
	// The per-pass buffers come from the run's passScratch (see
	// passScratch): allocated once per run, sized to chunkElems, and
	// reused across tensors - no per-tensor or per-chunk allocation. The
	// even-rounded chunkElems is <= the scratch's max window (a multiple
	// of 256), so every slice below fits.
	inBuf := sc.raw1[:int64(chunkElems)*int64(elemSize)]
	// inBuf2 is the prefetch (double-buffer) counterpart of inBuf: while
	// chunk i is being quantized, the next chunk i+1 is read into inBuf2
	// in a goroutine. +1 raw chunk buffer over the serial path, bounded
	// by the chunk (memory rule): both are chunk-sized, never tensor-sized.
	inBuf2 := sc.raw2[:int64(chunkElems)*int64(elemSize)]
	// One chunk-sized f32 decode scratch and one packed-output buffer
	// (at most (chunkElems+1)/2 bytes) reused across all chunks: O(chunk)
	// memory, no per-chunk allocations.
	fbuf := sc.fbuf[:chunkElems]
	packedBuf := sc.packedInt4[:(chunkElems+1)/2]

	// The first chunk has no predecessor to overlap with, so it is read
	// synchronously; every later chunk arrives via the previous
	// iteration's prefetch (see below), already in inBuf. A zero-element
	// tensor issues no read at all (the loop below never runs), matching
	// the serial path.
	if numElems > 0 {
		firstN := int64(chunkElems)
		if numElems < firstN {
			firstN = numElems
		}
		if _, err := r.ReadAt(inBuf[:firstN*int64(elemSize)], offset); err != nil {
			return err
		}
	}

	var done int64
	for done < numElems {
		n := int64(chunkElems)
		if numElems-done < n {
			n = numElems - done
		}
		chunkIn := inBuf[:n*int64(elemSize)]
		// Prefetch the next chunk into inBuf2 before this chunk's
		// quantize+pack, so its read latency overlaps this chunk's CPU
		// work. The last chunk has no successor, so no prefetch is
		// started for it.
		var nextErr chan error
		if done+n < numElems {
			nextN := int64(chunkElems)
			if rem := numElems - done - n; rem < nextN {
				nextN = rem
			}
			nextOff := offset + (done+n)*int64(elemSize)
			nextBuf := inBuf2
			nextErr = make(chan error, 1)
			go func() {
				_, e := r.ReadAt(nextBuf[:nextN*int64(elemSize)], nextOff)
				nextErr <- e
			}()
		}
		floats, err := toFloat32SliceInto(fbuf, chunkIn, srcDType)
		if err != nil {
			return err
		}
		// Fan the quantize+pack out over the chunk's (n+1)/2 packed bytes:
		// byte b holds elements 2b and 2b+1 (the latter only when it
		// exists - for the odd final chunk its missing high nibble is the
		// tensor-level pad, written as an explicit zero). Each worker
		// writes only its own disjoint packedBuf[lo:hi] sub-range; all
		// ranges are sub-slices of the existing per-tensor buffers - no
		// new allocations.
		nelems := int(n)
		mapContiguous((nelems+1)/2, func(_, lo, hi int) {
			for b := lo; b < hi; b++ {
				e := 2 * b
				v := uint8(f32ToInt4RNE(floats[e], scale) & 0x0F) // low nibble
				if e+1 < nelems {
					hi := uint8(f32ToInt4RNE(floats[e+1], scale) & 0x0F)
					v |= hi << 4 // high nibble
				}
				packedBuf[b] = v
			}
		})
		if _, err := w.Write(packedBuf[:(nelems+1)/2]); err != nil {
			return err
		}
		// Progress: this chunk's source bytes are quantized and handed to
		// the writer.
		if prog != nil {
			prog.Add(n * int64(elemSize))
		}
		// Join the prefetch before the next quantize: a failed read fails
		// the run here (at this chunk boundary, one chunk later than the
		// serial path, with the same ReadAt error). Then swap the buffers
		// so the prefetched chunk becomes the current one.
		if nextErr != nil {
			if err := <-nextErr; err != nil {
				return err
			}
			inBuf, inBuf2 = inBuf2, inBuf
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
//	the block max, store e8m0Encode(blockMax) in the chunk's scales
//	buffer; the chunk's scales are written in one call.
//	pass 2 (data): per block, RECOMPUTE the block max and code (recompute,
//	never buffer), s := e8m0Scale(code) (code 0 -> s 0), q_i :=
//	f32ToE2M1(x_i / s) (s == 0 -> 0), pack into the chunk's packed
//	buffer; the chunk's packed bytes are written in one call.
//
// The effective read chunk is a whole number of 32-element blocks: the
// caller's chunkElems is rounded up to the next multiple of 32 (a value
// below one block becomes a single block), so an interior read never
// splits a block - only the tensor's final block may be partial, and its
// packNibbles zero-pads the trailing nibble. Within each chunk the
// per-block work is fanned out across cores (mapStrided over the chunk's
// blocks): pass 1 stores one E8M0 byte per block in the chunk-sized
// scales buffer and pass 2 stores each block's packed bytes in a
// disjoint sub-range of the chunk-sized packed buffer, so each pass
// issues one write per chunk instead of one per block. The only
// non-chunk state is the chunk-sized f32 decode buffer, the chunk-sized
// scales and packed buffers, and the per-worker quantized-value scratch
// (one slice per fan-out worker, at most 32 uint8 each) - all bounded by
// the chunk or by the CPU count times the block size, never the tensor.
func streamMxFP4(r io.ReaderAt, w io.Writer, offset int64, srcDType DType, numElems int64, chunkElems int, sc *passScratch, prog *Progress) error {
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
	// The per-pass buffers come from the run's passScratch (see
	// passScratch): allocated once per run, sized to the max window, and
	// reused across tensors - no per-tensor or per-chunk allocation.
	// Memory bound (memory rule): rawBuf/rawBuf2 (the chunk's raw bytes,
	// the second is the prefetch's double buffer) and fbuf (the chunk's
	// f32 decode) scale with the chunk; scales holds one E8M0 byte per
	// block in a full chunk (chunk/32 bytes) and packed holds the chunk's
	// packed output ((chunk+1)/2 bytes). All are bounded by the chunk,
	// never the tensor. qs is the per-worker quantized-value scratch for
	// pass 2: parallelism-many slices, each at most mxfp4Block uint8,
	// bounded by the CPU count times the block size, never the tensor.
	rawBuf := sc.raw1[:chunk*int64(elemSize)]
	rawBuf2 := sc.raw2[:chunk*int64(elemSize)]
	fbuf := sc.fbuf[:int(chunk)]
	scales := sc.mxfp4Scales[:chunk/mxfp4Block]
	packed := sc.mxfp4Packed[:(chunk+1)/2]
	// qs gives each fan-out worker its own quantized-value scratch (at
	// most mxfp4Block uint8) so concurrent blocks never share a buffer.
	// Sized parallelism(full-chunk block count): bounded by the CPU count
	// (never the chunk or tensor), and parallelism is monotone, so the
	// scratch's slice (sized for the max window) covers every smaller
	// final chunk.
	qs := sc.mxfp4Qs[:max(1, parallelism(int(chunk/mxfp4Block)))]

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

	// forEachBlock reads the source in `chunk`-element chunks (whole
	// blocks) and invokes fn once per chunk with the decoded elements
	// (at most chunk; only the tensor's final chunk may be short). The fn
	// fans the per-block work out across cores (mapStrided) and writes the
	// chunk's output in one call. It is used by both passes, so pass 2
	// re-reads and recomputes rather than reusing pass 1's values.
	//
	// The chunk loop double-buffers the raw input (see
	// streamComputeMaxAbsScale): the first chunk is read synchronously,
	// each later chunk is prefetched into rawBuf2 while the previous
	// chunk's blocks are computed, so the next chunk's read latency
	// overlaps this chunk's CPU work. The prefetch changes only timing -
	// the bytes read and the result are identical to the serial loop.
	forEachBlock := func(fn func(fl []float32) error) error {
		// The first chunk has no predecessor to overlap with, so it is
		// read synchronously; every later chunk arrives via the previous
		// iteration's prefetch (see below), already in rawBuf. A
		// zero-element tensor issues no read at all (the loop below never
		// runs), matching the serial path.
		if numElems > 0 {
			firstN := chunk
			if numElems < firstN {
				firstN = numElems
			}
			if _, err := r.ReadAt(rawBuf[:firstN*int64(elemSize)], offset); err != nil {
				return err
			}
		}
		var start int64
		for start < numElems {
			n := chunk
			if numElems-start < n {
				n = numElems - start
			}
			chunkRaw := rawBuf[:n*int64(elemSize)]
			// Prefetch the next chunk into rawBuf2 before this chunk's
			// blocks are computed, so its read latency overlaps this
			// chunk's CPU work. The last chunk has no successor, so no
			// prefetch is started for it.
			var nextErr chan error
			if start+n < numElems {
				nextN := chunk
				if rem := numElems - start - n; rem < nextN {
					nextN = rem
				}
				nextOff := offset + (start+n)*int64(elemSize)
				nextBuf := rawBuf2
				nextErr = make(chan error, 1)
				go func() {
					_, e := r.ReadAt(nextBuf[:nextN*int64(elemSize)], nextOff)
					nextErr <- e
				}()
			}
			fl, err := toFloat32SliceInto(fbuf, chunkRaw, srcDType)
			if err != nil {
				return err
			}
			if err := fn(fl); err != nil {
				return err
			}
			// Progress: this chunk's source bytes went through the pass's
			// compute and write (the chunk's output was written inside fn).
			if prog != nil {
				prog.Add(n * int64(elemSize))
			}
			// Join the prefetch before the next chunk: a failed read fails
			// the run here (at this chunk boundary, one chunk later than
			// the serial path, with the same ReadAt error). Then swap the
			// buffers so the prefetched chunk becomes the current one.
			if nextErr != nil {
				if err := <-nextErr; err != nil {
					return err
				}
				rawBuf, rawBuf2 = rawBuf2, rawBuf
			}
			start += n
		}
		return nil
	}

	// Pass 1 (scales): one E8M0 byte per block, fanned out across the
	// chunk's blocks into the chunk-sized scales buffer (disjoint writes,
	// one per block), then written to w once per chunk before the data
	// (the ".block_scale" sibling precedes the owner in the file).
	if err := forEachBlock(func(fl []float32) error {
		numBlocks := (len(fl) + mxfp4Block - 1) / mxfp4Block
		mapStrided(numBlocks, func(b int) {
			lo := b * mxfp4Block
			cnt := mxfp4Block
			if len(fl)-lo < cnt {
				cnt = len(fl) - lo
			}
			scales[b] = e8m0Encode(blockMax(fl[lo : lo+cnt]))
		})
		_, err := w.Write(scales[:numBlocks])
		return err
	}); err != nil {
		return err
	}

	// Pass 2 (data): recompute each block's max and code, quantize the
	// elements to E2M1 at the decoded scale, and pack two per byte. The
	// per-block quantize+pack is fanned out across the chunk's blocks
	// (mapStrided): each worker writes its block's packed bytes into a
	// disjoint sub-range of the chunk-sized packed buffer, using its own
	// qs scratch (qs[b%k] for k workers - each worker owns one residue
	// class mod k, so no buffer is shared). The write is one per chunk.
	// A full 32-element block packs to exactly 16 bytes, so block b's
	// sub-range starts at byte b*16; the partial final block packs to
	// (cnt+1)/2 bytes and its trailing pad nibble is zeroed by
	// packNibblesInto's prefix zeroing.
	if err := forEachBlock(func(fl []float32) error {
		numBlocks := (len(fl) + mxfp4Block - 1) / mxfp4Block
		k := parallelism(numBlocks)
		mapStrided(numBlocks, func(b int) {
			lo := b * mxfp4Block
			cnt := mxfp4Block
			if len(fl)-lo < cnt {
				cnt = len(fl) - lo
			}
			vals := fl[lo : lo+cnt]
			s := e8m0Scale(e8m0Encode(blockMax(vals)))
			qv := qs[b%k][:cnt]
			for i, v := range vals {
				if s == 0 {
					qv[i] = 0
				} else {
					qv[i] = f32ToE2M1(v / s)
				}
			}
			// packNibblesInto zero-prefixes its dst, so the reused
			// packed sub-range cannot leak stale nibbles.
			packNibblesInto(packed[b*mxfp4Block/2:], qv)
		})
		// The chunk's packed output is (len(fl)+1)/2 bytes: each full
		// block contributes 16 and the partial final block (cnt+1)/2.
		_, err := w.Write(packed[:(len(fl)+1)/2])
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
//	pass 0: global max M (the S7 parallel chunked scan).
//	pass 1 (w at the global-scale region): write alpha as a 4-byte LE
//	F32 (once), then per block: RECOMPUTE blockMax and store
//	f32ToE4M3RNE(blockMax/(6*alpha)) (alpha == 0 -> 0) in the chunk's
//	scales buffer; the chunk's scales are written in one call.
//	pass 2 (data): per block: RECOMPUTE blockMax and the scale byte,
//	s := decode(byte), q_i := f32ToE2M1(x_i/(alpha*s)) (alpha*s == 0
//	-> 0), pack into the chunk's packed buffer; the chunk's packed bytes
//	are written in one call.
//
// The effective read chunk is a whole number of 16-element blocks: the
// caller's chunkElems is rounded up to the next multiple of 16 (a value
// below one block becomes a single block), so an interior read never
// splits a block - only the tensor's final block may be partial, and its
// packNibbles zero-pads the trailing nibble. Within each chunk the
// per-block work is fanned out across cores (mapStrided over the chunk's
// blocks): pass 1 stores one E4M3 byte per block in the chunk-sized
// scales buffer and pass 2 stores each block's packed bytes in a
// disjoint sub-range of the chunk-sized packed buffer, so each pass
// issues one write per chunk instead of one per block. The only
// non-chunk state is the chunk-sized f32 decode buffer, the chunk-sized
// scales and packed buffers, the scalars M and alpha, and the per-worker
// quantized-value scratch (one slice per fan-out worker, at most 16
// uint8 each) - all bounded by the chunk or by the CPU count times the
// block size, never the tensor.
func streamNVFP4(r io.ReaderAt, w io.Writer, offset int64, srcDType DType, numElems int64, chunkElems int, sc *passScratch, prog *Progress) (float32, error) {
	elemSize, err := srcDType.ByteSize()
	if err != nil {
		return 0, err
	}

	// Pass 0: global max M (a plain chunked scan; no block alignment
	// needed). It shares the run's passScratch with the block passes
	// below; the passes run sequentially, so the shared buffers never
	// overlap.
	M, err := streamComputeMaxAbsScale(r, offset, srcDType, numElems, chunkElems, sc, prog)
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
	// The per-pass buffers come from the run's passScratch (see
	// passScratch): allocated once per run, sized to the max window, and
	// reused across tensors - no per-tensor or per-chunk allocation.
	// Memory bound (memory rule): rawBuf/rawBuf2 (the chunk's raw bytes,
	// the second is the prefetch's double buffer) and fbuf (the chunk's
	// f32 decode) scale with the chunk; scales holds one E4M3 byte per
	// block in a full chunk (chunk/16 bytes) and packed holds the chunk's
	// packed output ((chunk+1)/2 bytes). All are bounded by the chunk,
	// never the tensor. qs is the per-worker quantized-value scratch for
	// pass 2: parallelism-many slices, each at most nvfp4Block uint8,
	// bounded by the CPU count times the block size, never the tensor.
	rawBuf := sc.raw1[:chunk*int64(elemSize)]
	rawBuf2 := sc.raw2[:chunk*int64(elemSize)]
	fbuf := sc.fbuf[:int(chunk)]
	scales := sc.nvfp4Scales[:chunk/nvfp4Block]
	packed := sc.nvfp4Packed[:(chunk+1)/2]
	// qs gives each fan-out worker its own quantized-value scratch (at
	// most nvfp4Block uint8) so concurrent blocks never share a buffer.
	// Sized parallelism(full-chunk block count): bounded by the CPU count
	// (never the chunk or tensor), and parallelism is monotone, so the
	// scratch's slice (sized for the max window) covers every smaller
	// final chunk.
	qs := sc.nvfp4Qs[:max(1, parallelism(int(chunk/nvfp4Block)))]

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

	// forEachBlock reads the source in `chunk`-element chunks (whole
	// blocks) and invokes fn once per chunk with the decoded elements
	// (at most chunk; only the tensor's final chunk may be short). The fn
	// fans the per-block work out across cores (mapStrided) and writes the
	// chunk's output in one call. It is used by passes 1 and 2, so each
	// pass re-reads and recomputes rather than reusing the previous
	// pass's values.
	//
	// The chunk loop double-buffers the raw input (see
	// streamComputeMaxAbsScale): the first chunk is read synchronously,
	// each later chunk is prefetched into rawBuf2 while the previous
	// chunk's blocks are computed, so the next chunk's read latency
	// overlaps this chunk's CPU work. The prefetch changes only timing -
	// the bytes read and the result are identical to the serial loop.
	forEachBlock := func(fn func(fl []float32) error) error {
		// The first chunk has no predecessor to overlap with, so it is
		// read synchronously; every later chunk arrives via the previous
		// iteration's prefetch (see below), already in rawBuf. A
		// zero-element tensor issues no read at all (the loop below never
		// runs), matching the serial path.
		if numElems > 0 {
			firstN := chunk
			if numElems < firstN {
				firstN = numElems
			}
			if _, err := r.ReadAt(rawBuf[:firstN*int64(elemSize)], offset); err != nil {
				return err
			}
		}
		var start int64
		for start < numElems {
			n := chunk
			if numElems-start < n {
				n = numElems - start
			}
			chunkRaw := rawBuf[:n*int64(elemSize)]
			// Prefetch the next chunk into rawBuf2 before this chunk's
			// blocks are computed, so its read latency overlaps this
			// chunk's CPU work. The last chunk has no successor, so no
			// prefetch is started for it.
			var nextErr chan error
			if start+n < numElems {
				nextN := chunk
				if rem := numElems - start - n; rem < nextN {
					nextN = rem
				}
				nextOff := offset + (start+n)*int64(elemSize)
				nextBuf := rawBuf2
				nextErr = make(chan error, 1)
				go func() {
					_, e := r.ReadAt(nextBuf[:nextN*int64(elemSize)], nextOff)
					nextErr <- e
				}()
			}
			fl, err := toFloat32SliceInto(fbuf, chunkRaw, srcDType)
			if err != nil {
				return err
			}
			if err := fn(fl); err != nil {
				return err
			}
			// Progress: this chunk's source bytes went through the pass's
			// compute and write (the chunk's output was written inside fn).
			if prog != nil {
				prog.Add(n * int64(elemSize))
			}
			// Join the prefetch before the next chunk: a failed read fails
			// the run here (at this chunk boundary, one chunk later than
			// the serial path, with the same ReadAt error). Then swap the
			// buffers so the prefetched chunk becomes the current one.
			if nextErr != nil {
				if err := <-nextErr; err != nil {
					return err
				}
				rawBuf, rawBuf2 = rawBuf2, rawBuf
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

	// Pass 1: the 4-byte LE F32 global scale (written once), then one
	// E4M3 byte per block, fanned out across the chunk's blocks into the
	// chunk-sized scales buffer and written once per chunk before the data
	// (both scale siblings precede the owner in the file).
	var gs [4]byte
	binary.LittleEndian.PutUint32(gs[:], math.Float32bits(alpha))
	if _, err := w.Write(gs[:]); err != nil {
		return 0, err
	}
	if err := forEachBlock(func(fl []float32) error {
		numBlocks := (len(fl) + nvfp4Block - 1) / nvfp4Block
		mapStrided(numBlocks, func(b int) {
			lo := b * nvfp4Block
			cnt := nvfp4Block
			if len(fl)-lo < cnt {
				cnt = len(fl) - lo
			}
			scales[b] = scaleByte(fl[lo : lo+cnt])
		})
		_, err := w.Write(scales[:numBlocks])
		return err
	}); err != nil {
		return 0, err
	}

	// Pass 2 (data): recompute each block's scale byte, decode it, and
	// quantize the elements to E2M1 at the step alpha*s, packing two per
	// byte. The per-block quantize+pack is fanned out across the chunk's
	// blocks (mapStrided): each worker writes its block's packed bytes
	// into a disjoint sub-range of the chunk-sized packed buffer, using
	// its own qs scratch (qs[b%k] for k workers - each worker owns one
	// residue class mod k, so no buffer is shared). The write is one per
	// chunk. A full 16-element block packs to exactly 8 bytes, so block
	// b's sub-range starts at byte b*8; the partial final block packs to
	// (cnt+1)/2 bytes and its trailing pad nibble is zeroed by
	// packNibblesInto's prefix zeroing.
	if err := forEachBlock(func(fl []float32) error {
		numBlocks := (len(fl) + nvfp4Block - 1) / nvfp4Block
		k := parallelism(numBlocks)
		mapStrided(numBlocks, func(b int) {
			lo := b * nvfp4Block
			cnt := nvfp4Block
			if len(fl)-lo < cnt {
				cnt = len(fl) - lo
			}
			vals := fl[lo : lo+cnt]
			step := alpha * f8E4M3ToF32(scaleByte(vals))
			qv := qs[b%k][:cnt]
			for i, v := range vals {
				if step == 0 {
					qv[i] = 0
				} else {
					qv[i] = f32ToE2M1(v / step)
				}
			}
			// packNibblesInto zero-prefixes its dst, so the reused
			// packed sub-range cannot leak stale nibbles.
			packNibblesInto(packed[b*nvfp4Block/2:], qv)
		})
		// The chunk's packed output is (len(fl)+1)/2 bytes: each full
		// block contributes 8 and the partial final block (cnt+1)/2.
		_, err := w.Write(packed[:(len(fl)+1)/2])
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
// order (no bit-reversal permutation), and the /16 = 1/sqrt(256) makes
// the transform orthonormal (H·Hᵀ = 256·I, so (H/16)·(H/16)ᵀ = I).
// The /16 normalization is folded into the final stage (s = 128) as
// (a,b) := (a+b)*0.0625, (a-b)*0.0625: for every f32, x*0.0625 and x/16
// are bit-identical - both are the correctly rounded exact rescale by
// 2^-4, subnormals and Inf/NaN included - so the folded stage changes no
// output bits and the separate pass of 256 divisions is gone. Every
// operation is an f32 add, subtract, or multiply-by-a-power-of-two - no
// rounding hazard for representable values - so the transform is
// deterministic: the same input always yields bit-identical output, which
// is what makes the multi-pass streaming conversion below reproducible.
func hadamard256(buf []float32) {
	for s := 1; s < convrotGroup/2; s <<= 1 {
		for i := 0; i < convrotGroup; i += 2 * s {
			for j := 0; j < s; j++ {
				a, b := buf[i+j], buf[i+j+s]
				buf[i+j], buf[i+j+s] = a+b, a-b
			}
		}
	}
	// Final stage (s = 128): the orthonormalizing /16 is folded in as
	// ×0.0625 (bit-identical for every f32, see above), so the transform
	// finishes in one pass with no division.
	s := convrotGroup / 2
	for i := 0; i < convrotGroup; i += 2 * s {
		for j := 0; j < s; j++ {
			a, b := buf[i+j], buf[i+j+s]
			buf[i+j], buf[i+j+s] = (a+b)*0.0625, (a-b)*0.0625
		}
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
// rescan, quantize). Raw reads are windowed: instead of one 256-byte
// ReadAt per group (about 4M syscalls per GB per pass, and there are
// three), the source is read in windows of chunkElems elements (rounded
// up to whole 256-element groups, at least one group) and each group is
// served from the window when in range. Each loaded window's groups are
// then decoded and Hadamard-rotated concurrently (mapStrided: disjoint
// winBuf sub-slices read, disjoint per-group f32 buffers written) - the
// rotation (1024 flops per group) dominates the pass, while the row-max
// scan and the int8 quantize are ~2 ops per element - and the rotated
// values stay in the window's group buffers until the next window load,
// so the sweeps read rotated values straight from the buffers: a row
// whose groups fit in one window is rotated once per pass, not once per
// sweep.
//
// Window reads are prefetched like the per-element passes' chunk reads
// (the S10 double-buffer pattern): the raw window is double-buffered, and
// each window's raw bytes are read in a goroutine while the previous
// window is being rotated and walked - but only when the next window
// load is known with certainty (nextWindowLoad), so the ReadAt sequence
// (offsets, byte counts) is exactly the serial one, only timed earlier.
// The window state (the two raw buffers plus the window's rotated groups)
// comes from the run's passScratch and is the only state that scales
// with the chunk (O(chunk), like every other pass); the rest is the
// current row's max - nothing grows with the tensor.
//
// w must be positioned at the ".scale" region: pass 1 writes the per-row
// scales in row order, then pass 2 writes the quantized data - matching
// the header layout, which emits the row-scale sibling before the owner.
// numElems must be a multiple of 256 (planTensor skips anything else);
// zero-element tensors are handled up front: they write exactly the
// planned row scales (one per row, int8Scale(0)) and no data.
func streamConvRot(r io.ReaderAt, w io.Writer, offset int64, srcDType DType, shape []int64, numElems int64, chunkElems int, sc *passScratch, prog *Progress) (int, error) {
	elemSize, err := srcDType.ByteSize()
	if err != nil {
		return 0, err
	}
	// rotateWindow decodes each group in a parallel worker that cannot
	// return an error, so the source dtype is validated up front: only
	// the four float dtypes decode. The plan only ever routes those
	// here (planTensor's convertible check); a direct call with
	// anything else fails before any read, not mid-rotation.
	switch srcDType {
	case DTypeF16, DTypeBF16, DTypeF32, DTypeF64:
	default:
		return 0, fmt.Errorf("convrot: unsupported source dtype %q", srcDType)
	}

	// Zero-element tensors: no data to read, no groups to rotate, and the
	// row-width check below would divide by zero (a 1-D tensor's row width
	// is numElems itself). Write exactly the sibling bytes planSiblings
	// planned - one F32 scale per row, with a 1-D tensor counting as one
	// row and a 2-D (or wider) tensor shape[0] rows - each
	// int8Scale(0) = 1.0, the same value an all-zero row would produce.
	// The output header's data_offsets were planned from this same
	// formula, so the written bytes must match the planned sibling bytes
	// exactly, or every tensor after this one in the file shifts by the
	// difference (silent corruption).
	if numElems == 0 {
		rows := int64(1)
		if len(shape) >= 2 {
			rows = shape[0]
		}
		var scaleBuf [4]byte
		binary.LittleEndian.PutUint32(scaleBuf[:], math.Float32bits(int8Scale(0)))
		for i := int64(0); i < rows; i++ {
			if _, err := w.Write(scaleBuf[:]); err != nil {
				return 0, err
			}
		}
		return int(rows), nil
	}

	// Zero-element tensors: no data to read, no groups to rotate, and the
	// row-width check below would divide by zero (a 1-D tensor's row width
	// is numElems itself). Write exactly the sibling bytes planSiblings
	// planned - one F32 scale per row, with a 1-D tensor counting as one
	// row and a 2-D (or wider) tensor shape[0] rows - each
	// int8Scale(0) = 1.0, the same value an all-zero row would produce.
	// The output header's data_offsets were planned from this same
	// formula, so the written bytes must match the planned sibling bytes
	// exactly, or every tensor after this one in the file shifts by the
	// difference (silent corruption).
	if numElems == 0 {
		rows := int64(1)
		if len(shape) >= 2 {
			rows = shape[0]
		}
		var scaleBuf [4]byte
		binary.LittleEndian.PutUint32(scaleBuf[:], math.Float32bits(int8Scale(0)))
		for i := int64(0); i < rows; i++ {
			if _, err := w.Write(scaleBuf[:]); err != nil {
				return 0, err
			}
		}
		return int(rows), nil
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
	// G is the total number of rotation groups in the tensor (numElems is
	// a multiple of convrotGroup - planTensor skips anything else).
	G := numElems / convrotGroup

	// Windowed reads: the raw read unit is a window of whole rotation
	// groups, not a single group. winElems is the caller's chunkElems
	// rounded up to the next multiple of convrotGroup (a value below one
	// group becomes one group - the same clamp as the mxfp4/nvfp4 block
	// rounding), so a group never straddles a window boundary and
	// chunkElems drives the reads like in every other pass.
	winElems := int64(convrotGroup)
	if c := int64(chunkElems); c > winElems {
		winElems = (c + convrotGroup - 1) / convrotGroup * convrotGroup
	}
	// Memory bound: the two raw window buffers (bufA/bufB - the second is
	// the prefetch's double buffer, +1 raw window buffer over the serial
	// path) and groups (one 256-element f32 buffer per group in the
	// window, holding each group's rotated values) are the only state
	// that scales with the chunk (O(chunk), like the other passes' decode
	// scratch); they come from the run's passScratch (allocated once per
	// run, sized to chunkElems - see passScratch), never per tensor,
	// and nothing here grows with the tensor or the file.
	bufA := sc.raw1[:winElems*int64(elemSize)]
	bufB := sc.raw2[:winElems*int64(elemSize)]
	groups := sc.convrotGroups[:int(winElems/convrotGroup)]

	// Window pipeline (the S10 double-buffer pattern for windows): cur
	// holds the current window's raw bytes, other is the free buffer the
	// prefetch reads into, and (prefetchG, prefetchCh) the one in-flight
	// prefetch, if any. winStart is the first group index in the current
	// window; winEnd the first past it (exclusive); -1/-1 = empty. The
	// last window of a pass may be shorter than winElems: it is clamped
	// to the tensor's end so a read never crosses into the next tensor's
	// bytes or past EOF.
	cur, other := bufA, bufB
	var winStart, winEnd int64 = -1, -1
	var prefetchG int64 = -1
	var prefetchCh chan error

	// startPrefetch begins reading the window at group g into other, in
	// a goroutine, to overlap with the current window's compute. It is
	// called only when the next loadWindow is known with certainty to be
	// for g (each call site computes it with nextWindowLoad from the
	// sweep's end and the current window), so every prefetch is consumed
	// by exactly that loadWindow: the ReadAt sequence (offsets, byte
	// counts) is exactly the one the serial code issues, only timed
	// earlier. By the same argument at most one prefetch is in flight at
	// a time, and other is always the free buffer when it is.
	startPrefetch := func(g int64) {
		n := winElems
		if rem := numElems - g*convrotGroup; rem < n {
			n = rem
		}
		prefetchG = g
		prefetchCh = make(chan error, 1)
		buf := other // captured: other only changes when the prefetch is joined
		go func() {
			_, e := r.ReadAt(buf[:n*int64(elemSize)], offset+g*convrotGroup*int64(elemSize))
			prefetchCh <- e
		}()
	}

	// loadWindow makes the window at group g the current one: it consumes
	// the pending prefetch for g when there is one (the common case - it
	// was started while the previous window was being computed) or reads
	// synchronously into cur otherwise. A failed prefetch fails the run
	// here (at this window boundary, one window later than the serial
	// path, with the same ReadAt error).
	loadWindow := func(g int64) error {
		if prefetchG == g {
			if err := <-prefetchCh; err != nil {
				return err
			}
			n := winElems
			if rem := numElems - g*convrotGroup; rem < n {
				n = rem
			}
			cur, other = other, cur
			winStart, winEnd = g, g+n/convrotGroup
			prefetchG = -1
			prefetchCh = nil
			return nil
		}
		n := winElems
		if rem := numElems - g*convrotGroup; rem < n {
			n = rem
		}
		if _, err := r.ReadAt(cur[:n*int64(elemSize)], offset+g*convrotGroup*int64(elemSize)); err != nil {
			return err
		}
		winStart, winEnd = g, g+n/convrotGroup
		return nil
	}

	// rotateWindow decodes and rotates the current window's groups
	// (winStart..winEnd) in parallel: worker gi decodes the window's
	// group gi from cur into groups[gi] and rotates it in place -
	// disjoint cur sub-slices read, disjoint groups slots written, no
	// shared state. It is called only right after a loadWindow, so a
	// window's groups are rotated exactly once per load; the rotated
	// values then stay in groups until the next loadWindow overwrites
	// them, which is what lets a row's rescan and quantize sweeps share
	// one rotation.
	rotateWindow := func() {
		es := int64(elemSize)
		mapStrided(int(winEnd-winStart), func(gi int) {
			base := int64(gi) * convrotGroup * es
			decodeFloat32Into(groups[gi][:], cur[base:base+convrotGroup*es], srcDType, convrotGroup)
			hadamard256(groups[gi][:])
		})
	}

	// nextWindowLoad is the prefetch decision for a sweep: the sweep
	// covers groups up to sweepEnd (exclusive) and, when hasNextSweep,
	// the next sweep's first wanted group is firstNext. After the
	// current window [winStart, winEnd) the next loadWindow is:
	//   - winEnd, when the sweep extends past the window (the in-sweep
	//     continuation - the next iteration's g == winEnd always loads),
	//   - firstNext, when the sweep ends within the window and the window
	//     does not cover the next sweep's first group (the next-sweep
	//     load - exactly the load condition the sweep's first iteration
	//     applies),
	//   - -1 otherwise (no next load).
	// Each case is the exact load the serial code would issue next, so
	// prefetching it keeps the ReadAt sequence unchanged (only timed
	// earlier).
	nextWindowLoad := func(winEnd, sweepEnd int64, firstNext int64, hasNextSweep bool) int64 {
		if winEnd < sweepEnd {
			return winEnd
		}
		if hasNextSweep && (firstNext < winStart || firstNext >= winEnd) {
			return firstNext
		}
		return -1
	}

	// Window discipline: group visits are in non-decreasing index order
	// in every sweep: pass 1 walks 0..G-1 flat, and the passes 2/3 row
	// rescans/quantizes walk each row's groups g0..g1 in order with rows
	// ascending and contiguous (row rr+1's g0 = (rr+1)*c/256 is >= row
	// rr's g1 = ((rr+1)*c-1)/256), so within a sweep the window only
	// ever moves forward. The one backward step - a row's rescan end to
	// its quantize start (g1 back to g0) - falls out of the window and
	// triggers a re-read (and re-rotation) of the row's head window; a
	// row whose groups all fit in the rescan's last window instead
	// quantizes straight from the rotated buffers, with no re-read and
	// no re-rotation. The range checks below are correct for any visit
	// order, so monotonicity is a performance property, not a
	// correctness one.

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
	// The walk is window by window (pass 1 always advances to a fresh
	// window), rotating each window's groups before the element walk.
	curRow := int64(0)
	var rowMax float32
	for g := int64(0); g*convrotGroup < numElems; g = winEnd {
		if err := loadWindow(g); err != nil {
			return 0, err
		}
		rotateWindow()
		// Prefetch the next load while this window's element walk runs:
		// the tensor's next window when the tensor extends past this
		// window, else row 0's rescan first load (group 0) when this
		// window does not cover it.
		if next := nextWindowLoad(winEnd, G, 0, true); next >= 0 {
			startPrefetch(next)
		}
		for gi := int64(0); gi < winEnd-winStart; gi++ {
			grp := &groups[gi]
			for j := 0; j < convrotGroup; j++ {
				if rr := ((g+gi)*convrotGroup + int64(j)) / c; rr > curRow {
					if err := writeScale(int8Scale(rowMax)); err != nil {
						return 0, err
					}
					curRow = rr
					rowMax = 0
				}
				rowMax = rowMaxOf(rowMax, grp[j])
			}
		}
		// Progress: this window's source bytes are scanned and its row
		// scales written.
		if prog != nil {
			prog.Add((winEnd - winStart) * convrotGroup * int64(elemSize))
		}
	}
	if err := writeScale(int8Scale(rowMax)); err != nil {
		return 0, err
	}

	// Pass 2 (data): for each row, (a) rescan the groups overlapping the
	// row to recompute its max - recompute, never buffer - then (b)
	// quantize, writing the I8 bytes in element order. The rescan
	// rotates every window the row touches (a row may span more than
	// one window when c exceeds the window's element count); the
	// quantize walk then reads the rotated values straight from groups,
	// re-loading and re-rotating only the windows the rescan's later
	// window loads overwrote (a multi-window row's head). Each write is
	// at most one group (256 bytes), so nothing here grows with the row
	// width or the tensor.
	var qbuf [convrotGroup]byte
	for rr := int64(0); rr < rows; rr++ {
		start, end := rr*c, (rr+1)*c
		g0, g1 := start/convrotGroup, (end-1)/convrotGroup
		// groupRange is the element range of group g that falls inside
		// [start, end), as indices within a group buffer.
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

		// Rescan: max |v| over the row's element ranges, window by
		// window. A window already loaded and rotated by an earlier row
		// is reused as-is - rotating it again would rotate the rotated
		// values.
		var m float32
		for g := g0; g <= g1; {
			if g < winStart || g >= winEnd {
				if err := loadWindow(g); err != nil {
					return 0, err
				}
				rotateWindow()
			}
			// Prefetch the next load while this window's row-max walk runs:
			// the row's next window when the row extends past this window,
			// else the quantize walk's first load (g0) when this window
			// does not cover it.
			if next := nextWindowLoad(winEnd, g1+1, g0, true); next >= 0 {
				startPrefetch(next)
			}
			last := g1 + 1
			if winEnd < last {
				last = winEnd
			}
			for gg := g; gg < last; gg++ {
				l, h := groupRange(gg)
				grp := &groups[gg-winStart]
				for j := l; j < h; j++ {
					m = rowMaxOf(m, grp[j])
				}
			}
			// Progress: this window's source bytes are rescanned for the
			// row's max (the rescan writes nothing).
			if prog != nil {
				prog.Add(int64(last-g) * convrotGroup * int64(elemSize))
			}
			g = last
		}
		scale := int8Scale(m)

		// Quantize: the rescan's last window is still the current one, so
		// its groups are read directly from groups (no re-rotation); a
		// group before winStart was overwritten by the rescan's later
		// window loads and its window is re-read and re-rotated.
		for g := g0; g <= g1; {
			if g < winStart || g >= winEnd {
				if err := loadWindow(g); err != nil {
					return 0, err
				}
				rotateWindow()
			}
			// Prefetch the next load while this window's quantize walk
			// runs: the row's next window when the row extends past this
			// window, else the next row's rescan first load when this
			// window does not cover it (the last row has no successor).
			var next int64 = -1
			if rr+1 < rows {
				next = nextWindowLoad(winEnd, g1+1, (rr+1)*c/convrotGroup, true)
			} else {
				next = nextWindowLoad(winEnd, g1+1, 0, false)
			}
			if next >= 0 {
				startPrefetch(next)
			}
			last := g1 + 1
			if winEnd < last {
				last = winEnd
			}
			for gg := g; gg < last; gg++ {
				l, h := groupRange(gg)
				grp := &groups[gg-winStart]
				for j := l; j < h; j++ {
					qbuf[j-l] = byte(f32ToInt8(grp[j], scale))
				}
				if _, err := w.Write(qbuf[:h-l]); err != nil {
					return 0, err
				}
			}
			// Progress: this window's source bytes are quantized and
			// handed to the writer.
			if prog != nil {
				prog.Add(int64(last-g) * convrotGroup * int64(elemSize))
			}
			g = last
		}
	}
	return int(rows), nil
}
