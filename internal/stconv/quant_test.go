package stconv

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// readTensorFloats decodes tensor name's stored bytes in the safetensors
// file at path into f32 values (test-side reference computations).
func readTensorFloats(t *testing.T, path, name string) []float32 {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("opening %s: %v", path, err)
	}
	defer f.Close()
	h, dataStart, err := ReadHeader(f)
	if err != nil {
		t.Fatalf("reading header of %s: %v", path, err)
	}
	for _, e := range h.Tensors {
		if e.Name != name {
			continue
		}
		n := e.Info.DataOffsets[1] - e.Info.DataOffsets[0]
		b := make([]byte, n)
		if _, err := f.ReadAt(b, dataStart+e.Info.DataOffsets[0]); err != nil {
			t.Fatalf("reading %s: %v", name, err)
		}
		fl, err := toFloat32Slice(b, e.Info.DType)
		if err != nil {
			t.Fatalf("decoding %s: %v", name, err)
		}
		return fl
	}
	t.Fatalf("tensor %s not found in %s", name, path)
	return nil
}

// TestHadamard256UnitVectors checks the exact unit-vector cases, where
// every butterfly value stays a power of two and the result must be
// bit-exact: e_0 maps to all 1/16 (column 0 of H is all +1), and e_1
// maps to the alternating +1/16, -1/16 pattern (column 1 of the
// Sylvester H is 1, -1, 1, -1, ...).
func TestHadamard256UnitVectors(t *testing.T) {
	x := make([]float32, 256)
	x[0] = 1
	hadamard256(x)
	for i, v := range x {
		if v != float32(1.0/16) {
			t.Fatalf("e_0: y[%d] = %v (bits %08X), want exactly 1/16", i, v, math.Float32bits(v))
		}
	}

	x = make([]float32, 256)
	x[1] = 1
	hadamard256(x)
	for i, v := range x {
		want := float32(1.0 / 16)
		if i%2 == 1 {
			want = -want
		}
		if v != want {
			t.Fatalf("e_1: y[%d] = %v (bits %08X), want exactly %v", i, v, math.Float32bits(v), want)
		}
	}
}

// TestHadamard256Orthonormal checks the orthonormality of H_256/16:
// (a) a seeded random vector keeps its norm, ||y|| == ||x|| (the plan's
// "||x||^2 * 256" factor belongs to H·Hᵀ = 256·I before the /16);
// (b) H·Hᵀ spot-checks: columns of H/16 (the images of unit vectors) are
// orthonormal, so their pairwise dot products are 1 on the diagonal and 0
// off it.
func TestHadamard256Orthonormal(t *testing.T) {
	rnd := rand.New(rand.NewSource(11))
	x := make([]float32, 256)
	for i := range x {
		x[i] = float32(rnd.Float64()*2 - 1)
	}
	norm2 := func(v []float32) float64 {
		var s float64
		for _, f := range v {
			s += float64(f) * float64(f)
		}
		return s
	}
	y := append([]float32(nil), x...)
	hadamard256(y)
	nx, ny := norm2(x), norm2(y)
	if rel := math.Abs(ny-nx) / nx; rel > 1e-5 {
		t.Errorf("||y||^2 = %v, ||x||^2 = %v: relative difference %v exceeds 1e-5", ny, nx, rel)
	}

	dot := func(a, b []float32) float64 {
		var s float64
		for i := range a {
			s += float64(a[i]) * float64(b[i])
		}
		return s
	}
	cols := make([][]float32, 256)
	for i := range cols {
		cols[i] = make([]float32, 256)
		cols[i][i] = 1
		hadamard256(cols[i])
	}
	for _, p := range [][2]int{{0, 0}, {1, 1}, {63, 63}, {64, 64}, {255, 255}, {0, 1}, {0, 64}, {63, 64}} {
		want := 0.0
		if p[0] == p[1] {
			want = 1.0
		}
		if d := dot(cols[p[0]], cols[p[1]]); math.Abs(d-want) > 1e-5 {
			t.Errorf("dot(col %d, col %d) = %v, want %v (H·Hᵀ spot check)", p[0], p[1], d, want)
		}
	}
}

// TestHadamard256Determinism checks that transforming the same input
// twice yields bit-identical output - the property the multi-pass
// streaming conversion relies on for byte-reproducible files.
func TestHadamard256Determinism(t *testing.T) {
	rnd := rand.New(rand.NewSource(1234))
	x := make([]float32, 256)
	for i := range x {
		x[i] = float32(rnd.Float64()*2 - 1)
	}
	a := append([]float32(nil), x...)
	b := append([]float32(nil), x...)
	hadamard256(a)
	hadamard256(b)
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			t.Fatalf("determinism broken at %d: bits %08X vs %08X", i, math.Float32bits(a[i]), math.Float32bits(b[i]))
		}
	}
}

// TestConvertConvRotSingle256 is the end-to-end check on the committed
// testdata/single256 fixture (BF16 [2,128]: row 0 = 0.01 with an 85.0
// outlier at index 64, row 1 = 0.05*(i%7-3)). It verifies the P2 S3
// header layout (row-scale sibling before the owner), exact scale bytes
// (rowMax/127 of the rotated values), the per-element dequantization
// error bound, and the outlier-smearing effect on row 0.
func TestConvertConvRotSingle256(t *testing.T) {
	in := filepath.Join("..", "..", "testdata", "single256", "rot.safetensors")
	outPath := filepath.Join(t.TempDir(), "out.safetensors")
	stats, err := ConvertFile(ConvertOptions{
		InputPath:  in,
		OutputPath: outPath,
		Default:    TargetInt8ConvRot,
		ChunkElems: 64,
	})
	if err != nil {
		t.Fatalf("ConvertFile: %v", err)
	}
	if len(stats) != 1 {
		t.Fatalf("got %d stats, want 1", len(stats))
	}
	if stats[0].Note != "convrot rows=2" {
		t.Errorf("stats[0].Note = %q, want %q", stats[0].Note, "convrot rows=2")
	}

	// Reference: decode the stored BF16 values and rotate once (the
	// fixture is exactly one 256-element group).
	x := readTensorFloats(t, in, "rot.weight")
	if len(x) != 256 {
		t.Fatalf("fixture has %d elements, want 256", len(x))
	}
	rot := append([]float32(nil), x...)
	hadamard256(rot)
	rowMax := [2]float32{}
	for i, v := range rot {
		a := math.Abs(float64(v))
		if a > float64(rowMax[i/128]) {
			rowMax[i/128] = float32(a)
		}
	}
	scale := [2]float32{int8Scale(rowMax[0]), int8Scale(rowMax[1])}

	// Header per the P2 S3 layout: the F32 row-scale sibling (shape [2])
	// precedes the I8 owner.
	out, err := os.Open(outPath)
	if err != nil {
		t.Fatalf("opening output: %v", err)
	}
	defer out.Close()
	outHeader, _, err := ReadHeader(out)
	if err != nil {
		t.Fatalf("reading output header: %v", err)
	}
	checkOutHeader(t, outHeader, []outEntry{
		{"rot.weight.scale", DTypeF32, [2]int64{0, 8}},
		{"rot.weight", DTypeI8, [2]int64{8, 264}},
	})
	for _, e := range outHeader.Tensors {
		if e.Name == "rot.weight.scale" && !reflect.DeepEqual(e.Info.Shape, []int64{2}) {
			t.Errorf("scale shape = %v, want [2]", e.Info.Shape)
		}
	}

	scaleBytes := readTensorBytes(t, outPath, "rot.weight.scale")
	gotScale := [2]float32{}
	for r := 0; r < 2; r++ {
		bits := binary.LittleEndian.Uint32(scaleBytes[r*4:])
		gotScale[r] = math.Float32frombits(bits)
		// Same arithmetic as the conversion pass -> identical bits.
		if math.Float32bits(gotScale[r]) != math.Float32bits(scale[r]) {
			t.Errorf("scale[%d] = %v (bits %08X), want rowMax/127 = %v (bits %08X)",
				r, gotScale[r], bits, scale[r], math.Float32bits(scale[r]))
		}
	}

	// Dequantized values q*scale_r: |deq - rotated(x)| <= scale_r/2 per
	// element (plus a small f32 slack for the non-power-of-two scale).
	data := readTensorBytes(t, outPath, "rot.weight")
	if len(data) != 256 {
		t.Fatalf("owner has %d bytes, want 256", len(data))
	}
	deqRow0 := make([]float32, 128)
	for i := 0; i < 256; i++ {
		r := i / 128
		deq := float32(int8(data[i])) * gotScale[r]
		d := math.Abs(float64(deq) - float64(rot[i]))
		bound := float64(gotScale[r])/2 + 2e-6*math.Max(1, math.Abs(float64(rot[i])))
		if d > bound {
			t.Errorf("elem %d (row %d): |deq - rot| = %v > %v", i, r, d, bound)
		}
		if r == 0 {
			deqRow0[i] = deq
		}
	}

	// Smearing: the 85.0 outlier dominates the raw row 0
	// (max/mean ~ 8500) but must not dominate the rotated row 0
	// (max/mean stays small - a loose bound pins the effect).
	rawAbs, rawSum := 0.0, 0.0
	for i := 0; i < 128; i++ {
		a := math.Abs(float64(x[i]))
		rawAbs = math.Max(rawAbs, a)
		rawSum += a
	}
	if rawAbs/(rawSum/128) < 100 {
		t.Fatalf("fixture row 0 max/mean = %v, outlier no longer dominant - test broken", rawAbs/(rawSum/128))
	}
	rotAbs, rotSum := 0.0, 0.0
	for _, v := range deqRow0 {
		a := math.Abs(float64(v))
		rotAbs = math.Max(rotAbs, a)
		rotSum += a
	}
	if ratio := rotAbs / (rotSum / 128); ratio > 10 {
		t.Errorf("rotated row 0 max/mean = %v: outlier not smeared (want < 10)", ratio)
	}
}

// TestConvertConvRot1D checks a 1-D [256] tensor: one row, one scale,
// and that scale equals the plain per-tensor int8 scale of the rotated
// data (rows=1 makes ConvRot's scale exactly the int8 scale).
func TestConvertConvRot1D(t *testing.T) {
	rnd := rand.New(rand.NewSource(77))
	vals := make([]float32, 256)
	for i := range vals {
		vals[i] = float32(rnd.Float64()*2 - 1)
	}
	vals[100] = 33.0 // outlier, sets the scale
	in := writeF32Shard(t, f32Tensor{name: "w", vals: vals})
	outPath := filepath.Join(t.TempDir(), "out.safetensors")
	stats, err := ConvertFile(ConvertOptions{
		InputPath:  in,
		OutputPath: outPath,
		Default:    TargetInt8ConvRot,
	})
	if err != nil {
		t.Fatalf("ConvertFile: %v", err)
	}
	if stats[0].Note != "convrot rows=1" {
		t.Fatalf("stats[0].Note = %q, want %q", stats[0].Note, "convrot rows=1")
	}

	rot := append([]float32(nil), vals...)
	hadamard256(rot)
	var maxAbs float32
	for _, v := range rot {
		a := math.Abs(float64(v))
		if a > float64(maxAbs) {
			maxAbs = float32(a)
		}
	}
	wantScale := int8Scale(maxAbs) // the per-tensor int8 scale of the rotated data

	out, err := os.Open(outPath)
	if err != nil {
		t.Fatalf("opening output: %v", err)
	}
	defer out.Close()
	outHeader, _, err := ReadHeader(out)
	if err != nil {
		t.Fatalf("reading output header: %v", err)
	}
	checkOutHeader(t, outHeader, []outEntry{
		{"w.scale", DTypeF32, [2]int64{0, 4}},
		{"w", DTypeI8, [2]int64{4, 260}},
	})
	for _, e := range outHeader.Tensors {
		// P2 S3 convention: a 1-element scale vector is a scalar tensor
		// (empty shape), like the int8 ".scale".
		if e.Name == "w.scale" && !reflect.DeepEqual(e.Info.Shape, []int64{}) {
			t.Errorf("scale shape = %v, want [] (scalar)", e.Info.Shape)
		}
	}

	scaleBytes := readTensorBytes(t, outPath, "w.scale")
	if got := math.Float32frombits(binary.LittleEndian.Uint32(scaleBytes)); got != wantScale {
		t.Errorf("scale = %v (bits %08X), want %v (bits %08X)", got, binary.LittleEndian.Uint32(scaleBytes), wantScale, math.Float32bits(wantScale))
	}

	data := readTensorBytes(t, outPath, "w")
	for i, q := range data {
		deq := float32(int8(q)) * wantScale
		d := math.Abs(float64(deq) - float64(rot[i]))
		if d > float64(wantScale)/2+2e-6 {
			t.Errorf("elem %d: |deq - rot| = %v > scale/2", i, d)
		}
	}
}

// TestConvertConvRotDeterministic converts the single256 fixture twice
// and requires byte-identical outputs: the Hadamard is deterministic and
// the multi-pass conversion recomputes (never buffers) everything, so
// reruns must reproduce the exact same file.
func TestConvertConvRotDeterministic(t *testing.T) {
	in := filepath.Join("..", "..", "testdata", "single256", "rot.safetensors")
	var out1, out2 []byte
	for i, dir := range []string{"a", "b"} {
		outPath := filepath.Join(t.TempDir(), dir, "out.safetensors")
		if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
			t.Fatal(err)
		}
		if _, err := ConvertFile(ConvertOptions{
			InputPath:  in,
			OutputPath: outPath,
			Default:    TargetInt8ConvRot,
			ChunkElems: 33, // odd, non-power-of-two chunk size on purpose
		}); err != nil {
			t.Fatalf("ConvertFile %d: %v", i, err)
		}
		b, err := os.ReadFile(outPath)
		if err != nil {
			t.Fatal(err)
		}
		if i == 0 {
			out1 = b
		} else {
			out2 = b
		}
	}
	if !bytes.Equal(out1, out2) {
		t.Fatalf("two conversions of the same input differ (%d vs %d bytes)", len(out1), len(out2))
	}
}

// countingReaderAt wraps an io.ReaderAt, counting ReadAt calls and bytes
// and tracking the farthest byte offset read (to assert reads stay within
// the tensor).
type countingReaderAt struct {
	r      io.ReaderAt
	calls  int
	bytes  int64
	maxEnd int64
}

func (c *countingReaderAt) ReadAt(p []byte, off int64) (int, error) {
	n, err := c.r.ReadAt(p, off)
	c.calls++
	c.bytes += int64(n)
	if end := off + int64(n); end > c.maxEnd {
		c.maxEnd = end
	}
	return n, err
}

// runConvRot runs streamConvRot over vals (presented as an F32 source
// with the given shape) and returns the written bytes (the row scales,
// then the rotated I8 data) and the counting reader.
func runConvRot(t *testing.T, vals []float32, shape []int64, chunkElems int) ([]byte, *countingReaderAt) {
	t.Helper()
	cr := &countingReaderAt{r: bytes.NewReader(f32ToBytes(vals))}
	var w bytes.Buffer
	if _, err := streamConvRot(cr, &w, 0, DTypeF32, shape, int64(len(vals)), chunkElems); err != nil {
		t.Fatalf("streamConvRot: %v", err)
	}
	return w.Bytes(), cr
}

// TestStreamConvRotWindowedReads pins the chunkElems-driven read window
// of streamConvRot: (a) the output bytes are identical for every window
// size - the window changes how the source is read, never what is
// computed - and match the reference (per-group Hadamard, per-row scale
// = rowMax/127 over the rotated values, f32ToInt8 quantization) byte for
// byte; (b) the ReadAt call count drops from one per group visit (3
// sweeps x G groups) to O(G/window) per sweep; (c) no read crosses the
// tensor's end (the window is clamped to numElems).
func TestStreamConvRotWindowedReads(t *testing.T) {
	rnd := rand.New(rand.NewSource(4242))
	const groups = 6 // 1536 elements
	vals := make([]float32, groups*convrotGroup)
	for i := range vals {
		vals[i] = float32(rnd.Float64()*2 - 1)
	}
	vals[1000] = 33.0 // outlier so row scales are non-trivial
	// Shape [12, 128]: rows of 128 elements, so each 256-element group
	// straddles two rows (exercises pass 1's per-element row tracking
	// and pass 2/3's multi-group-per-row ranges).
	shape := []int64{12, 128}
	const c = int64(128) // row width
	rows := int64(len(vals)) / c

	// Reference: rotate every group in place, then per row take the max
	// |v| over the rotated values (scale = rowMax/127) and quantize.
	rot := append([]float32(nil), vals...)
	for g := 0; g < groups; g++ {
		hadamard256(rot[g*convrotGroup : (g+1)*convrotGroup])
	}
	want := make([]byte, 4*int(rows)+len(vals))
	for rr := int64(0); rr < rows; rr++ {
		var m float32
		for j := rr * c; j < (rr+1)*c; j++ {
			v := rot[j]
			if v < 0 {
				v = -v
			}
			if v > m {
				m = v
			}
		}
		scale := int8Scale(m)
		binary.LittleEndian.PutUint32(want[rr*4:], math.Float32bits(scale))
		for j := rr * c; j < (rr+1)*c; j++ {
			want[4*rows+j] = byte(f32ToInt8(rot[j], scale))
		}
	}

	// (a) Byte identity across window sizes, against the reference.
	outSmall, rSmall := runConvRot(t, vals, shape, 256) // window = 1 group
	outMid, rMid := runConvRot(t, vals, shape, 1024)    // window = 4 groups
	outBig, rBig := runConvRot(t, vals, shape, 1<<20)   // window > tensor
	if !bytes.Equal(outMid, want) {
		t.Fatalf("window=1024 output differs from reference (%d vs %d bytes)", len(outMid), len(want))
	}
	if !bytes.Equal(outSmall, outMid) || !bytes.Equal(outBig, outMid) {
		t.Fatalf("outputs differ across window sizes (small=%d mid=%d big=%d bytes)",
			len(outSmall), len(outMid), len(outBig))
	}

	// (c) No read crosses the tensor's end (1536 f32 elements = 6144
	// bytes).
	tensorBytes := int64(len(vals)) * 4
	for name, cr := range map[string]*countingReaderAt{
		"window=256": rSmall, "window=1024": rMid, "window=2^20": rBig,
	} {
		if cr.maxEnd > tensorBytes {
			t.Errorf("%s: read reached byte %d, past the tensor end %d", name, cr.maxEnd, tensorBytes)
		}
	}

	// (b) ReadAt counts: the un-windowed code issues one ReadAt per
	// group visit (6 pass-1 visits + 12 rows x 2 visits = 30, for any
	// chunkElems). With a 1-group window, re-visits of the same group
	// (a row's rescan->quantize pair, and the two rows sharing each
	// straddled group) are served from the window, so each pass reads
	// each group exactly once (6 + 6); a 4-group window covers most of
	// the tensor per read; a window bigger than the tensor is served by
	// a single whole-tensor read shared by all three sweeps.
	if rSmall.calls != 12 {
		t.Errorf("window=256: %d ReadAt calls, want 12 (6 groups x 2 passes)", rSmall.calls)
	}
	if rSmall.bytes != 12288 {
		t.Errorf("window=256: %d bytes read, want 12288", rSmall.bytes)
	}
	// window=1024: pass 1 reads [0,4) and [4,6); passes 2/3 read [0,4)
	// for rows 0-7 and [4,6) for rows 8-11 -> 4 reads of
	// 4096/2048/4096/2048 bytes.
	if rMid.calls != 4 {
		t.Errorf("window=1024: %d ReadAt calls, want 4", rMid.calls)
	}
	if rMid.bytes != 12288 {
		t.Errorf("window=1024: %d bytes read, want 12288", rMid.bytes)
	}
	if rBig.calls != 1 {
		t.Errorf("window=2^20: %d ReadAt calls, want 1 (whole tensor, shared by all sweeps)", rBig.calls)
	}
	if rBig.bytes != 6144 {
		t.Errorf("window=2^20: %d bytes read, want 6144", rBig.bytes)
	}
}

// f32ToBytes encodes vals as little-endian F32 bytes (a test-side source
// tensor payload).
func f32ToBytes(vals []float32) []byte {
	b := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	return b
}

// runMxFP4 runs streamMxFP4 over vals (presented as an F32 source) with
// the given chunk size and returns the concatenated output: the E8M0
// block scales (one byte per 32-element block) followed by the packed
// E2M1 data.
func runMxFP4(t *testing.T, vals []float32, chunkElems int) []byte {
	t.Helper()
	src := bytes.NewReader(f32ToBytes(vals))
	var w bytes.Buffer
	if err := streamMxFP4(src, &w, 0, DTypeF32, int64(len(vals)), chunkElems); err != nil {
		t.Fatalf("streamMxFP4: %v", err)
	}
	return w.Bytes()
}

// runNVFP4 runs streamNVFP4 over vals (presented as an F32 source) with
// the given chunk size and returns the concatenated output - the 4-byte
// LE F32 global scale, the E4M3 block scales (one byte per 16-element
// block), and the packed E2M1 data - plus the returned alpha.
func runNVFP4(t *testing.T, vals []float32, chunkElems int) ([]byte, float32) {
	t.Helper()
	src := bytes.NewReader(f32ToBytes(vals))
	var w bytes.Buffer
	alpha, err := streamNVFP4(src, &w, 0, DTypeF32, int64(len(vals)), chunkElems)
	if err != nil {
		t.Fatalf("streamNVFP4: %v", err)
	}
	return w.Bytes(), alpha
}

// TestStreamNVFP4Pinned pins streamNVFP4's output bytes on hand-computed
// synthetic blocks: the F32 global scale, the E4M3 block scale byte(s),
// and the packed E2M1 data.
func TestStreamNVFP4Pinned(t *testing.T) {
	// 16-block with max 2688 = 6*448: M = 2688 -> alpha = 1 exactly;
	// block scale = f32ToE4M3RNE(2688/(6*1)) = f32ToE4M3RNE(448) = 0x7E;
	// step alpha*s = 448, so the elements 2688, 448*4, 448*1.5, 448*0.5
	// quantize to e2m1(6), e2m1(4), e2m1(1.5), e2m1(0.5) = 0x7, 0x6, 0x3,
	// 0x1 and the rest to 0 -> data bytes 67 13 00...
	blk := make([]float32, 16)
	blk[0], blk[1], blk[2], blk[3] = 2688, 1792, 672, 224
	wantAlpha := make([]byte, 4)
	binary.LittleEndian.PutUint32(wantAlpha, math.Float32bits(1.0))
	want := append(append([]byte{}, wantAlpha...), 0x7E, 0x67, 0x13, 0, 0, 0, 0, 0, 0)
	got, alpha := runNVFP4(t, blk, 16)
	if alpha != 1.0 {
		t.Errorf("alpha = %v (bits %08X), want exactly 1.0", alpha, math.Float32bits(alpha))
	}
	if !bytes.Equal(got, want) {
		t.Errorf("max-2688 block: got % X, want % X", got, want)
	}
	// A chunk smaller than a block (5) is rounded up to 16 -> identical.
	if got, _ := runNVFP4(t, blk, 5); !bytes.Equal(got, want) {
		t.Errorf("max-2688 block chunk=5: got % X, want % X", got, want)
	}

	// All-zero tensor: M = 0 -> alpha = 0, every block scale 0, every
	// element 0 (32 elements = 2 blocks: 4 alpha bytes + 2 scale bytes +
	// 16 data bytes).
	zeros := make([]float32, 32)
	wantZeros := make([]byte, 22)
	gotZ, alphaZ := runNVFP4(t, zeros, 16)
	if alphaZ != 0 {
		t.Errorf("all-zero alpha = %v, want 0", alphaZ)
	}
	if !bytes.Equal(gotZ, wantZeros) {
		t.Errorf("all-zero: got % X, want all-zero 22 bytes", gotZ)
	}

	// 17 ones: M = 1 -> alpha = 1/2688; both blocks (16 + 1) get scale
	// f32ToE4M3RNE(1/(6*alpha)) = f32ToE4M3RNE(448) = 0x7E; step
	// alpha*448 = 1/6, so every element quantizes to e2m1(6) = 0x07:
	// 8 bytes of 0x77 plus one 0x07 (trailing pad nibble 0) = 9 data
	// bytes, 2 scale entries.
	seventeen := make([]float32, 17)
	for i := range seventeen {
		seventeen[i] = 1.0
	}
	got17, alpha17 := runNVFP4(t, seventeen, 16)
	if alpha17 == 0 || alpha17 != float32(1.0/2688) {
		t.Errorf("17-ones alpha = %v (bits %08X), want 1/2688", alpha17, math.Float32bits(alpha17))
	}
	if len(got17) != 4+2+9 {
		t.Fatalf("17-ones output = %d bytes, want 15 (4 alpha + 2 scale + 9 data)", len(got17))
	}
	if !bytes.Equal(got17[4:6], []byte{0x7E, 0x7E}) {
		t.Errorf("17-ones scales = % X, want 7E 7E", got17[4:6])
	}
	// Element 2i goes to the low nibble: 8 bytes of 0x77 (elements
	// 0..15) plus 0x07 (element 16 in the low nibble, pad high nibble 0).
	want17Data := []byte{0x77, 0x77, 0x77, 0x77, 0x77, 0x77, 0x77, 0x77, 0x07}
	if !bytes.Equal(got17[6:], want17Data) {
		t.Errorf("17-ones data = % X, want % X", got17[6:], want17Data)
	}
	if got17[14]&0xF0 != 0 {
		t.Errorf("17-ones final nibble = 0x%02x, want high nibble 0", got17[14])
	}
}

// TestConvertNVFP4E2E runs the full NVFP4 conversion on a seeded random
// tensor: it pins the P2 S3 header layout (the F32 global-scale scalar
// and the F8_E4M3 block-scale sibling before the U8 owner), checks the
// output sizes and the report note, and verifies the RNE properties -
// each scale byte is exactly the RNE E4M3 of blockMax/(6*alpha) (never
// the 0x7F pattern, never above 448), each stored nibble is exactly the
// RNE E2M1 of x/(alpha*s), and the dequantized value |deq - x| stays
// within the E2M1 quantization bound.
func TestConvertNVFP4E2E(t *testing.T) {
	rnd := rand.New(rand.NewSource(1234))
	n := 1001 // 62 full blocks + one 9-element partial block (odd -> pad)
	vals := make([]float32, n)
	var M float32
	for i := range vals {
		// Bounded away from 0 so no block scale flushes to 0.
		vals[i] = float32(0.1 + 0.9*rnd.Float64())
		if vals[i] > M {
			M = vals[i]
		}
	}
	wantAlpha := M / (6 * 448)
	in := writeF32Shard(t, f32Tensor{name: "w", vals: vals})
	outPath := filepath.Join(t.TempDir(), "out.safetensors")
	stats, err := ConvertFile(ConvertOptions{
		InputPath:  in,
		OutputPath: outPath,
		Default:    TargetNVFP4,
		ChunkElems: 17, // not a multiple of 16: exercises the round-up
	})
	if err != nil {
		t.Fatalf("ConvertFile: %v", err)
	}
	if len(stats) != 1 {
		t.Fatalf("got %d stats, want 1", len(stats))
	}
	if want := fmt.Sprintf("nvfp4 scale=%g", wantAlpha); stats[0].Note != want {
		t.Errorf("stat.Note = %q, want %q", stats[0].Note, want)
	}

	out, err := os.Open(outPath)
	if err != nil {
		t.Fatalf("opening output: %v", err)
	}
	defer out.Close()
	outHeader, _, err := ReadHeader(out)
	if err != nil {
		t.Fatalf("reading output header: %v", err)
	}
	// P2 S3 layout: the F32 global-scale scalar (4 bytes) and the
	// F8_E4M3 block-scale sibling (ceil(n/16) = 63 bytes) precede the
	// U8 owner (ceil(n/2) = 501 bytes).
	checkOutHeader(t, outHeader, []outEntry{
		{"w.global_scale", DTypeF32, [2]int64{0, 4}},
		{"w.block_scale", DTypeF8E4M3, [2]int64{4, 67}},
		{"w", DTypeU8, [2]int64{67, 568}},
	})
	for _, e := range outHeader.Tensors {
		switch e.Name {
		case "w.global_scale":
			if len(e.Info.Shape) != 0 {
				t.Errorf("global_scale shape = %v, want []", e.Info.Shape)
			}
		case "w.block_scale":
			if !reflect.DeepEqual(e.Info.Shape, []int64{63}) {
				t.Errorf("block_scale shape = %v, want [63]", e.Info.Shape)
			}
		}
	}

	gs := readTensorBytes(t, outPath, "w.global_scale")
	var gotAlpha float32
	if len(gs) == 4 {
		gotAlpha = math.Float32frombits(binary.LittleEndian.Uint32(gs))
	}
	if gotAlpha != wantAlpha {
		t.Errorf("global scale = %v (bits %08X), want %v (bits %08X)",
			gotAlpha, math.Float32bits(gotAlpha), wantAlpha, math.Float32bits(wantAlpha))
	}

	scales := readTensorBytes(t, outPath, "w.block_scale")
	if len(scales) != 63 {
		t.Fatalf("block_scale has %d bytes, want 63", len(scales))
	}
	data := readTensorBytes(t, outPath, "w")
	if len(data) != 501 {
		t.Fatalf("owner has %d bytes, want 501", len(data))
	}
	nibbles := unpackNibbles(data)
	if len(nibbles) != n+1 { // 1001 is odd -> one pad nibble
		t.Fatalf("decoded %d nibbles, want %d (incl. pad)", len(nibbles), n+1)
	}
	if nibbles[n] != 0 {
		t.Errorf("trailing pad nibble = 0x%02x, want 0x00", nibbles[n])
	}

	// Per block: the stored scale is exactly the RNE E4M3 of
	// blockMax/(6*alpha), is never the 0x7F NaN pattern, and never
	// decodes above 448 (blockMax <= M, so the ratio is <= 448).
	for b := 0; b < 63; b++ {
		lo, hi := b*nvfp4Block, (b+1)*nvfp4Block
		if hi > n {
			hi = n
		}
		var bm float32
		for _, v := range vals[lo:hi] {
			if v > bm {
				bm = v
			}
		}
		want := f32ToE4M3RNE(bm / (6 * wantAlpha))
		if scales[b] != want {
			t.Errorf("block %d: scale 0x%02x, want 0x%02x (blockMax %v)", b, scales[b], want, bm)
		}
		if scales[b] == 0x7F {
			t.Errorf("block %d: scale is the 0x7F NaN pattern", b)
		}
		if s := f8E4M3ToF32(scales[b]); s > 448 {
			t.Errorf("block %d: scale decodes to %v > 448", b, s)
		}
	}

	// RNE property, per element: the stored nibble equals the RNE E2M1 of
	// x/(alpha*s) exactly, and |deq - x| is within the E2M1 quantization
	// bound. The bound is s*alpha (the plan's 0.5*s*alpha is too tight
	// for the non-uniform E2M1 grid: the scale targets the block max to
	// ~6, and the top grid step 4->6 has a half-gap of 1.0*s*alpha);
	// the exact nibble match is the real RNE verification.
	bad := 0
	for i, x := range vals {
		s := f8E4M3ToF32(scales[i/nvfp4Block])
		step := wantAlpha * s
		var want uint8
		if step != 0 {
			want = f32ToE2M1(x / step)
		}
		if nibbles[i] != want {
			if bad < 20 {
				t.Errorf("elem %d (x=%v, s=%v): nibble 0x%02x, want 0x%02x", i, x, s, nibbles[i], want)
			}
			bad++
			continue
		}
		deq := e2m1ToF32(nibbles[i]) * s * wantAlpha
		if d := math.Abs(float64(deq) - float64(x)); d > float64(s*wantAlpha)+2e-6 {
			if bad < 40 {
				t.Errorf("elem %d (x=%v, deq=%v): |deq - x| = %v > s*alpha = %v", i, x, deq, d, s*wantAlpha)
			}
			bad++
		}
	}
	if bad > 40 {
		t.Errorf("... and %d more violations", bad-40)
	}
}

// TestStreamMxFP4Pinned pins streamMxFP4's output bytes on hand-computed
// synthetic blocks: the E8M0 scale byte(s) and the packed E2M1 data.
func TestStreamMxFP4Pinned(t *testing.T) {
	// All-ones 32-block: blockMax 1 -> code 125 (s = 0.25); each element
	// 1.0/0.25 = 4 -> e2m1(4) = 0x06, so every data byte is 0x66.
	// (The plan's "0x44" predates the P1 S1 fix: e2m1(4) is nibble 0x06.)
	ones := make([]float32, 32)
	for i := range ones {
		ones[i] = 1.0
	}
	wantOnes := append([]byte{125}, make([]byte, 16)...)
	for i := 1; i < len(wantOnes); i++ {
		wantOnes[i] = 0x66
	}
	if got := runMxFP4(t, ones, 32); !bytes.Equal(got, wantOnes) {
		t.Errorf("all-ones: got % X, want % X", got, wantOnes)
	}
	// A chunk smaller than a block (5) is rounded up to 32 -> identical.
	if got := runMxFP4(t, ones, 5); !bytes.Equal(got, wantOnes) {
		t.Errorf("all-ones chunk=5: got % X, want % X", got, wantOnes)
	}

	// {6, 3, 1.5, 0.5, rest 0}: blockMax 6 -> code 127 (s = 1); q =
	// {6,3,1.5,0.5} -> {0x7,0x5,0x3,0x1,0,...} -> bytes 57 13 00...
	b6 := make([]float32, 32)
	b6[0], b6[1], b6[2], b6[3] = 6, 3, 1.5, 0.5
	wantB6 := append([]byte{127, 0x57, 0x13}, make([]byte, 14)...)
	if got := runMxFP4(t, b6, 32); !bytes.Equal(got, wantB6) {
		t.Errorf("block6: got % X, want % X", got, wantB6)
	}

	// All-zero 32-block: blockMax 0 -> code 0 (s = 0); every q = 0.
	zeros := make([]float32, 32)
	wantZeros := append([]byte{0}, make([]byte, 16)...)
	if got := runMxFP4(t, zeros, 32); !bytes.Equal(got, wantZeros) {
		t.Errorf("zeros: got % X, want % X", got, wantZeros)
	}

	// Tiny block: element 0 = 2^-126 (smallest normal f32), rest 1e-40.
	// blockMax = 2^-126 -> code clamps to 1 (s = 2^-126); q[0] =
	// e2m1(2^-126 / 2^-126) = e2m1(1) = 0x02, the rest (x*2^126 < 0.25)
	// round to 0.
	tiny := make([]float32, 32)
	for i := range tiny {
		tiny[i] = 1e-40
	}
	tiny[0] = float32(math.Ldexp(1, -126)) // 2^-126, smallest normal f32
	wantTiny := append([]byte{1, 0x02}, make([]byte, 15)...)
	if got := runMxFP4(t, tiny, 32); !bytes.Equal(got, wantTiny) {
		t.Errorf("tiny: got % X, want % X", got, wantTiny)
	}

	// 33 elements: 2 blocks (32 + 1). Scales: two code-125 bytes. Data:
	// 16 bytes of 0x66 (full block) + 1 byte 0x06 (the single element
	// 1.0, trailing pad nibble 0) = 17 bytes; the final nibble is 0.
	thirtythree := make([]float32, 33)
	for i := range thirtythree {
		thirtythree[i] = 1.0
	}
	want33 := append([]byte{125, 125}, make([]byte, 17)...)
	for i := 2; i < 18; i++ {
		want33[i] = 0x66
	}
	want33[18] = 0x06
	if got := runMxFP4(t, thirtythree, 32); !bytes.Equal(got, want33) {
		t.Errorf("33 elems: got % X, want % X", got, want33)
	}
	// A non-multiple-of-32 chunk (10 -> rounded up to 32) is identical.
	if got := runMxFP4(t, thirtythree, 10); !bytes.Equal(got, want33) {
		t.Errorf("33 elems chunk=10: got % X, want % X", got, want33)
	}
}

// TestConvertMxFP4E2E runs the full MXFP4 conversion on a seeded random
// tensor: it pins the P2 S3 header layout (the U8 block-scale sibling
// before the U8 owner), checks the output sizes, and verifies the RNE
// property - each stored nibble is exactly the RNE E2M1 of x/s, and the
// dequantized value |deq - x| stays within the E2M1 quantization bound.
func TestConvertMxFP4E2E(t *testing.T) {
	rnd := rand.New(rand.NewSource(99))
	n := 1001 // 31 full blocks + one 9-element partial block (odd -> pad)
	vals := make([]float32, n)
	for i := range vals {
		vals[i] = float32(rnd.Float64()*2 - 1)
	}
	in := writeF32Shard(t, f32Tensor{name: "w", vals: vals})
	outPath := filepath.Join(t.TempDir(), "out.safetensors")
	stats, err := ConvertFile(ConvertOptions{
		InputPath:  in,
		OutputPath: outPath,
		Default:    TargetMxFP4,
		ChunkElems: 17, // not a multiple of 32: exercises the round-up
	})
	if err != nil {
		t.Fatalf("ConvertFile: %v", err)
	}
	if len(stats) != 1 {
		t.Fatalf("got %d stats, want 1", len(stats))
	}

	out, err := os.Open(outPath)
	if err != nil {
		t.Fatalf("opening output: %v", err)
	}
	defer out.Close()
	outHeader, _, err := ReadHeader(out)
	if err != nil {
		t.Fatalf("reading output header: %v", err)
	}
	// P2 S3 layout: the U8 block-scale sibling (ceil(n/32) bytes) precedes
	// the U8 owner (ceil(n/2) bytes).
	checkOutHeader(t, outHeader, []outEntry{
		{"w.block_scale", DTypeU8, [2]int64{0, 32}},
		{"w", DTypeU8, [2]int64{32, 533}},
	})
	for _, e := range outHeader.Tensors {
		if e.Name == "w.block_scale" && !reflect.DeepEqual(e.Info.Shape, []int64{32}) {
			t.Errorf("block_scale shape = %v, want [32]", e.Info.Shape)
		}
	}

	scales := readTensorBytes(t, outPath, "w.block_scale")
	if len(scales) != 32 {
		t.Fatalf("block_scale has %d bytes, want 32", len(scales))
	}
	data := readTensorBytes(t, outPath, "w")
	if len(data) != 501 {
		t.Fatalf("owner has %d bytes, want 501", len(data))
	}
	nibbles := unpackNibbles(data)
	if len(nibbles) != n+1 { // 1001 is odd -> one pad nibble
		t.Fatalf("decoded %d nibbles, want %d (incl. pad)", len(nibbles), n+1)
	}
	if nibbles[n] != 0 {
		t.Errorf("trailing pad nibble = 0x%02x, want 0x00", nibbles[n])
	}

	// RNE property, per element: the stored nibble equals the RNE E2M1 of
	// x/s exactly, and |deq - x| is within the E2M1 quantization bound.
	// The bound is s (the plan's 0.5*s is too tight for the non-uniform
	// E2M1 grid: the max-magnitude element sits in (3,6] where the largest
	// half-gap is 1.0*s); since s is a power of two, deq = g*s is exact.
	bad := 0
	for i, x := range vals {
		s := e8m0Scale(scales[i/mxfp4Block])
		var want uint8
		if s != 0 {
			want = f32ToE2M1(x / s)
		}
		if nibbles[i] != want {
			if bad < 20 {
				t.Errorf("elem %d (x=%v, s=%v): nibble 0x%02x, want 0x%02x", i, x, s, nibbles[i], want)
			}
			bad++
			continue
		}
		deq := e2m1ToF32(nibbles[i]) * s
		if d := math.Abs(float64(deq) - float64(x)); d > float64(s)+2e-6 {
			if bad < 40 {
				t.Errorf("elem %d (x=%v, deq=%v): |deq - x| = %v > s = %v", i, x, deq, d, s)
			}
			bad++
		}
	}
	if bad > 40 {
		t.Errorf("... and %d more violations", bad-40)
	}
}
