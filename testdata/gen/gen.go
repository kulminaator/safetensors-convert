// Standalone generator (not part of the main module build) used only to
// produce small synthetic .safetensors fixtures for manual end-to-end
// testing.
//
// Usage:
//
//	go run ./testdata/gen single <outFile>   // one .safetensors, 3 float tensors
//	go run ./testdata/gen multi <outDir>     // 2-shard model dir + index JSON
//	go run ./testdata/gen single256 <outFile> // one BF16 [2,128] tensor (256 elems, outlier)
//	go run ./testdata/gen multirot <outDir>   // 2-shard model dir with 256-elem tensors
//	go run ./testdata/gen qwenlike <outFile>  // one BF16 file named like a small Qwen model
//	go run ./testdata/gen empty <outFile>     // one BF16 file with zero-element tensors
//
// Tensor values are fixed below so tests can spot-check converted output
// against known inputs.
package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
)

func f32ToBF16(f float32) uint16 {
	bits := math.Float32bits(f)
	roundBias := uint32(0x7FFF) + ((bits >> 16) & 1)
	bits += roundBias
	return uint16(bits >> 16)
}

// naive fp32->fp16 for test-data generation only (doesn't need to be
// bulletproof, just needs to be a plausible fp16 payload).
func f32ToF16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int32((bits>>23)&0xFF) - 127 + 15
	mant := bits & 0x7FFFFF
	if exp <= 0 {
		return sign
	}
	if exp >= 0x1F {
		return sign | 0x7C00
	}
	return sign | uint16(exp)<<10 | uint16(mant>>13)
}

func packF16(vals ...float32) []byte {
	out := make([]byte, len(vals)*2)
	for i, v := range vals {
		binary.LittleEndian.PutUint16(out[i*2:], f32ToF16(v))
	}
	return out
}

func packBF16(vals ...float32) []byte {
	out := make([]byte, len(vals)*2)
	for i, v := range vals {
		binary.LittleEndian.PutUint16(out[i*2:], f32ToBF16(v))
	}
	return out
}

func packI32(vals ...int32) []byte {
	out := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(out[i*4:], uint32(v))
	}
	return out
}

// fixtureTensor is one entry of a synthetic safetensors file: name, dtype,
// preformatted JSON shape (e.g. "[2,4]"), and the full little-endian payload.
type fixtureTensor struct {
	name  string
	dtype string
	shape string
	data  []byte
}

func upProjTensor() fixtureTensor {
	return fixtureTensor{
		name:  "model.layers.0.mlp.up_proj.weight",
		dtype: "BF16",
		shape: "[2,4]",
		data:  packBF16(0.1, -0.2, 0.3, -0.4, 1.5, -1.5, 3.25, -3.25),
	}
}

func qProjTensor() fixtureTensor {
	return fixtureTensor{
		name:  "model.layers.0.self_attn.q_proj.weight",
		dtype: "F16",
		shape: "[2,2]",
		data:  packF16(2.0, -2.0, 0.5, -0.5),
	}
}

func normTensor() fixtureTensor {
	return fixtureTensor{
		name:  "model.layers.0.input_layernorm.weight",
		dtype: "F16",
		shape: "[4]",
		data:  packF16(1.0, 1.0, 1.0, 1.0),
	}
}

func positionIDsTensor() fixtureTensor {
	return fixtureTensor{
		name:  "model.layers.0.mlp.position_ids",
		dtype: "I32",
		shape: "[4]",
		data:  packI32(0, 1, 2, 3),
	}
}

// rotWeightTensor is a BF16 [2,128] tensor (256 elements - a multiple of
// the ConvRot rotation group size of 256, which the 8/4/4-element
// single/multi fixtures can never be).
//
// Values (deterministic):
//   - row 0: 0.01 everywhere except index 64 = 85.0 - a strong outlier
//     meant to exercise rotation smearing.
//   - row 1: 0.05·(i%7-3) for i in 0..127 - small structured values in
//     [-0.15, 0.15].
func rotWeightTensor() fixtureTensor {
	vals := make([]float32, 0, 256)
	for i := 0; i < 128; i++ {
		v := float32(0.01)
		if i == 64 {
			v = 85.0
		}
		vals = append(vals, v)
	}
	for i := 0; i < 128; i++ {
		vals = append(vals, 0.05*float32(i%7-3))
	}
	return fixtureTensor{
		name:  "rot.weight",
		dtype: "BF16",
		shape: "[2,128]",
		data:  packBF16(vals...),
	}
}

// rotATensor is a BF16 [128,2] tensor (256 elements, 128 rows of width 2):
// rotation groups of 256 straddle 128 rows, exercising cross-row group
// handling.
//
// Values (deterministic): row i, column j = 0.02·(i%5-2) + 0.1·j for
// i in 0..127, j in 0..1 (range [-0.4, 0.5]); index [37,1] = -42.0 as an
// outlier that falls inside the second half of the rotation group.
func rotATensor() fixtureTensor {
	vals := make([]float32, 0, 256)
	for i := 0; i < 128; i++ {
		for j := 0; j < 2; j++ {
			v := 0.02*float32(i%5-2) + 0.1*float32(j)
			if i == 37 && j == 1 {
				v = -42.0
			}
			vals = append(vals, v)
		}
	}
	return fixtureTensor{
		name:  "rot.a",
		dtype: "BF16",
		shape: "[128,2]",
		data:  packBF16(vals...),
	}
}

// rotBTensor is an F16 [2,128] tensor (256 elements, 2 rows) with a
// pattern different from rotWeightTensor's, plus one outlier.
//
// Values (deterministic):
//   - row 0: 0.03·(i%5-2) for i in 0..127 (range [-0.06, 0.06]).
//   - row 1: 0.04·(i%11-5) for i in 0..127 (range [-0.2, 0.2]), except
//     index 90 = 61.0 (an outlier).
func rotBTensor() fixtureTensor {
	vals := make([]float32, 0, 256)
	for i := 0; i < 128; i++ {
		vals = append(vals, 0.03*float32(i%5-2))
	}
	for i := 0; i < 128; i++ {
		v := 0.04 * float32(i%11-5)
		if i == 90 {
			v = 61.0
		}
		vals = append(vals, v)
	}
	return fixtureTensor{
		name:  "rot.b",
		dtype: "F16",
		shape: "[2,128]",
		data:  packF16(vals...),
	}
}

// buildHeader renders the safetensors header JSON for tensors, in slice
// order, with data offsets relative to the start of the data block.
func buildHeader(tensors []fixtureTensor) string {
	var b strings.Builder
	b.WriteByte('{')
	off := 0
	for i, t := range tensors {
		if i > 0 {
			b.WriteByte(',')
		}
		fmt.Fprintf(&b, "%q:{\"dtype\":%q,\"shape\":%s,\"data_offsets\":[%d,%d]}",
			t.name, t.dtype, t.shape, off, off+len(t.data))
		off += len(t.data)
	}
	b.WriteByte('}')
	return b.String()
}

// writeSafetensors writes a self-contained .safetensors file: 8-byte
// little-endian header length, header JSON, then tensor payloads in the
// same order as the header. If pad is true, the header is space-padded so
// the data block starts on an 8-byte boundary, like the reference
// implementation (real HF shards are padded).
func writeSafetensors(path string, tensors []fixtureTensor, pad bool) {
	header := buildHeader(tensors)
	if pad {
		if p := (8 - (len(header)+8)%8) % 8; p > 0 {
			header += strings.Repeat(" ", p)
		}
	}

	f, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(header)))
	if _, err := f.Write(lenBuf[:]); err != nil {
		f.Close()
		panic(err)
	}
	if _, err := f.WriteString(header); err != nil {
		f.Close()
		panic(err)
	}
	for _, t := range tensors {
		if _, err := f.Write(t.data); err != nil {
			f.Close()
			panic(err)
		}
	}
	if err := f.Close(); err != nil {
		panic(err)
	}
	fmt.Println("wrote", path)
}

func writeSingle(outPath string) {
	writeSafetensors(outPath, []fixtureTensor{
		upProjTensor(),
		qProjTensor(),
		normTensor(),
	}, false)
}

// shard is a shard filename plus the tensors it holds, in file order.
type shard struct {
	name    string
	tensors []fixtureTensor
}

func writeSingle256(outPath string) {
	writeSafetensors(outPath, []fixtureTensor{rotWeightTensor()}, false)
}

// emptyBigTensor is a BF16 [256] tensor (one ConvRot rotation group) with
// a deterministic value pattern: 0.05*(i%7-3) (range [-0.15, 0.15]) plus an
// outlier at index 100 = 33.0 that sets the row scale.
func emptyBigTensor() fixtureTensor {
	vals := make([]float32, 256)
	for i := range vals {
		vals[i] = 0.05 * float32(i%7-3)
	}
	vals[100] = 33.0
	return fixtureTensor{name: "big.w", dtype: "BF16", shape: "[256]", data: packBF16(vals...)}
}

// emptySmallTensor is a BF16 [16] tensor (not a multiple of the ConvRot
// group size of 256, so it exercises the convrot skip): 0.1*i for
// i in 0..15.
func emptySmallTensor() fixtureTensor {
	vals := make([]float32, 16)
	for i := range vals {
		vals[i] = 0.1 * float32(i)
	}
	return fixtureTensor{name: "small.w", dtype: "BF16", shape: "[16]", data: packBF16(vals...)}
}

// writeEmpty writes a single BF16 file whose header order is: empty2d.w
// [0,8] (zero elements), big.w [256] (a ConvRot group, fixed values),
// empty1d.w [0] (zero elements), small.w [16] (not a 256-multiple).
// It is the fixture for the zero-element convrot regression test.
func writeEmpty(outPath string) {
	writeSafetensors(outPath, []fixtureTensor{
		{name: "empty2d.w", dtype: "BF16", shape: "[0,8]", data: nil},
		emptyBigTensor(),
		{name: "empty1d.w", dtype: "BF16", shape: "[0]", data: nil},
		emptySmallTensor(),
	}, false)
}

// qwenLikeTensor is a small BF16 tensor with a deterministic value
// pattern (0.05*(i%7-3), range [-0.15, 0.15]); n is the element count.
// The e2e tests assert dtypes and passthrough byte identity, not
// converted values, so the pattern just needs to be fixed and non-zero.
func qwenLikeTensor(name, shape string, n int) fixtureTensor {
	vals := make([]float32, n)
	for i := range vals {
		vals[i] = 0.05 * float32(i%7-3)
	}
	return fixtureTensor{name: name, dtype: "BF16", shape: shape, data: packBF16(vals...)}
}

// writeQwenLike writes a single-shard file whose tensor names follow the
// bundled Qwen models' naming (embedding, layernorms, q/k/v/o_proj,
// q/k_norm, mlp projections, final norm, lm_head) plus one I32 buffer to
// keep exercising the non-float passthrough. It is the fixture for the
// default-precision-policy e2e test.
func writeQwenLike(outPath string) {
	writeSafetensors(outPath, []fixtureTensor{
		qwenLikeTensor("model.embed_tokens.weight", "[64,32]", 64*32),
		qwenLikeTensor("model.layers.0.input_layernorm.weight", "[32]", 32),
		qwenLikeTensor("model.layers.0.self_attn.q_proj.weight", "[32,16]", 32*16),
		qwenLikeTensor("model.layers.0.self_attn.k_proj.weight", "[32,16]", 32*16),
		qwenLikeTensor("model.layers.0.self_attn.v_proj.weight", "[32,16]", 32*16),
		qwenLikeTensor("model.layers.0.self_attn.o_proj.weight", "[32,16]", 32*16),
		qwenLikeTensor("model.layers.0.self_attn.q_norm.weight", "[32]", 32),
		qwenLikeTensor("model.layers.0.self_attn.k_norm.weight", "[32]", 32),
		qwenLikeTensor("model.layers.0.mlp.gate_proj.weight", "[32,16]", 32*16),
		qwenLikeTensor("model.layers.0.mlp.up_proj.weight", "[32,16]", 32*16),
		qwenLikeTensor("model.layers.0.mlp.down_proj.weight", "[16,32]", 16*32),
		qwenLikeTensor("model.layers.0.post_attention_layernorm.weight", "[32]", 32),
		qwenLikeTensor("model.norm.weight", "[32]", 32),
		qwenLikeTensor("lm_head.weight", "[16,8]", 16*8),
		{name: "model.layers.0.mlp.position_ids", dtype: "I32", shape: "[4]", data: packI32(0, 1, 2, 3)},
	}, false)
}

// writeMultiRot writes a 2-shard model directory whose tensors are
// 256-element multiples (for the ConvRot e2e tests): shard 1 holds
// rot.a (BF16 [128,2], rotation groups straddle rows), shard 2 holds
// rot.b (F16 [2,128], outlier).
func writeMultiRot(outDir string) {
	const (
		shard1Name = "model.safetensors-00001-of-00002.safetensors"
		shard2Name = "model.safetensors-00002-of-00002.safetensors"
	)
	shards := []shard{
		{shard1Name, []fixtureTensor{rotATensor()}},
		{shard2Name, []fixtureTensor{rotBTensor()}},
	}

	if err := os.MkdirAll(outDir, 0o755); err != nil {
		panic(err)
	}

	totalSize := 0
	for _, s := range shards {
		writeSafetensors(filepath.Join(outDir, s.name), s.tensors, true)
		for _, t := range s.tensors {
			totalSize += len(t.data)
		}
	}
	writeIndex(filepath.Join(outDir, "model.safetensors.index.json"), shards, totalSize)
}

// writeMulti writes a 2-shard model directory: two self-contained
// .safetensors shards plus model.safetensors.index.json mapping every
// tensor to its shard. Shard 2 carries an I32 buffer to exercise the
// non-float passthrough path.
func writeMulti(outDir string) {
	const (
		shard1Name = "model.safetensors-00001-of-00002.safetensors"
		shard2Name = "model.safetensors-00002-of-00002.safetensors"
	)
	shards := []shard{
		{shard1Name, []fixtureTensor{upProjTensor(), qProjTensor()}},
		{shard2Name, []fixtureTensor{normTensor(), positionIDsTensor()}},
	}

	if err := os.MkdirAll(outDir, 0o755); err != nil {
		panic(err)
	}

	totalSize := 0
	for _, s := range shards {
		writeSafetensors(filepath.Join(outDir, s.name), s.tensors, true)
		for _, t := range s.tensors {
			totalSize += len(t.data)
		}
	}
	writeIndex(filepath.Join(outDir, "model.safetensors.index.json"), shards, totalSize)
}

// writeIndex writes the HF index file. weight_map entries follow global
// tensor order (shard order, then in-shard header order); total_size is
// the sum of all tensor data bytes.
func writeIndex(path string, shards []shard, totalSize int) {
	var b strings.Builder
	b.WriteString("{\n")
	b.WriteString("  \"metadata\": {\n")
	fmt.Fprintf(&b, "    \"total_size\": %d\n", totalSize)
	b.WriteString("  },\n")
	n := 0
	for _, s := range shards {
		n += len(s.tensors)
	}
	pairs := make([][2]string, 0, n)
	for _, s := range shards {
		for _, t := range s.tensors {
			pairs = append(pairs, [2]string{t.name, s.name})
		}
	}
	b.WriteString("  \"weight_map\": {\n")
	for i, p := range pairs {
		if i > 0 {
			b.WriteString(",\n")
		}
		fmt.Fprintf(&b, "    %q: %q", p[0], p[1])
	}
	b.WriteString("\n  }\n")
	b.WriteString("}\n")

	if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
		panic(err)
	}
	fmt.Println("wrote", path)
}

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "usage: gen single <outFile> | gen multi <outDir> | gen single256 <outFile> | gen multirot <outDir> | gen qwenlike <outFile> | gen empty <outFile>")
		os.Exit(1)
	}
	switch os.Args[1] {
	case "single":
		writeSingle(os.Args[2])
	case "multi":
		writeMulti(os.Args[2])
	case "single256":
		writeSingle256(os.Args[2])
	case "multirot":
		writeMultiRot(os.Args[2])
	case "qwenlike":
		writeQwenLike(os.Args[2])
	case "empty":
		writeEmpty(os.Args[2])
	default:
		fmt.Fprintf(os.Stderr, "unknown mode %q\n", os.Args[1])
		os.Exit(1)
	}
}
