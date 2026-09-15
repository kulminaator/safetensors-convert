// Micro-benchmarks for the hot per-element dtype conversions and the
// 256-group Hadamard rotation (plan "superfly", Step 1 baselines).
//
// Each benchmark times a tight loop over a preallocated 64KB input slice,
// so ns/op divided by the element count is the per-element cost of the
// current implementation - the number the Phase 2 rewrites are measured
// against. Inputs are filled once, outside the timed loop, with fixed-seed
// PRNGs, so the value-class mix (normal/subnormal/NaN/Inf patterns for the
// 16-bit decoders) is stable across runs. BenchmarkHadamard256 is the
// exception: one op is one 256-element group rotation, cycling through the
// 64 groups in the slice (the rotation is in place and orthonormal, so
// values stay bounded across iterations). BenchmarkPackNibbles times
// packNibblesInto with a preallocated destination - the form the streaming
// passes call - so the measured unit is the per-nibble packing cost, not
// the one-shot wrapper's allocation.
package stconv

import (
	"math/rand"
	"testing"
)

// benchBytes is the per-op input size for every benchmark: 64KB.
const benchBytes = 64 * 1024

// benchSink* are package-level sinks so the compiler cannot optimize away
// the benchmark loops: each benchmark reads one output element into a sink
// after the timed section.
var (
	benchSinkF32 float32
	benchSinkU8  uint8
	benchSinkI8  int8
)

// benchHalfBits returns a 64KB slice of random 16-bit patterns (fixed
// seed): the f16/bf16 decode benchmarks' input.
func benchHalfBits() []uint16 {
	rnd := rand.New(rand.NewSource(1))
	bits := make([]uint16, benchBytes/2)
	for i := range bits {
		bits[i] = uint16(rnd.Uint32())
	}
	return bits
}

// benchF32s returns a 64KB slice of pseudo-random values in [-1,1) (fixed
// seed): the f32 encoder benchmarks' input.
func benchF32s() []float32 {
	rnd := rand.New(rand.NewSource(2))
	f := make([]float32, benchBytes/4)
	for i := range f {
		f[i] = rnd.Float32()*2 - 1
	}
	return f
}

// benchNibbles returns a 64KB slice of random 4-bit values (fixed seed):
// the nibble-packing benchmark's input.
func benchNibbles() []uint8 {
	rnd := rand.New(rand.NewSource(3))
	v := make([]uint8, benchBytes)
	for i := range v {
		v[i] = uint8(rnd.Intn(16))
	}
	return v
}

// BenchmarkF16ToF32 decodes 32K f16 bit patterns (64KB) per op.
func BenchmarkF16ToF32(b *testing.B) {
	in := benchHalfBits()
	out := make([]float32, len(in))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j, bits := range in {
			out[j] = f16ToF32(bits)
		}
	}
	benchSinkF32 = out[len(out)-1]
}

// BenchmarkBf16ToF32 decodes 32K bf16 bit patterns (64KB) per op. It is
// the control: bf16->f32 is already a single bit shift.
func BenchmarkBf16ToF32(b *testing.B) {
	in := benchHalfBits()
	out := make([]float32, len(in))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j, bits := range in {
			out[j] = bf16ToF32(bits)
		}
	}
	benchSinkF32 = out[len(out)-1]
}

// BenchmarkF32ToF8E4M3 encodes 16K f32 values (64KB) per op.
func BenchmarkF32ToF8E4M3(b *testing.B) {
	in := benchF32s()
	out := make([]uint8, len(in))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j, f := range in {
			out[j] = f32ToF8E4M3(f)
		}
	}
	benchSinkU8 = out[len(out)-1]
}

// BenchmarkF32ToF8E5M2 encodes 16K f32 values (64KB) per op.
func BenchmarkF32ToF8E5M2(b *testing.B) {
	in := benchF32s()
	out := make([]uint8, len(in))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j, f := range in {
			out[j] = f32ToF8E5M2(f)
		}
	}
	benchSinkU8 = out[len(out)-1]
}

// BenchmarkF32ToInt8 quantizes 16K f32 values (64KB) per op with a fixed
// scale = 1/127 (the int8Scale for maxAbs = 1), so the [-1,1) values never
// hit the clamp path - the common case for real weights.
func BenchmarkF32ToInt8(b *testing.B) {
	in := benchF32s()
	out := make([]int8, len(in))
	const scale = 1.0 / 127.0
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j, f := range in {
			out[j] = f32ToInt8(f, scale)
		}
	}
	benchSinkI8 = out[len(out)-1]
}

// BenchmarkF32ToInt4RNE quantizes 16K f32 values (64KB) per op with a
// fixed scale = 1/7 (the int4Scale for maxAbs = 1). S4 rewrote the
// quantizer from float64 to f32-only bit manipulation; this benchmark
// tracks that change (pre-S4: ~138us/op, S4: ~105us/op on the dev
// machine).
func BenchmarkF32ToInt4RNE(b *testing.B) {
	in := benchF32s()
	out := make([]int8, len(in))
	const scale = 1.0 / 7.0
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j, f := range in {
			out[j] = f32ToInt4RNE(f, scale)
		}
	}
	benchSinkI8 = out[len(out)-1]
}

// BenchmarkHadamard256 rotates one 256-element group per op, cycling
// through the 64 groups in the 64KB slice.
func BenchmarkHadamard256(b *testing.B) {
	buf := benchF32s()
	groups := len(buf) / convrotGroup
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g := i % groups
		hadamard256(buf[g*convrotGroup : (g+1)*convrotGroup])
	}
	benchSinkF32 = buf[0]
}

// BenchmarkPackNibbles packs 64KB of 4-bit values (65536 elements, 32KB
// output) per op. It is the control for the 4-bit packing cost.
func BenchmarkPackNibbles(b *testing.B) {
	in := benchNibbles()
	out := make([]byte, len(in)/2)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		packNibblesInto(out, in)
	}
	benchSinkU8 = out[len(out)-1]
}
