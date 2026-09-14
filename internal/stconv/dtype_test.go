package stconv

import (
	"bytes"
	"math"
	"math/rand"
	"testing"
)

func almostEqual(a, b, tol float32) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= tol
}

func TestF16RoundTrip(t *testing.T) {
	cases := []float32{0, 1, -1, 0.5, 2, 65504, -65504, 0.000060976, 3.14159}
	for _, c := range cases {
		got := f16ToF32(f32ToF16(c))
		if !almostEqual(got, c, float32(math.Abs(float64(c)))*0.001+1e-6) {
			t.Errorf("f16 round trip %v -> %v", c, got)
		}
	}
	// overflow -> inf
	if got := f16ToF32(f32ToF16(1e9)); !math.IsInf(float64(got), 1) {
		t.Errorf("expected +Inf for overflow, got %v", got)
	}
}

func TestBF16RoundTrip(t *testing.T) {
	cases := []float32{0, 1, -1, 100.5, 12345.6, -0.001}
	for _, c := range cases {
		got := bf16ToF32(f32ToBF16(c))
		if !almostEqual(got, c, float32(math.Abs(float64(c)))*0.02+1e-6) {
			t.Errorf("bf16 round trip %v -> %v", c, got)
		}
	}
}

func TestF8E4M3Max(t *testing.T) {
	// Known max finite magnitude for e4m3fn is 448.
	got := f8E4M3ToF32(f32ToF8E4M3(448))
	if !almostEqual(got, 448, 0.01) {
		t.Errorf("expected ~448, got %v", got)
	}
	// 1.0 should round-trip exactly (representable).
	if got := f8E4M3ToF32(f32ToF8E4M3(1.0)); got != 1.0 {
		t.Errorf("expected exact 1.0, got %v", got)
	}
	// Values well beyond range should land on the NaN pattern.
	got = f8E4M3ToF32(f32ToF8E4M3(100000))
	if !math.IsNaN(float64(got)) {
		t.Errorf("expected NaN for large overflow, got %v", got)
	}
}

func TestE4M3RNEPinned(t *testing.T) {
	// Pinned values for the RNE e4m3 encoder used by the NVFP4 block-scale
	// path. Note the plan's original "0.015625->0x01" is a typo: 0.015625 =
	// 2^-6 is the smallest NORMAL (0x08); 0x01 is the smallest SUBNORMAL
	// (0.001953125 = 2^-9). Both are pinned here.
	cases := []struct {
		f    float32
		want uint8
	}{
		{0, 0x00},
		{0.001953125, 0x01}, // 2^-9, smallest subnormal
		{0.015625, 0x08},    // 2^-6, smallest normal
		{1.0, 0x38},
		{1.5625, 0x3C}, // midpoint between 1.5 (man 4) and 1.625 (man 5): RNE -> man 4
		{448, 0x7E},    // max finite
		{448.5, 0x7E},  // nearest finite (rounds to 448)
		{512, 0x7F},    // saturate
		{float32(math.Inf(1)), 0x7F},
		{math.Float32frombits(0x7fc00000), 0x7F}, // NaN
	}
	for _, c := range cases {
		if got := f32ToE4M3RNE(c.f); got != c.want {
			t.Errorf("f32ToE4M3RNE(%v) = %02x, want %02x", c.f, got, c.want)
		}
	}
}

func TestE4M3RNEContrastWithHalfAway(t *testing.T) {
	// This contrast documents why two e4m3 encoders exist. At exact mantissa
	// midpoints f32ToF8E4M3 (round-half-away, golden-pinned) and
	// f32ToE4M3RNE (round-to-nearest-even) diverge:
	//   1.3125: mantissa 2.5 -> half-away 3 (0x3B) vs RNE 2 (0x3A)
	//   1.5625: mantissa 4.5 -> half-away 5 (0x3D) vs RNE 4 (0x3C)
	if got := f32ToF8E4M3(1.3125); got != 0x3B {
		t.Errorf("f32ToF8E4M3(1.3125) = %02x, want %02x (half-away)", got, 0x3B)
	}
	if got := f32ToE4M3RNE(1.3125); got != 0x3A {
		t.Errorf("f32ToE4M3RNE(1.3125) = %02x, want %02x (RNE)", got, 0x3A)
	}
	if got := f32ToF8E4M3(1.5625); got != 0x3D {
		t.Errorf("f32ToF8E4M3(1.5625) = %02x, want %02x (half-away)", got, 0x3D)
	}
	if got := f32ToE4M3RNE(1.5625); got != 0x3C {
		t.Errorf("f32ToE4M3RNE(1.5625) = %02x, want %02x (RNE)", got, 0x3C)
	}
}

func TestE4M3RNEAgreesWithHalfAwayOnNonTies(t *testing.T) {
	// On every non-tie input the two encoders must emit identical bytes.
	// (Values like 100 = 1.5625*2^6 or 200 = 1.5625*2^7 ARE exact mantissa
	// midpoints and legitimately differ - see the contrast test above.)
	nonTies := []float32{0.015625, 0.03, 0.5, 0.75, 1.0, 1.1, 2.0, 4.0, 6.0,
		3.33, 5.5, 10.25, 100.1, 200.3, 447.0, 448.0, 448.5, -1.0, -5.5, -12.4, -448}
	for _, f := range nonTies {
		if a, b := f32ToF8E4M3(f), f32ToE4M3RNE(f); a != b {
			t.Errorf("non-tie %v: half-away %02x != RNE %02x", f, a, b)
		}
	}
}

func TestF8E5M2Max(t *testing.T) {
	// Known max finite magnitude for e5m2 is 57344.
	got := f8E5M2ToF32(f32ToF8E5M2(57344))
	if !almostEqual(got, 57344, 1) {
		t.Errorf("expected ~57344, got %v", got)
	}
	if got := f8E5M2ToF32(f32ToF8E5M2(1.0)); got != 1.0 {
		t.Errorf("expected exact 1.0, got %v", got)
	}
	// Overflow -> +Inf (e5m2 supports infinity).
	got = f8E5M2ToF32(f32ToF8E5M2(1e9))
	if !math.IsInf(float64(got), 1) {
		t.Errorf("expected +Inf, got %v", got)
	}
}

func TestInt8QuantizeSymmetric(t *testing.T) {
	scale := int8Scale(10.0)
	if q := f32ToInt8(10.0, scale); q != 127 {
		t.Errorf("expected 127, got %d", q)
	}
	if q := f32ToInt8(-10.0, scale); q != -127 {
		t.Errorf("expected -127, got %d", q)
	}
	if q := f32ToInt8(0, scale); q != 0 {
		t.Errorf("expected 0, got %d", q)
	}
	back := int8ToF32(127, scale)
	if !almostEqual(back, 10.0, 0.01) {
		t.Errorf("dequant mismatch: %v", back)
	}
}

func TestInt4Scale(t *testing.T) {
	cases := []struct {
		maxAbs float32
		want   float32
	}{
		{0, 1},
		{7, 1},
		{21, 3},
	}
	for _, c := range cases {
		if got := int4Scale(c.maxAbs); got != c.want {
			t.Errorf("int4Scale(%v) = %v, want %v", c.maxAbs, got, c.want)
		}
	}
}

func TestF32ToInt4RNE(t *testing.T) {
	// Pinned at scale=1. Note the RNE tie behavior: 2.5 -> 2 (not 3, which
	// would be math.Round's half-away answer); 3.5 -> 4.
	cases := []struct {
		in   float32
		want int8
	}{
		{0, 0},
		{0.5, 0},
		{1.5, 2},
		{2.5, 2},
		{3.5, 4},
		{6.5, 6},
		{7, 7},
		{7.4, 7},
		{7.6, 7}, // RNE would give 8, clamped to 7
		{-0.5, 0},
		{-1.5, -2},
		{-2.5, -2},
		{-3.5, -4},
		{float32(math.Inf(1)), 7},
		{float32(math.Inf(-1)), -7},
		{float32(math.NaN()), 0},
	}
	for _, c := range cases {
		if got := f32ToInt4RNE(c.in, 1); got != c.want {
			t.Errorf("f32ToInt4RNE(%v, 1) = %d, want %d", c.in, got, c.want)
		}
	}
	// scale == 0 -> 0 for any finite input.
	if got := f32ToInt4RNE(3.3, 0); got != 0 {
		t.Errorf("f32ToInt4RNE(3.3, 0) = %d, want 0", got)
	}
	// Non-unit scale: with scale=2, 5.0 -> 2.5 -> RNE 2 (tie to even).
	if got := f32ToInt4RNE(5.0, 2); got != 2 {
		t.Errorf("f32ToInt4RNE(5.0, 2) = %d, want 2", got)
	}
	// RNE contrast vs math.Round: 2.5 quantizes to 2, not 3.
	if got := f32ToInt4RNE(2.5, 1); got != 2 {
		t.Errorf("RNE tie 2.5 = %d, want 2 (half-away would give 3)", got)
	}
}

func TestInt4PackRoundTrip(t *testing.T) {
	// Two's-complement nibbles: -1 -> 0x0F, -7 -> 0x09.
	// pack([-1,-7]) == 0x9F (low nibble 0xF, high nibble 0x9).
	if got := packNibbles([]uint8{0x0F, 0x09}); !bytes.Equal(got, []byte{0x9F}) {
		t.Errorf("pack([-1,-7]) = %x, want 9f", got)
	}
	// pack([1,2,3]) == [0x21, 0x03]: element 2 (value 3) goes to the low
	// nibble of byte 1 (element 2i -> low nibble), with a trailing zero pad
	// in the high nibble. (Matches the pinned pack([A,B,C]) == [BA, 0C].)
	if got := packNibbles([]uint8{0x01, 0x02, 0x03}); !bytes.Equal(got, []byte{0x21, 0x03}) {
		t.Errorf("pack([1,2,3]) = %x, want 21 03", got)
	}
	// int4ToF32 decode of two's-complement nibbles (scale=1).
	if got := int4ToF32(0x0F, 1); got != -1 {
		t.Errorf("int4ToF32(0xF) = %v, want -1", got)
	}
	if got := int4ToF32(0x09, 1); got != -7 {
		t.Errorf("int4ToF32(0x9) = %v, want -7", got)
	}
	if got := int4ToF32(0x07, 1); got != 7 {
		t.Errorf("int4ToF32(0x7) = %v, want 7", got)
	}
	// Full round trip: quantize a set of values at a scale, pack, unpack,
	// decode, and check the dequantized value is within half a step of the
	// original (RNE bound).
	scale := float32(1.5)
	inputs := []float32{0, 1.5, 2.5, -2.5, 6.5, -6.5, 9.9, -9.9, 0.1}
	vals := make([]uint8, len(inputs))
	for i, x := range inputs {
		vals[i] = uint8(f32ToInt4RNE(x, scale)) & 0x0F
	}
	packed := packNibbles(vals)
	unpacked := unpackNibbles(packed)
	for i, x := range inputs {
		q := f32ToInt4RNE(x, scale)
		nib := int32(unpacked[i])
		if nib >= 8 {
			nib -= 16
		}
		if nib != int32(q) {
			t.Errorf("round trip idx %d: unpack %d, want %d", i, nib, q)
		}
		dq := int4ToF32(unpacked[i], scale)
		d := dq - x
		if d < 0 {
			d = -d
		}
		if d > scale/2 {
			t.Errorf("dequant idx %d: |%v - %v| = %v > scale/2 = %v", i, dq, x, d, scale/2)
		}
	}
}

func TestE2M1DecodeGrid(t *testing.T) {
	// Positive grid by code: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (code = e<<1|m).
	want := []float32{0, 0.5, 1, 1.5, 2, 3, 4, 6}
	for code := 0; code < 16; code++ {
		s := float32(1)
		if code&0x08 != 0 {
			s = -1
		}
		if got := e2m1ToF32(uint8(code)); got != s*want[code&0x07] {
			t.Errorf("decode %02x: got %v, want %v", code, got, s*want[code&0x07])
		}
	}
}

func TestE2M1EncodePinned(t *testing.T) {
	cases := []struct {
		f    float32
		want uint8
	}{
		// Grid values map to themselves.
		{0, 0x00}, {0.5, 0x01}, {1, 0x02}, {1.5, 0x03},
		{2, 0x04}, {3, 0x05}, {4, 0x06}, {6, 0x07},
		{-0.5, 0x09}, {-1, 0x0A}, {-1.5, 0x0B}, {-2, 0x0C},
		{-3, 0x0D}, {-4, 0x0E}, {-6, 0x0F},
		// Exact midpoints tie to the even mantissa bit (0, 1, 2, 4).
		{0.25, 0x00}, {0.75, 0x02}, {1.25, 0x02}, {1.75, 0x04},
		{2.5, 0x04}, {3.5, 0x06}, {5, 0x06},
		{-0.25, 0x00}, {-0.75, 0x0A}, {-1.25, 0x0A}, {-1.75, 0x0C},
		{-2.5, 0x0C}, {-3.5, 0x0E}, {-5, 0x0E},
		// Non-tie values.
		{0.1, 0x00}, {0.3, 0x01}, {-0.1, 0x00}, {-0.3, 0x09},
		// Saturation and special values (6 = code 0x07).
		{6.1, 0x07}, {-6.1, 0x0F}, {1e30, 0x07}, {-1e30, 0x0F},
		{float32(math.Inf(1)), 0x07}, {float32(math.Inf(-1)), 0x0F},
		{math.Float32frombits(0x7fc00000), 0x00}, // NaN
	}
	for _, c := range cases {
		if got := f32ToE2M1(c.f); got != c.want {
			t.Errorf("f32ToE2M1(%v) = %02x, want %02x", c.f, got, c.want)
		}
	}
}

// e2m1RNENearest is the independent reference for the round-trip test:
// nearest grid value by distance, ties to the even mantissa bit (even index).
func e2m1RNENearest(f float32) float32 {
	s := float32(1)
	if math.Signbit(float64(f)) {
		s = -1
	}
	af := math.Abs(float64(f))
	if math.IsNaN(float64(f)) {
		return 0
	}
	if af == math.Inf(1) || af > 6 {
		return s * 6
	}
	best := 0
	for i, g := range e2m1Grid {
		d1 := af - float64(g)
		d0 := af - float64(e2m1Grid[best])
		if d1 < 0 {
			d1 = -d1
		}
		if d0 < 0 {
			d0 = -d0
		}
		if d1 < d0 || (d1 == d0 && i%2 == 0) {
			best = i
		}
	}
	return s * e2m1Grid[best]
}

func TestE2M1RoundTrip(t *testing.T) {
	vals := make([]float32, 0, 128)
	for _, g := range e2m1Grid {
		vals = append(vals, g, -g)
	}
	for i := 1; i < 8; i++ { // exact midpoints
		mid := (e2m1Grid[i-1] + e2m1Grid[i]) / 2
		vals = append(vals, mid, -mid)
	}
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 1000; i++ {
		// Uniform in [-7, 7] (covers saturation) plus a few large values.
		vals = append(vals, float32(rng.Float64()*14-7))
		if i%50 == 0 {
			vals = append(vals, float32(rng.Float64())*1e30)
		}
	}
	for _, v := range vals {
		got := e2m1ToF32(f32ToE2M1(v))
		want := e2m1RNENearest(v)
		if got != want {
			t.Errorf("round trip %v: got %v, want %v", v, got, want)
		}
	}
}

func TestE8M0EncodePinned(t *testing.T) {
	cases := []struct {
		m    float32
		code uint8
	}{
		{0, 0},
		{6, 127},                      // s=1, m/s=6
		{7, 128},                      // s=2, m/s=3.5
		{3, 126},                      // s=0.5, m/s=6
		{2.9, 126},                    // s=0.5, m/s=5.8
		{math.Nextafter32(3, 4), 127}, // s=1, m/s=3.0000005 <= 6
		{1, 125},                      // s=0.25, m/s=4
		{1.5, 125},                    // s=0.25, m/s=6
		{1.5000001, 126},              // s=0.5, m/s=3.0000002
		{448, 134},                    // s=128, m/s=3.5
		{math.MaxFloat32, 253},
		{float32(math.Ldexp(1, -126)), 1}, // 2^-126 (smallest normal): lower clamp
		{1e-40, 1},                        // subnormal: lower clamp
		{math.SmallestNonzeroFloat32, 1},  // 2^-149: lower clamp
	}
	for _, c := range cases {
		if got := e8m0Encode(c.m); got != c.code {
			t.Errorf("e8m0Encode(%v) = %d, want %d", c.m, got, c.code)
		}
	}
}

func TestE8M0Scale(t *testing.T) {
	if got := e8m0Scale(0); got != 0 {
		t.Errorf("e8m0Scale(0) = %v, want 0", got)
	}
	if got := e8m0Scale(255); !math.IsInf(float64(got), 1) {
		t.Errorf("e8m0Scale(255) = %v, want +Inf", got)
	}
	// Powers of two decode exactly.
	cases := []struct {
		code uint8
		want float32
	}{
		{127, 1},
		{128, 2},
		{126, 0.5},
		{1, float32(math.Ldexp(1, -126))}, // 2^-126, the smallest positive scale
		{254, float32(math.Ldexp(1, 127))},
	}
	for _, c := range cases {
		if got := e8m0Scale(c.code); got != c.want {
			t.Errorf("e8m0Scale(%d) = %v, want %v", c.code, got, c.want)
		}
	}
}

func TestE8M0Property(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for i := 0; i < 20000; i++ {
		// Log-uniform m: uniform exponent in [-149, 127] (covers subnormals
		// up to MaxFloat32), uniform mantissa in [1, 2).
		e := rng.Intn(277) - 149
		m := float32(math.Ldexp(1+rng.Float64(), e))
		if m == 0 {
			continue
		}
		code := e8m0Encode(m)
		if code == 255 {
			t.Fatalf("e8m0Encode(%v) = 255 (Inf code)", m)
		}
		s := e8m0Scale(code)
		ratio := m / s
		if ratio > 6 {
			t.Fatalf("m=%v code=%d: m/s = %v > 6", m, code, ratio)
		}
		if ratio <= 3 && m >= float32(math.Ldexp(6, -126)) {
			t.Fatalf("m=%v code=%d: m/s = %v in (0,3] with m >= 6·2^-126", m, code, ratio)
		}
	}
}

func TestPackNibbles(t *testing.T) {
	// Even count, little-endian element order: e0 low, e1 high.
	if got := packNibbles([]uint8{0x1, 0x2}); !bytes.Equal(got, []byte{0x21}) {
		t.Errorf("pack([1,2]) = %x, want 21", got)
	}
	// Odd count: trailing zero pad nibble.
	if got := packNibbles([]uint8{0xA, 0xB, 0xC}); !bytes.Equal(got, []byte{0xBA, 0x0C}) {
		t.Errorf("pack([A,B,C]) = %x, want ba 0c", got)
	}
	// "Signed" (two's-complement) values pack as raw 4-bit patterns.
	if got := packNibbles([]uint8{0x0F, 0x00}); !bytes.Equal(got, []byte{0x0F}) {
		t.Errorf("pack([F,0]) = %x, want 0f", got)
	}
	// Empty input.
	if got := packNibbles(nil); len(got) != 0 {
		t.Errorf("pack(nil) = %x, want empty", got)
	}
	// Unpack round-trips even and odd counts; pad nibble is zero.
	for _, n := range []int{0, 1, 2, 5, 8} {
		vals := make([]uint8, n)
		for i := range vals {
			vals[i] = uint8(i*3+1) & 0x0F
		}
		b := packNibbles(vals)
		if len(b) != (n+1)/2 {
			t.Fatalf("pack len for n=%d = %d, want %d", n, len(b), (n+1)/2)
		}
		got := unpackNibbles(b)
		if len(got) != 2*len(b) {
			t.Fatalf("unpack len for n=%d = %d, want %d", n, len(got), 2*len(b))
		}
		for i := 0; i < n; i++ {
			if got[i] != vals[i] {
				t.Errorf("round trip n=%d idx %d: got %x, want %x", n, i, got[i], vals[i])
			}
		}
		if n%2 == 1 && got[n] != 0 {
			t.Errorf("round trip n=%d: pad nibble = %x, want 0", n, got[n])
		}
	}
}

func TestF16Zero(t *testing.T) {
	if bits := f32ToF16(0); bits != 0 {
		t.Errorf("expected 0 bits for +0, got %x", bits)
	}
	if bits := f32ToF16(float32(math.Copysign(0, -1))); bits != 0x8000 {
		t.Errorf("expected sign bit for -0, got %x", bits)
	}
}
