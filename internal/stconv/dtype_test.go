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

// f16ToF32Ref is the pre-S2 float64-arithmetic f16 decoder (math.Pow
// based), kept verbatim as the oracle for
// TestF16ToF32Exhaustive. It must not be updated when f16ToF32 changes.
func f16ToF32Ref(bits uint16) float32 {
	sign := bits >> 15
	exp := (bits >> 10) & 0x1F
	man := bits & 0x3FF

	var val float64
	switch {
	case exp == 0x1F:
		if man == 0 {
			val = math.Inf(1)
		} else {
			val = math.NaN()
		}
	case exp == 0:
		val = float64(man) / 1024 * math.Pow(2, -14)
	default:
		val = (1 + float64(man)/1024) * math.Pow(2, float64(exp)-15)
	}
	if sign == 1 {
		val = -val
	}
	return float32(val)
}

// TestF16ToF32Exhaustive checks the bit-manipulation f16ToF32 against the
// float64-arithmetic reference for all 65536 uint16 patterns. f16->f32 is
// an exact conversion, so non-NaN results must agree bit-for-bit; NaNs are
// compared by class and sign only, because the reference canonicalizes to
// Go's quiet NaN while the bit decoder carries the f16 payload over (the
// plan's "all-ones exponent + nonzero mantissa" pattern).
func TestF16ToF32Exhaustive(t *testing.T) {
	for i := 0; i < 65536; i++ {
		u := uint16(i)
		got := f16ToF32(u)
		want := f16ToF32Ref(u)
		if math.IsNaN(float64(want)) {
			if !math.IsNaN(float64(got)) || math.Signbit(float64(got)) != math.Signbit(float64(want)) {
				t.Fatalf("f16ToF32(%04x) = %08x, want a NaN with sign %v (ref %08x)",
					u, math.Float32bits(got), math.Signbit(float64(want)), math.Float32bits(want))
			}
			continue
		}
		if math.Float32bits(got) != math.Float32bits(want) {
			t.Fatalf("f16ToF32(%04x) = %08x (%v), want %08x (%v)",
				u, math.Float32bits(got), got, math.Float32bits(want), want)
		}
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
	// Values beyond 448 saturate to 448, the max finite value - torch's
	// overflow behavior (e4m3fn has no Inf; the 0x7F NaN pattern is
	// reserved for NaN inputs).
	got = f8E4M3ToF32(f32ToF8E4M3(100000))
	if !almostEqual(got, 448, 0.01) {
		t.Errorf("expected 448 for large overflow, got %v", got)
	}
	// +/-Inf saturate to +/-448, matching torch's cast (pinned against
	// torch 2.14.0+cpu).
	if got := f32ToF8E4M3(float32(math.Inf(1))); got != 0x7E {
		t.Errorf("f32ToF8E4M3(+Inf) = %02x, want 7e", got)
	}
	if got := f32ToF8E4M3(float32(math.Inf(-1))); got != 0xFE {
		t.Errorf("f32ToF8E4M3(-Inf) = %02x, want fe", got)
	}
}

func TestF8E4M3Pinned(t *testing.T) {
	// Pinned values for the e4m3 encoder, verified against torch's
	// float8_e4m3fn cast (torch 2.14.0+cpu): RNE mantissa rounding,
	// overflow and +/-Inf saturate to 448, NaN -> 0x7F. Note 0.015625 =
	// 2^-6 is the smallest NORMAL (0x08); 0x01 is the smallest SUBNORMAL
	// (0.001953125 = 2^-9).
	cases := []struct {
		f    float32
		want uint8
	}{
		{0, 0x00},
		{0.001953125, 0x01},  // 2^-9, smallest subnormal
		{0.0009765625, 0x00}, // half a subnormal step: RNE tie -> 0 (even)
		{0.0029296875, 0x02}, // 1.5*2^-9: RNE tie -> even 2
		{0.013671875, 0x07},  // 7*2^-9
		{0.0146484375, 0x08}, // 7.5*2^-9: RNE tie -> smallest normal
		{0.015625, 0x08},     // 2^-6, smallest normal
		{1.0, 0x38},
		{1.3125, 0x3A}, // midpoint, mantissa 2.5: RNE -> even 2
		{1.5625, 0x3C}, // midpoint, mantissa 4.5: RNE -> even 4
		{448, 0x7E},    // max finite
		{448.5, 0x7E},  // nearest finite (rounds to 448)
		{464, 0x7E},    // 448/480 midpoint: RNE -> 448 (even mantissa)
		{480, 0x7E},    // overflow: saturate to 448
		{512, 0x7E},
		{float32(math.Inf(1)), 0x7E},
		{float32(math.Inf(-1)), 0xFE},
		{math.Float32frombits(0x7fc00000), 0x7F}, // NaN
		{math.Float32frombits(0xffc00000), 0xFF}, // -NaN
	}
	for _, c := range cases {
		if got := f32ToF8E4M3(c.f); got != c.want {
			t.Errorf("f32ToF8E4M3(%v) = %02x, want %02x", c.f, got, c.want)
		}
	}
}

// TestF8E4M3ExhaustiveOracle checks f32ToF8E4M3 against the Frexp-based
// RNE oracle for every f32 magnitude (all 2^24 non-negative bit patterns
// from +0 to the largest finite, both signs) plus the Inf/NaN patterns -
// the whole f32 domain, no sampling.
func TestF8E4M3ExhaustiveOracle(t *testing.T) {
	bads := 0
	check := func(s uint32) {
		f := math.Float32frombits(s)
		if got, want := f32ToF8E4M3(f), f32ToF8E4M3Ref(f); got != want {
			if bads < 20 {
				t.Errorf("f32ToF8E4M3(%08x) = %02x, want %02x (oracle)", s, got, want)
			}
			bads++
		}
	}
	for p := uint32(0); p <= 0x7F7FFFFF; p++ {
		check(p)
		check(p | 0x80000000)
	}
	for _, s := range []uint32{0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000,
		0x7F800001, 0xFF800001, 0x7FFFFFFF, 0xFFFFFFFF} {
		check(s)
	}
	if bads > 0 {
		t.Errorf("%d total mismatches against the oracle", bads)
	}
}

func TestF8E5M2Pinned(t *testing.T) {
	// Pinned values for the e5m2 encoder, verified against torch's
	// float8_e5m2 cast (torch 2.14.0+cpu): RNE mantissa rounding in both
	// the normal and subnormal paths, overflow and +/-Inf -> +/-Inf
	// (0x7C), NaN -> 0x7F. Note 2^-14 is the smallest NORMAL (0x04);
	// 2^-16 is the smallest SUBNORMAL (0x01).
	cases := []struct {
		f    float32
		want uint8
	}{
		{0, 0x00},
		{0.00000762939453125, 0x00}, // 0.5*2^-16, half a subnormal step: RNE tie -> 0 (even)
		{0.0000152587890625, 0x01},  // 2^-16, smallest subnormal
		{0.00002288818359375, 0x02}, // 1.5*2^-16: RNE tie -> even 2
		{0.0000457763671875, 0x03},  // 3*2^-16
		{0.00005340576171875, 0x04}, // 3.5*2^-16: RNE tie -> smallest normal
		{0.00006103515625, 0x04},    // 2^-14, smallest normal
		{1.0, 0x3C},
		{1.125, 0x3C}, // tie 1.0/1.25: RNE -> even man 0
		{1.375, 0x3E}, // tie 1.25/1.5: RNE -> even man 2
		{1.625, 0x3E}, // tie 1.5/1.75: RNE -> even man 2
		{57344, 0x7B}, // max finite
		{57344.0001, 0x7B},
		{61440, 0x7C}, // 57344/Inf tie: RNE -> Inf (man 0 even)
		{65536, 0x7C},
		{100000, 0x7C},
		{float32(math.Inf(1)), 0x7C},
		{float32(math.Inf(-1)), 0xFC},
		{math.Float32frombits(0x7fc00000), 0x7F}, // NaN
		{math.Float32frombits(0xffc00000), 0xFF}, // -NaN
	}
	for _, c := range cases {
		if got := f32ToF8E5M2(c.f); got != c.want {
			t.Errorf("f32ToF8E5M2(%v) = %02x, want %02x", c.f, got, c.want)
		}
	}
}

// TestF8E5M2ExhaustiveOracle checks f32ToF8E5M2 against the Frexp-based
// RNE oracle for every f32 magnitude (all 2^24 non-negative bit patterns
// from +0 to the largest finite, both signs) plus the Inf/NaN patterns -
// the whole f32 domain, no sampling.
func TestF8E5M2ExhaustiveOracle(t *testing.T) {
	bads := 0
	check := func(s uint32) {
		f := math.Float32frombits(s)
		if got, want := f32ToF8E5M2(f), f32ToF8E5M2Ref(f); got != want {
			if bads < 20 {
				t.Errorf("f32ToF8E5M2(%08x) = %02x, want %02x (oracle)", s, got, want)
			}
			bads++
		}
	}
	for p := uint32(0); p <= 0x7F7FFFFF; p++ {
		check(p)
		check(p | 0x80000000)
	}
	for _, s := range []uint32{0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000,
		0x7F800001, 0xFF800001, 0x7FFFFFFF, 0xFFFFFFFF} {
		check(s)
	}
	if bads > 0 {
		t.Errorf("%d total mismatches against the oracle", bads)
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

// rneRoundF64 rounds a non-negative float64 to the nearest integer, ties
// to even (round-to-nearest-even). The caller must pass a value whose
// fractional part is exact (see f32ToF8E4M3Ref for why float64 suffices).
func rneRoundF64(t float64) int32 {
	q := int32(t)
	frac := t - float64(q)
	switch {
	case frac > 0.5:
		q++
	case frac == 0.5 && q%2 == 1:
		q++
	}
	return q
}

// f32ToF8E4M3Ref is the Frexp-based RNE e4m3 oracle for
// TestF32ToF8EncodersOracle and TestF8E4M3ExhaustiveOracle: an independent,
// straightforward implementation of the same function as f32ToF8E4M3 -
// round-to-nearest-even mantissa, overflow and +/-Inf saturating to 448
// (0x7E), NaN -> 0x7F - computed in float64, where an f32 significand is
// exact enough for tie detection. The bit-manipulation encoder must agree
// with it on every f32 input.
func f32ToF8E4M3Ref(f float32) uint8 {
	sign := uint8(0)
	if math.Signbit(float64(f)) {
		sign = 0x80
	}
	af := math.Abs(float64(f))

	if math.IsNaN(float64(f)) {
		return sign | 0x7F
	}
	if math.IsInf(af, 1) {
		return sign | 0x7E // no Inf; saturate to the max finite value
	}
	if af == 0 {
		return sign
	}

	const manBits = 3
	const denom = 1 << manBits // 8
	const bias = 7

	frac, exp := math.Frexp(af)
	m := frac * 2
	e := int32(exp) - 1
	biasedExp := e + bias

	if biasedExp < 1 {
		// Subnormal range; smallest normal is 2^(1-bias) = 2^-6.
		t := af / math.Pow(2, 1-bias) * denom
		man := rneRoundF64(t)
		if man <= 0 {
			return sign
		}
		if man >= denom {
			return sign | (1 << manBits) // rounds up into smallest normal
		}
		return sign | uint8(man)
	}

	manF := rneRoundF64((m - 1) * denom)
	if manF >= denom {
		manF = 0
		biasedExp++
	}
	const maxExp = 0xF // 4 exponent bits, all-ones is usable except mantissa==7
	if biasedExp > maxExp || (biasedExp == maxExp && manF >= 7) {
		// Overflow (or the mantissa whose pattern is reserved for NaN):
		// saturate to 0x7E, the max finite value 448.
		return sign | 0x7E
	}
	return sign | uint8(biasedExp)<<manBits | uint8(manF)
}

// f32ToF8E5M2Ref is the Frexp-based RNE e5m2 oracle for
// TestF32ToF8EncodersOracle and TestF8E5M2ExhaustiveOracle: an independent,
// straightforward implementation of the same function as f32ToF8E5M2 -
// round-to-nearest-even mantissa, NaN -> 0x7F, +/-Inf and overflow ->
// +/-Inf (0x7C) - computed in float64, where an f32 significand is exact
// enough for tie detection. The bit-manipulation encoder must agree with
// it on every f32 input.
func f32ToF8E5M2Ref(f float32) uint8 {
	sign := uint8(0)
	if math.Signbit(float64(f)) {
		sign = 0x80
	}
	af := math.Abs(float64(f))

	if math.IsNaN(float64(f)) {
		return sign | 0x7F // exp=11111, man=11 -> NaN
	}
	if math.IsInf(float64(f), 0) {
		return sign | 0x7C // exp=11111, man=00 -> Inf
	}
	if af == 0 {
		return sign
	}

	const manBits = 2
	const denom = 1 << manBits // 4
	const bias = 15

	frac, exp := math.Frexp(af)
	m := frac * 2
	e := int32(exp) - 1
	biasedExp := e + bias

	if biasedExp < 1 {
		val := af / math.Pow(2, 1-bias) * denom
		man := rneRoundF64(val)
		if man <= 0 {
			return sign
		}
		if man >= denom {
			return sign | (1 << manBits)
		}
		return sign | uint8(man)
	}

	manF := rneRoundF64((m - 1) * denom)
	if manF >= denom {
		manF = 0
		biasedExp++
	}
	if biasedExp >= 0x1F {
		return sign | 0x7C // overflow -> Inf
	}
	return sign | uint8(biasedExp)<<manBits | uint8(manF)
}

// fp8EdgeCorpus builds the structured edge corpus for
// TestF32ToF8EncodersOracle: the subnormal/normal boundary regions of both
// formats, exact mantissa midpoints at every f32 exponent (the RNE
// ties), the saturation boundaries
// (+/-448 for e4m3, +/-57344 for e5m2) and
// their ULP neighbors, zero, f32 denormals, +/-Inf, NaN payloads, the
// 0x7F/Inf-adjacent values, and the extremes.
func fp8EdgeCorpus() []float32 {
	var c []float32
	add := func(f ...float32) { c = append(c, f...) }
	// ulpAround adds x plus n ULPs of f32 in each direction from x.
	ulpAround := func(x float32, n int) {
		dn, up := x, x
		for i := 0; i < n; i++ {
			dn = math.Nextafter32(dn, 0)
			up = math.Nextafter32(up, 1)
			add(dn, up)
		}
		add(x)
	}

	// e4m3: subnormal step 2^-9, smallest normal 2^-6, flush at 2^-10.
	step4 := float32(math.Ldexp(1, -9))
	for k := 0; k <= 16; k++ {
		// Subnormal units k*2^-9 and the RNE ties (k+0.5)*2^-9.
		add(float32(k)*step4, (float32(k)+0.5)*step4)
	}
	ulpAround(float32(math.Ldexp(1, -10)), 4)             // flush boundary
	ulpAround(step4, 4)                                   // 2^-9 +/- a few ULPs
	ulpAround(float32(1.5)*step4, 4)                      // 1.5-unit tie
	ulpAround(float32(math.Ldexp(1, -6)), 4)              // smallest normal
	ulpAround(float32(1.5)*float32(math.Ldexp(1, -6)), 4) // 7.5-unit tie: rounds up into the smallest normal

	// e5m2: subnormal step 2^-16, smallest normal 2^-14, flush at 2^-17.
	step5 := float32(math.Ldexp(1, -16))
	for k := 0; k <= 10; k++ {
		add(float32(k)*step5, (float32(k)+0.5)*step5)
	}
	ulpAround(float32(math.Ldexp(1, -17)), 4)              // flush boundary
	ulpAround(step5, 4)                                    // 2^-16 +/- a few ULPs
	ulpAround(float32(1.5)*step5, 4)                       // 1.5-unit tie
	ulpAround(float32(math.Ldexp(1, -14)), 4)              // smallest normal
	ulpAround(float32(1.5)*float32(math.Ldexp(1, -14)), 4) // 3.5-unit tie: rounds up into the smallest normal

	// Exact mantissa midpoints at every f32 exponent (the RNE ties).
	// e4m3 midpoint at unbiased exponent e, k: (1+(k+0.5)/8)*2^e =
	// (16+2k+1)*2^(e-4); e5m2: (1+(k+0.5)/4)*2^e = (8+2k+1)*2^(e-3).
	// Both are odd*2^int, so exact in f32 across the whole exponent range
	// (below 2^-126 they are exact f32 subnormals).
	for e := -126; e <= 127; e++ {
		for k := 0; k < 8; k++ {
			add(float32(math.Ldexp(float64(16+2*k+1), e-4)))
		}
		for k := 0; k < 4; k++ {
			add(float32(math.Ldexp(float64(8+2*k+1), e-3)))
		}
	}

	// Saturation boundaries, ULP neighbors, and the 0x7F/Inf-adjacent
	// values.
	ulpAround(448, 4) // e4m3 max finite
	ulpAround(-448, 4)
	ulpAround(464, 2) // 448/480 tie -> 0x7E (448)
	ulpAround(-464, 2)
	ulpAround(480, 2) // overflow past the 448/480 tie -> 0x7E
	ulpAround(-480, 2)
	ulpAround(496, 2) // carry boundary -> 0x7E
	ulpAround(-496, 2)
	ulpAround(57344, 4) // e5m2 max finite
	ulpAround(-57344, 4)
	ulpAround(61440, 2) // 57344/65536 tie -> Inf
	ulpAround(-61440, 2)

	// Zero, f32 denormals (all flush to zero in fp8), +/-Inf, NaN payloads,
	// extremes.
	add(0, math.Float32frombits(0x80000000)) // -0
	add(math.SmallestNonzeroFloat32, math.Float32frombits(1<<22),
		math.Float32frombits(0x400000), math.Float32frombits(0x7FFFFF))
	add(-math.SmallestNonzeroFloat32, math.Float32frombits(0x80000001),
		math.Float32frombits(0xC0000000), math.Float32frombits(0xFF7FFFFF))
	add(math.Float32frombits(0x7F800000), math.Float32frombits(0xFF800000)) // +/-Inf
	add(math.Float32frombits(0x7FC00000), math.Float32frombits(0xFFC00000),
		math.Float32frombits(0x7F800001), math.Float32frombits(0xFF800001),
		math.Float32frombits(0x7FFFFFFF), math.Float32frombits(0xFFFFFFFF))
	add(math.MaxFloat32, -math.MaxFloat32, 1e30, -1e30)
	return c
}

// TestF32ToF8EncodersOracle checks the bit-manipulation fp8 encoders
// against the Frexp-based references (f32ToF8E4M3Ref/f32ToF8E5M2Ref) for
// 1,000,000 random f32 bit patterns (fixed seed) plus the structured edge
// corpus from fp8EdgeCorpus. The new encoders must agree with the
// references bit-for-bit on every input.
func TestF32ToF8EncodersOracle(t *testing.T) {
	check := func(f float32) {
		if got, want := f32ToF8E4M3(f), f32ToF8E4M3Ref(f); got != want {
			t.Errorf("f32ToF8E4M3(%08x) = %02x, want %02x (ref)",
				math.Float32bits(f), got, want)
		}
		if got, want := f32ToF8E5M2(f), f32ToF8E5M2Ref(f); got != want {
			t.Errorf("f32ToF8E5M2(%08x) = %02x, want %02x (ref)",
				math.Float32bits(f), got, want)
		}
	}

	rng := rand.New(rand.NewSource(11))
	for i := 0; i < 1000000; i++ {
		check(math.Float32frombits(rng.Uint32()))
	}
	for _, f := range fp8EdgeCorpus() {
		check(f)
	}
}

// f32ToInt8Ref is the RNE int8 oracle for TestInt8Int4QuantizeOracle and
// TestF32ToInt8ExhaustiveOracle: an independent, straightforward
// implementation of the same function as f32ToInt8 - round-to-nearest-
// even on the f32 quotient, clamped to [-127,127], NaN -> 0, +/-Inf ->
// +/-127, scale == 0 -> 0 - via rneRoundF64 on the clamped magnitude with
// the sign re-applied (RNE is sign-symmetric). The float64 implementation
// must agree with it on every input.
func f32ToInt8Ref(f, scale float32) int8 {
	if math.IsNaN(float64(f)) {
		return 0
	}
	if scale == 0 {
		return 0
	}
	t := float64(f / scale)
	if math.IsNaN(t) {
		return 0
	}
	// Clamp the float before rneRoundF64: |t| <= 127 keeps the int32
	// conversion in range and maps +/-Inf to the bounds, as the
	// production code does.
	if t > 127 {
		t = 127
	}
	if t < -127 {
		t = -127
	}
	q := rneRoundF64(math.Abs(t))
	if math.Signbit(t) {
		q = -q
	}
	return int8(q)
}

// f32ToInt4RNERef is the pre-S4 float64-based int4 RNE quantizer, kept
// verbatim as the oracle for TestInt8Int4QuantizeOracle. It must not be
// updated when f32ToInt4RNE changes.
func f32ToInt4RNERef(f, scale float32) int8 {
	if scale == 0 {
		return 0
	}
	if math.IsNaN(float64(f)) {
		return 0
	}
	if math.IsInf(float64(f), 0) {
		if math.Signbit(float64(f)) {
			return -7
		}
		return 7
	}
	v := f / scale
	av := math.Abs(float64(v))
	// Clamp in float space before int32(av): the conversion is
	// implementation-defined once av exceeds the int32 range, and the
	// post-conversion q > 7 clamp below would never see the overflow.
	// av < 8 keeps the int32 path in range; the q > 7 clamp still handles
	// the 7.5 -> 8 RNE tie. Mirrors f32ToInt8's float-space clamping.
	if !(av < 8) {
		if math.Signbit(float64(f)) {
			return -7
		}
		return 7
	}
	q := int32(av)
	frac := av - float64(q)
	switch {
	case frac > 0.5:
		q++
	case frac == 0.5 && q%2 == 1:
		q++
	}
	if q > 7 {
		q = 7
	}
	if math.Signbit(float64(f)) {
		q = -q
	}
	return int8(q)
}

// TestInt8Int4QuantizeOracle checks f32ToInt8/f32ToInt4RNE against the
// float64 references for 1,000,000 random (f, scale) pairs - f an
// arbitrary f32 bit pattern, scale = +/-2^e * u with e in -60..60 and u in
// [0.5, 1), so the quotients land at every exponent and on the clamp
// paths - plus an explicit edge table: rounding midpoints with their f32
// ULP neighbors, the float-space clamp boundaries, quotients at the 2^k
// +- 0.5 corners, exact midpoints through non-unit scales, and the
// special values (NaN, +/-Inf, scale 0, scale NaN, Inf/Inf quotients).
// The f32ToInt4RNE comparison is the load-bearing one: f32ToInt4RNE is
// the S4 f32-only rewrite and must agree with the float64 reference on
// every pair. The f32ToInt8 comparison guards that f32ToInt8 (kept as the
// float64 implementation for speed) keeps agreeing with the reference.
// Both implementations must agree with the references on every pair.
func TestInt8Int4QuantizeOracle(t *testing.T) {
	check := func(f, scale float32) {
		if got, want := f32ToInt8(f, scale), f32ToInt8Ref(f, scale); got != want {
			t.Errorf("f32ToInt8(%g, %g) = %d, want %d (ref)", f, scale, got, want)
		}
		if got, want := f32ToInt4RNE(f, scale), f32ToInt4RNERef(f, scale); got != want {
			t.Errorf("f32ToInt4RNE(%g, %g) = %d, want %d (ref)", f, scale, got, want)
		}
	}

	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 1000000; i++ {
		f := math.Float32frombits(rng.Uint32())
		e := rng.Intn(121) - 60
		scale := float32(math.Pow(2, float64(e))) * (0.5 + rng.Float32()*0.5)
		if rng.Intn(2) == 1 {
			scale = -scale
		}
		check(f, scale)
	}

	// Rounding midpoints and clamp boundaries at scale = 1, each with its
	// two f32 ULP neighbors, so the tie handling is pinned at the
	// boundary. (0x3EFFFFFF = 0.5-2^-25 is the f32 just below 0.5: the
	// corner where q+0.5 is an inexact f32 tie that rounds up to 1.0.)
	midpoints := []float32{
		126.5, 127, 127.4999, 7.5, 7.0, 6.5,
		0.5, 1.5, 2.5, 3.5, 4.5, 5.5,
	}
	for _, m := range midpoints {
		for _, s := range []float32{m, -m} {
			b := math.Float32bits(s)
			check(math.Float32frombits(b-1), 1)
			check(s, 1)
			check(math.Float32frombits(b+1), 1)
		}
	}

	// Quotients at the 2^k +- 0.5 corners (k = -4..7): q = 2^k - 0.5 and
	// the f32 just below 2^k are where the q +- 0.5 sum's exactness
	// argument is stressed; 2^k +- 0.5 are the tie points. At k = 7 these
	// double as the 127.5 clamp-boundary midpoints.
	for k := -4; k <= 7; k++ {
		pb := uint32(k+127) << 23 // bit pattern of 2^k
		check(math.Float32frombits(pb)-0.5, 1)
		check(math.Float32frombits(pb-1), 1)
		check(math.Float32frombits(pb), 1)
		check(math.Float32frombits(pb)+0.5, 1)
	}

	// Exact midpoints reached through non-unit scales: the quotient is the
	// f32 division result, not a literal.
	for _, ps := range []struct{ f, scale float32 }{
		{253, 2}, {255, 2}, {254, 2}, {15, 2}, {13, 2}, {5, 2}, {3, 2}, {1, 2},
		{253, -2}, {-255, 2}, {1023, 8}, {7.5, 1.5},
	} {
		check(ps.f, ps.scale)
	}

	// Special values: NaN, +/-Inf, scale 0, scale NaN, and Inf/Inf
	// quotients (the one corner where a NaN quotient reaches the
	// float->int conversion; both sides do the same conversion).
	for _, ps := range []struct{ f, scale float32 }{
		{math.Float32frombits(0x7fc00000), 1},
		{math.Float32frombits(0xffc00000), 1},
		{math.Float32frombits(0x7fc00000), 0},
		{math.Float32frombits(0xffc00000), 0},
		{float32(math.Inf(1)), 1},
		{float32(math.Inf(-1)), 1},
		{float32(math.Inf(1)), 0.25},
		{float32(math.Inf(-1)), 0.25},
		{float32(math.Inf(1)), 0},
		{float32(math.Inf(-1)), 0},
		{float32(math.Inf(1)), float32(math.Inf(1))},
		{float32(math.Inf(-1)), float32(math.Inf(-1))},
		{1.5, 0},
		{100, 0},
		{1.5, math.Float32frombits(0x7fc00000)},
		{float32(math.Inf(1)), math.Float32frombits(0x7fc00000)},
		{126.5, math.Float32frombits(0x7fc00000)},
	} {
		check(ps.f, ps.scale)
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

// TestF32ToInt8SpecialValues pins the file-header NaN/Inf convention for
// f32ToInt8 (B4): NaN quantizes to 0 explicitly (int8(NaN) is
// implementation-defined per the Go spec - 0 on amd64 today, but not
// guaranteed), +/-Inf clamp to +/-127.
func TestF32ToInt8SpecialValues(t *testing.T) {
	scale := int8Scale(10.0)
	cases := []struct {
		in   float32
		want int8
	}{
		{math.Float32frombits(0x7fc00000), 0}, // NaN
		{float32(math.Inf(1)), 127},
		{float32(math.Inf(-1)), -127},
	}
	for _, c := range cases {
		if got := f32ToInt8(c.in, scale); got != c.want {
			t.Errorf("f32ToInt8(%v, %v) = %d, want %d", c.in, scale, got, c.want)
		}
	}
}

func TestF32ToInt8Pinned(t *testing.T) {
	// Pinned at scale = 1 (quotient == value), torch-verified tie
	// behavior: torch.round is RNE - 2.5 -> 2 (not math.Round's half-away
	// 3), 3.5 -> 4, -2.5 -> -2, -3.5 -> -4.
	cases := []struct {
		in   float32
		want int8
	}{
		{0, 0},
		{0.5, 0}, // RNE tie -> even 0
		{1.5, 2}, // RNE tie -> even 2
		{2.5, 2}, // RNE tie -> even 2 (half-away would give 3)
		{3.5, 4}, // RNE tie -> even 4
		{5.5, 6}, // RNE tie -> even 6
		{6.5, 6}, // RNE tie -> even 6
		{126.5, 126},
		{127, 127},
		{127.5, 127}, // RNE would give 128, clamped to 127
		{-0.5, 0},
		{-1.5, -2},
		{-2.5, -2}, // RNE tie -> even -2
		{-3.5, -4}, // RNE tie -> even -4
		{float32(math.Inf(1)), 127},
		{float32(math.Inf(-1)), -127},
		{math.Float32frombits(0x7fc00000), 0}, // NaN
	}
	for _, c := range cases {
		if got := f32ToInt8(c.in, 1); got != c.want {
			t.Errorf("f32ToInt8(%v, 1) = %d, want %d", c.in, got, c.want)
		}
	}
	// scale == 0 -> 0 for any finite input.
	if got := f32ToInt8(3.3, 0); got != 0 {
		t.Errorf("f32ToInt8(3.3, 0) = %d, want 0", got)
	}
	// Non-unit scale: with scale = 2, 5.0 -> quotient 2.5 -> RNE tie to
	// the even integer 2.
	if got := f32ToInt8(5.0, 2); got != 2 {
		t.Errorf("f32ToInt8(5.0, 2) = %d, want 2", got)
	}
	// Out-of-range quotients must clamp in float space (no out-of-range
	// float->int conversion).
	for _, c := range []struct {
		f, scale float32
		want     int8
	}{
		{1e30, 1, 127},
		{-1e30, 1, -127},
	} {
		if got := f32ToInt8(c.f, c.scale); got != c.want {
			t.Errorf("f32ToInt8(%g, %g) = %d, want %d", c.f, c.scale, got, c.want)
		}
	}
}

// TestF32ToInt8ExhaustiveOracle checks f32ToInt8 against the RNE oracle
// for every f32 bit pattern (all 2^24 non-negative magnitudes, both
// signs) at scale = 1.0 - where the quotient is exactly f (IEEE division
// by 1.0 is exact), so the whole quotient domain is covered, no sampling.
func TestF32ToInt8ExhaustiveOracle(t *testing.T) {
	bads := 0
	check := func(s uint32) {
		f := math.Float32frombits(s)
		if got, want := f32ToInt8(f, 1), f32ToInt8Ref(f, 1); got != want {
			if bads < 20 {
				t.Errorf("f32ToInt8(%08x, 1) = %d, want %d (oracle)", s, got, want)
			}
			bads++
		}
	}
	for p := uint32(0); p <= 0x7F7FFFFF; p++ {
		check(p)
		check(p | 0x80000000)
	}
	for _, s := range []uint32{0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000,
		0x7F800001, 0xFF800001, 0x7FFFFFFF, 0xFFFFFFFF} {
		check(s)
	}
	if bads > 0 {
		t.Errorf("%d total mismatches against the oracle", bads)
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
		{7.5, 7}, // RNE tie to 8, clamped to 7
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
	// Out-of-int32-range quotients must clamp to ±7 in float space (B2):
	// pre-fix, int32(av) overflowed and f32ToInt4RNE(1e20, 1e-3) returned
	// 1 instead of 7.
	ovf := []struct {
		f, scale float32
		want     int8
	}{
		{1e20, 1e-3, 7},
		{-1e20, 1e-3, -7},
		{1e30, 1, 7},
	}
	for _, c := range ovf {
		if got := f32ToInt4RNE(c.f, c.scale); got != c.want {
			t.Errorf("f32ToInt4RNE(%g, %g) = %d, want %d", c.f, c.scale, got, c.want)
		}
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

// TestPackNibblesInto pins the pack-into-scratch variant the streaming
// passes use: same layout as packNibbles, into the caller's buffer, and -
// crucially - no stale-nibble leakage when the buffer is reused across
// packs of different lengths (an odd pack's pad nibble must be zero even
// if the buffer previously held a longer pack).
func TestPackNibblesInto(t *testing.T) {
	dst := make([]byte, 8)
	// Even count: same bytes as the allocating wrapper, aliasing dst.
	got := packNibblesInto(dst, []uint8{0x1, 0x2})
	if !bytes.Equal(got, []byte{0x21}) {
		t.Errorf("into([1,2]) = %x, want 21", got)
	}
	if &got[0] != &dst[0] {
		t.Fatal("result does not alias the caller's dst")
	}
	// Odd count: trailing zero pad nibble, identical to the wrapper.
	if got := packNibblesInto(dst, []uint8{0xA, 0xB, 0xC}); !bytes.Equal(got, []byte{0xBA, 0x0C}) {
		t.Errorf("into([A,B,C]) = %x, want ba 0c", got)
	}
	// Reuse: dst[0] still holds the 0xBA/0x21 leftovers from the packs
	// above; packing one value writes only the low nibble of byte 0, so
	// the high nibble must be zeroed, not leaked from the previous pack.
	if got := packNibblesInto(dst, []uint8{0x5}); !bytes.Equal(got, []byte{0x05}) {
		t.Errorf("into([5]) on reused dst = %x, want 05 (stale high nibble leaked?)", got)
	}
	// Empty input: empty prefix, buffer untouched by the call.
	if got := packNibblesInto(dst, nil); len(got) != 0 {
		t.Errorf("into(nil) = %x, want empty", got)
	}
	// Exhaustive: for every length 0..7, into agrees with the wrapper on
	// a reused (dirty) buffer.
	dirty := make([]byte, 4)
	for i := range dirty {
		dirty[i] = 0xFF
	}
	for n := 0; n <= 7; n++ {
		vals := make([]uint8, n)
		for i := range vals {
			vals[i] = uint8(0xF - i) // descending, so stale bytes differ from packed ones
		}
		want := packNibbles(vals)
		if got := packNibblesInto(dirty, vals); !bytes.Equal(got, want) {
			t.Errorf("n=%d: into on dirty dst = %x, wrapper = %x", n, got, want)
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
