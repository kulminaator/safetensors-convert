package stconv

import (
	"math"
	"math/bits"
)

// This file implements the production dtype conversions: decoders from
// {float16, bfloat16, float8_e4m3fn, float8_e5m2} to float32, encoders from
// float32 to {float8_e4m3fn, float8_e5m2, int8, int4, e2m1}, the E8M0
// block-scale codec, a round-to-nearest-even e4m3 encoder for the NVFP4
// block-scale path, and the shared 4-bit nibble packing, using only the
// standard library. The test-only round-trip/encoder counterparts (f32ToF16,
// f32ToBF16, int8ToF32, int4ToF32, e2m1ToF32, unpackNibbles) live in
// dtype_helpers_test.go.
//
// float8_e4m3fn and float8_e5m2 follow the OCP 8-bit floating point spec
// (the same layouts used by PyTorch's torch.float8_e4m3fn / torch.float8_e5m2
// and ONNX's Float8E4M3FN / Float8E5M2):
//
//   e4m3fn: 1 sign | 4 exponent (bias 7)  | 3 mantissa. No infinities.
//           Exponent==0xF, mantissa==0x7 is the only NaN pattern; all other
//           exponent==0xF values are ordinary (large) normals. Max finite
//           magnitude is 448.
//   e5m2:   1 sign | 5 exponent (bias 15) | 2 mantissa. IEEE-754-like:
//           exponent==0x1F, mantissa==0 is +/-Inf, mantissa!=0 is NaN.
//           Max finite magnitude is 57344.
//
// NaN/Inf convention for the quantizing encoders: NaN quantizes to 0; +/-Inf
// clamps to the format's finite max; every quantizing encoder does this
// EXPLICITLY - none of them rely on the implementation-defined float->int
// conversions of the Go spec (e.g. int8(NaN) is 0 on amd64 today, but the
// spec does not guarantee it). Concretely:
//
//   - int8 (f32ToInt8): NaN -> 0; +/-Inf -> +/-127.
//   - int4 (f32ToInt4RNE): NaN -> 0; +/-Inf -> +/-7.
//   - e2m1 (f32ToE2M1): NaN -> 0x00 (no NaN code exists); +/-Inf -> +/-6.
//   - e4m3fn (f32ToF8E4M3, f32ToE4M3RNE): NaN and +/-Inf -> the 0x7F
//     pattern - the same sentinel finite overflow already saturates to,
//     since e4m3fn has no infinity.
//   - e5m2 (f32ToF8E5M2): NaN -> the NaN pattern; +/-Inf -> the Inf
//     pattern (e5m2 represents infinity, so it is kept, not clamped).
//   - e8m0 (e8m0Encode, a scale codec): NaN and m <= 0 -> code 0 (scale
//     0); +Inf -> code 254 (saturates; callers pass finite block maxima).
//
// The max-abs scale scans in convert.go/quant.go skip NaN/Inf on the
// strength of this promise: such values can never poison a scale and are
// pinned to defined output codes by the encoder they are passed to.

// ---------- float16 (IEEE-754 binary16) ----------

// f16ToF32 decodes an f16 bit pattern to float32 by pure bit manipulation.
// Every f16 value is exactly representable in f32, so the decode is a
// field remap, not a value conversion:
//
//   - sign: bit 15 -> bit 31.
//   - normal (exp 1..0x1E): unbias the 5-bit exponent (bias 15 -> 127) and
//     left-shift the 10-bit mantissa into f32's 23-bit fraction field.
//   - subnormal (exp 0, man != 0): value = man * 2^-24. Left-shift man until
//     its leading 1 lands at bit 9 (shift = 9 - leadingOnePos =
//     LeadingZeros16(man) - 6, in [0,9]); the value then reads as
//     (1 + (m-512)/512) * 2^(-15-shift), i.e. f32 exponent 112-shift and
//     fraction (m-512) << 14 (the leading 1 becomes f32's implicit bit-22 1).
//   - +/-Inf, NaN (exp == 0x1F): the all-ones f32 exponent
//     (0x7F800000) plus the f16 mantissa carried into the top ten bits of
//     f32's fraction field (man << 13) - nonzero for NaN, which is what
//     distinguishes it from Inf.
//   - +/-0 (exp == 0, man == 0): the sign bit alone.
//
// No float math anywhere: f16->f32 is exact, so the decode is a remap.
func f16ToF32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1F
	man := h & 0x3FF

	var u uint32
	switch {
	case exp == 0x1F:
		// +/-Inf (man == 0) or NaN (man != 0); see the doc comment.
		u = sign | 0x7F800000 | uint32(man)<<13
	case exp == 0:
		if man == 0 {
			u = sign // +/-0
		} else {
			shift := bits.LeadingZeros16(man) - 6 // 0..9
			m := uint32(man) << uint(shift)
			u = sign | uint32(112-shift)<<23 | (m-512)<<14
		}
	default:
		u = sign | (exp+112)<<23 | uint32(man)<<13
	}
	return math.Float32frombits(u)
}

// ---------- bfloat16 ----------

func bf16ToF32(bits uint16) float32 {
	return math.Float32frombits(uint32(bits) << 16)
}

// ---------- float8_e4m3fn ----------

// f32ToF8E4M3 encodes f to e4m3fn with round-half-away-from-zero mantissa
// rounding (deliberate; pinned by golden tests). It produces the same
// layout as the PyTorch/ONNX e4m3fn, but at exact mantissa midpoints its
// bytes differ from torch's RNE cast - f32ToE4M3RNE is the RNE variant,
// used only for NVFP4 block scales. NaN and +/-Inf encode to the 0x7F
// NaN pattern; magnitudes beyond 448 encode to 0x7F as well (e4m3fn has
// no Inf), and tiny magnitudes flush to zero.
//
// Pure bit manipulation, no Frexp/Round/Pow: the magnitude is read as the
// 24-bit significand M (leading 1 explicit) and unbiased exponent e,
// af = M * 2^(e-23), and the encoding is a fixed-point shift-and-round of
// M's bits:
//
//   - normal (biasedExp = e+7 >= 1): the 3 mantissa bits are the top
//     fraction bits, man>>20; the dropped low 20 bits round
//     half-away-from-zero - up if dropped*2 >= 2^20 - and a carry (mantissa
//     8) propagates into the exponent. biasedExp > 15, or biasedExp == 15
//     with mantissa 7 (the reserved NaN pattern), saturates to 0x7F.
//   - subnormal (biasedExp < 1): the value in units of the smallest
//     subnormal step 2^-9 is af/2^-9 = M/2^(14-e) - a right shift of M by
//     14-e with the same half-away round on the dropped bits. e <= -11
//     leaves it below half a step, which flushes to zero.
//
// f32 subnormals (max magnitude (2^23-1)*2^-149) are far below half an
// e4m3 subnormal step (2^-10), so they flush to zero as well.
func f32ToF8E4M3(f float32) uint8 {
	u := math.Float32bits(f)
	sign := uint8(u>>31) << 7
	ef := (u >> 23) & 0xFF
	man := u & 0x7FFFFF

	switch {
	case ef == 0xFF:
		// NaN (man != 0) and +/-Inf (man == 0) both encode to the 0x7F
		// pattern: e4m3fn has no Inf, and that pattern is its NaN.
		return sign | 0x7F
	case ef == 0:
		// +/-0 (man == 0) and f32 subnormals (man != 0) - the latter are
		// below half the smallest e4m3 subnormal step - both encode to
		// the sign bit alone (see doc comment).
		return sign
	}

	e := int(ef) - 127
	biasedExp := e + 7

	if biasedExp < 1 {
		// Subnormal range; smallest normal is 2^(1-7) = 2^-6, step 2^-9.
		// d = 14-e is the shift taking M to units of 2^-9; e <= -7, so
		// d >= 21.
		d := 14 - e
		if d >= 25 {
			return sign // value < half a subnormal step: flush to zero
		}
		M := man | 0x800000
		q := M >> uint(d)
		if M-(q<<uint(d)) >= 1<<uint(d-1) {
			q++ // half-away-from-zero on the dropped bits
		}
		if q >= 8 {
			return sign | 8 // rounds up into the smallest normal
		}
		return sign | uint8(q)
	}

	manF := man >> 20 // top 3 bits of the 23-bit fraction
	if (man&0xFFFFF)*2 >= 1<<20 {
		manF++ // half-away-from-zero on the dropped low 20 bits
	}
	if manF >= 8 {
		manF = 0
		biasedExp++ // carry into the exponent
	}
	const maxExp = 0xF // 4 exponent bits, all-ones is usable except mantissa==7
	if biasedExp > maxExp {
		// True overflow (beyond even the NaN-adjacent max): saturate into
		// the NaN pattern, since e4m3fn has no infinity to represent this.
		return sign | 0x7F
	}
	if biasedExp == maxExp && manF >= 7 {
		// This exact bit pattern is reserved for NaN.
		return sign | 0x7F
	}
	return sign | uint8(biasedExp)<<3 | uint8(manF)
}

// rneRound rounds a non-negative float64 to the nearest integer using
// round-to-nearest-even (RNE): a fraction > 0.5 rounds up, and a fraction
// exactly 0.5 rounds up only to an even integer (ties to even). The input
// must be exact enough that the fractional part is computed without rounding
// error (see f32ToE4M3RNE for why float64 suffices).
func rneRound(t float64) int32 {
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

// f32ToE4M3RNE encodes f to e4m3fn using round-to-nearest-even (RNE), the
// same layout and edge behavior as f32ToF8E4M3 (no Inf; the 0x7F pattern is
// the saturation/NaN marker; NaN input -> 0x7F; magnitudes beyond 448
// saturate to 0x7F) but with true RNE mantissa rounding.
//
// Why a second e4m3 encoder exists: f32ToF8E4M3 rounds the mantissa with
// math.Round, which is round-half-AWAY (e.g. mantissa units 2.5 -> 3). That
// encoder's exact output bytes are pinned by the fp8_e4m3 golden tests, so it
// must not change. The NVIDIA NVFP4 recipe, however, rounds per-block scales
// to e4m3 with RNE, so the scale path needs this distinct encoder. The two
// agree on every non-tie input and differ only at exact midpoints.
//
// RNE is applied in both the normal and subnormal paths by rounding
// t = mantissa value in integer units (computed in float64, exact enough for
// tie detection) with rneRound. In the normal path t = (m-1)*8 with m in
// [1,2) in float64; in the subnormal path t is the value in units of the
// smallest subnormal step (2^-6/8).
func f32ToE4M3RNE(f float32) uint8 {
	sign := uint8(0)
	if math.Signbit(float64(f)) {
		sign = 0x80
	}
	af := math.Abs(float64(f))

	if math.IsNaN(float64(f)) {
		return sign | 0x7F
	}
	if math.IsInf(af, 1) {
		return sign | 0x7F // no Inf; saturate into the NaN pattern
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
		man := rneRound(t)
		if man <= 0 {
			return sign
		}
		if man >= denom {
			return sign | (1 << manBits) // rounds up into smallest normal
		}
		return sign | uint8(man)
	}

	t := (m - 1) * denom
	manF := rneRound(t)
	if manF >= denom {
		manF = 0
		biasedExp++
	}
	const maxExp = 0xF // 4 exponent bits, all-ones is usable except mantissa==7
	if biasedExp > maxExp {
		// True overflow (beyond even the NaN-adjacent max): saturate into
		// the NaN pattern, since e4m3fn has no infinity to represent this.
		return sign | 0x7F
	}
	if biasedExp == maxExp && manF >= 7 {
		// This exact bit pattern is reserved for NaN.
		return sign | 0x7F
	}
	return sign | uint8(biasedExp)<<manBits | uint8(manF)
}

func f8E4M3ToF32(bits uint8) float32 {
	sign := bits >> 7
	exp := (bits >> 3) & 0xF
	man := bits & 0x7

	var val float64
	switch {
	case exp == 0xF && man == 0x7:
		val = math.NaN()
	case exp == 0:
		val = float64(man) / 8 * math.Pow(2, -6)
	default:
		val = (1 + float64(man)/8) * math.Pow(2, float64(exp)-7)
	}
	if sign == 1 {
		val = -val
	}
	return float32(val)
}

// ---------- float8_e5m2 ----------

// f32ToF8E5M2 encodes f to e5m2 with round-half-away-from-zero mantissa
// rounding (deliberate; pinned by golden tests). Same layout as PyTorch's
// torch.float8_e5m2 and ONNX's Float8E5M2. NaN encodes to the NaN pattern
// (0x7F); +/-Inf encode to the Inf pattern (0x7C) - e5m2 represents
// infinity, so it is kept, not clamped; magnitudes beyond 57344 encode to
// +/-Inf as well, and tiny magnitudes flush to zero.
//
// Pure bit manipulation, no Frexp/Round/Pow: the magnitude is read as the
// 24-bit significand M (leading 1 explicit) and unbiased exponent e,
// af = M * 2^(e-23), and the encoding is a fixed-point shift-and-round of
// M's bits:
//
//   - normal (biasedExp = e+15 >= 1): the 2 mantissa bits are the top
//     fraction bits, man>>21; the dropped low 21 bits round
//     half-away-from-zero - up if dropped*2 >= 2^21 - and a carry
//     (mantissa 4) propagates into the exponent. biasedExp >= 31
//     saturates to +/-Inf.
//   - subnormal (biasedExp < 1): the value in units of the smallest
//     subnormal step 2^-16 is af/2^-16 = M/2^(7-e) - a right shift of M by
//     7-e with the same half-away round on the dropped bits. e <= -18
//     leaves it below half a step, which flushes to zero.
//
// f32 subnormals (max magnitude (2^23-1)*2^-149) are far below half an
// e5m2 subnormal step (2^-17), so they flush to zero as well.
func f32ToF8E5M2(f float32) uint8 {
	u := math.Float32bits(f)
	sign := uint8(u>>31) << 7
	ef := (u >> 23) & 0xFF
	man := u & 0x7FFFFF

	switch {
	case ef == 0xFF:
		if man == 0 {
			return sign | 0x7C // +/-Inf: e5m2 represents infinity
		}
		return sign | 0x7F // NaN
	case ef == 0:
		// +/-0 (man == 0) and f32 subnormals (man != 0) - the latter are
		// below half the smallest e5m2 subnormal step - both encode to
		// the sign bit alone (see doc comment).
		return sign
	}

	e := int(ef) - 127
	biasedExp := e + 15

	if biasedExp < 1 {
		// Subnormal range; smallest normal is 2^(1-15) = 2^-14, step 2^-16.
		// d = 7-e is the shift taking M to units of 2^-16; e <= -15, so
		// d >= 22.
		d := 7 - e
		if d >= 25 {
			return sign // value < half a subnormal step: flush to zero
		}
		M := man | 0x800000
		q := M >> uint(d)
		if M-(q<<uint(d)) >= 1<<uint(d-1) {
			q++ // half-away-from-zero on the dropped bits
		}
		if q >= 4 {
			return sign | 4 // rounds up into the smallest normal
		}
		return sign | uint8(q)
	}

	manF := man >> 21 // top 2 bits of the 23-bit fraction
	if (man&0x1FFFFF)*2 >= 1<<21 {
		manF++ // half-away-from-zero on the dropped low 21 bits
	}
	if manF >= 4 {
		manF = 0
		biasedExp++ // carry into the exponent
	}
	if biasedExp >= 0x1F {
		return sign | 0x7C // overflow -> Inf
	}
	return sign | uint8(biasedExp)<<2 | uint8(manF)
}

func f8E5M2ToF32(bits uint8) float32 {
	sign := bits >> 7
	exp := (bits >> 2) & 0x1F
	man := bits & 0x3

	var val float64
	switch {
	case exp == 0x1F:
		if man == 0 {
			val = math.Inf(1)
		} else {
			val = math.NaN()
		}
	case exp == 0:
		val = float64(man) / 4 * math.Pow(2, -14)
	default:
		val = (1 + float64(man)/4) * math.Pow(2, float64(exp)-15)
	}
	if sign == 1 {
		val = -val
	}
	return float32(val)
}

// ---------- int8 (symmetric, per-tensor scale) ----------

// int8Scale computes a symmetric per-tensor quantization scale such that
// round(maxAbs/scale) == 127.
func int8Scale(maxAbs float32) float32 {
	if maxAbs == 0 {
		return 1 // arbitrary; every value is zero anyway
	}
	return maxAbs / 127.0
}

// f32ToInt8 quantizes the f32 quotient f/scale to the nearest int8 in
// [-127,127] using math.Round (half-away-from-zero). Edge behavior per the
// file-header convention: NaN -> 0 (handled explicitly - the clamps below
// are float comparisons that are false for NaN, and int8(NaN) is
// implementation-defined per the Go spec), +Inf -> 127, -Inf -> -127,
// scale == 0 -> 0.
//
// S4 note: an f32-only rewrite (one f32 divide, magnitude clamps and the
// sign read from the f32 bits, no float64, no math.Round) was prototyped
// and is byte-identical to this version (see f32ToInt8Ref and
// TestInt8Int4QuantizeOracle), but on this machine it measured ~2.1x
// slower than the float64 path in the S1 microbenchmark, so the float64
// implementation is kept.
func f32ToInt8(f, scale float32) int8 {
	if math.IsNaN(float64(f)) {
		return 0
	}
	if scale == 0 {
		return 0
	}
	q := math.Round(float64(f / scale))
	if q > 127 {
		q = 127
	}
	if q < -127 {
		q = -127
	}
	return int8(q)
}

// ---------- int4 (symmetric, per-tensor scale, round-to-nearest-even) ----------
//
// "Naive 4-bit": plain integer 4-bit, the direct analogue of the per-tensor
// int8 above. Values live in [-7,7], packed two's-complement, two per byte
// via packNibbles (see the shared nibble section below).

// int4Scale computes a symmetric per-tensor quantization scale such that
// round(maxAbs/scale) == 7. Mirrors int8Scale.
func int4Scale(maxAbs float32) float32 {
	if maxAbs == 0 {
		return 1 // arbitrary; every value is zero anyway
	}
	return maxAbs / 7.0
}

// f32ToInt4RNE quantizes the f32 quotient f/scale to the nearest int4 value
// in [-7,7], using true round-to-nearest-even (RNE), as exact bit
// manipulation - one f32 divide, the magnitude clamp and the RNE rounding
// on v's f32 bits, no float64 promotion. It is deliberately distinct from
// f32ToInt8's half-away-from-zero: a quotient of exactly 2.5 rounds to 2
// here, not 3. For |v| < 8 the RNE tests reduce to exact integer compares
// on v's 24-bit significand - the same rounding the pre-S4 float64 version
// performed, bit for bit.
//
// Edge behavior: NaN -> 0, +Inf -> 7, -Inf -> -7, scale == 0 -> 0, and
// magnitudes >= 7.5 clamp to 7 - clamped in float space before the integer
// conversion, so out-of-int32-range quotients clamp too instead of
// overflowing.
func f32ToInt4RNE(f, scale float32) int8 {
	if math.Float32bits(scale)&0x7FFFFFFF == 0 { // scale == 0 (incl. -0)
		return 0
	}
	u := math.Float32bits(f)
	if u&0x7FFFFFFF > 0x7F800000 { // NaN
		return 0
	}
	if u&0x7FFFFFFF == 0x7F800000 { // +/-Inf
		if u&0x80000000 != 0 {
			return -7
		}
		return 7
	}
	v := f / scale
	a := math.Float32bits(v) & 0x7FFFFFFF
	// Clamp in float space before any integer conversion: |v| >= 8 (or a
	// NaN v from a NaN scale, or +/-Inf from a division overflow) rounds
	// to 8 or more, so it saturates to the int4 bound either way - the
	// same values the pre-S4 float-space clamp produced.
	if a >= 0x41000000 || a > 0x7F800000 {
		if u&0x80000000 != 0 {
			return -7
		}
		return 7
	}
	// |v| < 8: RNE on v's bits in integer space. The magnitude's exponent
	// is e in {0, 126, 127, 128, 129}; v = M' * 2^(e-150) with M' the
	// 24-bit significand. For e >= 127 the integer part is the top
	// 150-e bits of M' and the fraction the rest, so the RNE tests are
	// exact integer compares - the same rounding the pre-S4 float64
	// version performed, bit for bit.
	var q int32
	if a >= 0x3F800000 { // |v| >= 1 (e in {127, 128, 129})
		M := a&0x7FFFFF | 0x800000 // 24-bit significand
		s := 150 - int(a>>23)      // in {23, 22, 21}
		q = int32(M >> uint(s))
		frac := M & (1<<s - 1)
		if frac*2 > 1<<s || (frac*2 == 1<<s && q%2 == 1) {
			q++ // fraction > 0.5, or exactly 0.5 with odd integer part
		}
		if q > 7 {
			q = 7 // the 7.5 -> 8 RNE tie
		}
	} else if a > 0x3F000000 { // 0.5 < |v| < 1 -> 1; |v| <= 0.5 -> 0 (the
		q = 1 // 0.5 tie goes to the even integer 0)
	}
	if u&0x80000000 != 0 {
		q = -q
	}
	return int8(q)
}

// ---------- e2m1 (4-bit float, MXFP4/NVFP4 element format) ----------
//
// OCP 4-bit float: 1 sign | 2 exponent (bias 1) | 1 mantissa.
// Subnormals (e==0): value = 0.5*m. Normals: value = (1 + 0.5*m) * 2^(e-1).
// Positive value grid: {0, 0.5, 1, 1.5, 2, 3, 4, 6} (code = e<<1 | m).
// Max magnitude is 6; all 16 patterns are finite (no Inf/NaN codes).

var e2m1Grid = [8]float32{0, 0.5, 1, 1.5, 2, 3, 4, 6}

// f32ToE2M1 encodes f to the nearest e2m1 code using round-to-nearest-even.
// The magnitude is compared against the 8 positive grid values; an exact
// midpoint between two grid values ties to the one with the even mantissa
// bit (m==0, i.e. the grid values 0, 1, 2, 4). Edge behavior: NaN -> 0x00,
// +Inf -> 0x07, -Inf -> 0x0F, |f| > 6 saturates to 6 (code 0x07), and a
// magnitude that rounds to 0 emits 0x00 (no negative zero is ever emitted).
func f32ToE2M1(f float32) uint8 {
	var sign uint8
	if math.Signbit(float64(f)) {
		sign = 0x08
	}
	af := math.Abs(float64(f))
	if math.IsNaN(float64(f)) {
		return 0 // no NaN pattern exists
	}
	if af == math.Inf(1) || af > 6 {
		return sign | 0x07 // saturate to max magnitude 6
	}
	// Midpoint between grid[i-1] and grid[i]: below it, grid[i-1] is
	// nearest; above it, grid[i] is nearest. At the exact midpoint, RNE
	// picks the even mantissa bit: for odd i the lower value has m==0 and
	// wins the tie (af <= mid); for even i the upper value has m==0 and
	// wins (af < mid, so the tie falls through to grid[i]).
	for i := 1; i < 8; i++ {
		mid := (float64(e2m1Grid[i-1]) + float64(e2m1Grid[i])) / 2
		if (i%2 == 1 && af <= mid) || (i%2 == 0 && af < mid) {
			code := sign | uint8(i-1)
			// A magnitude that rounds to 0 emits 0x00 (no negative zero).
			if code&0x07 == 0 {
				return 0
			}
			return code
		}
	}
	// Above the last midpoint (5): nearest is 6 (code 0x07).
	return sign | 0x07
}

// ---------- e8m0 (8-bit exponent-only scale, MXFP4 block scale) ----------
//
// OCP MXFP4 block scale: 8-bit exponent-only, bias 127. Code 0 -> scale 0;
// code c in [1,254] -> scale 2^(c-127); code 255 -> Inf (reserved; the
// encoder never emits it).
//
// Scale selection per the OCP MX reference: for a block max m >= 0, pick
// the smallest power of two s >= m/6, so the scaled block max m/s lands in
// (3, 6] - the upper half of the E2M1 dynamic range. Equivalently
// e = ceil(log2(m/6)).
//
// Exact computation (no float log/rounding):
//
//	m == 0 -> code 0.
//	L = floor(log2(m)) read straight off the f32 bits: for a normal,
//	(bits>>23)-127; for a subnormal, highestSetBit(mantissa)-149, since
//	m = mantissa·2^-149 and log2(mantissa) is the highest set bit.
//	Since 2^L <= m < 2^(L+1), m/6 < 2^(L+1)/6 < 2^(L-1), so
//	ceil(log2(m/6)) is either L-2 or L-1; it crosses 2^(L-2) exactly when
//	m/6 crosses it, i.e. at m = 6·2^(L-2). With
//	B := 6·2^(L-2)  (computed exactly with math.Ldexp, exact for the
//	L >= -148 range that matters - at L == -149 the rounding is harmless
//	because the code clamps to 1 either way),
//	e = ceil(log2(m/6)) = L-1 if m > B, else L-2.
//	code = clamp(e+127, 1, 254).
//
// The lower clamp covers tiny m: the smallest positive scale 2^-126 still
// satisfies m/s <= 6 for any m <= 6·2^-126, and below that the elements
// just quantize toward 0 (no special flush needed). The upper clamp never
// fires for finite f32 (m/6 < 2^127 always); it is kept so the 255/Inf
// code is structurally unreachable.

func e8m0Encode(m float32) uint8 {
	if m <= 0 || math.IsNaN(float64(m)) {
		return 0
	}
	if math.IsInf(float64(m), 1) {
		return 254 // saturate; callers pass finite block maxima
	}
	fbits := math.Float32bits(m)
	var l int
	if fbits>>23&0xFF != 0 {
		l = int(fbits>>23) - 127
	} else {
		mant := fbits & 0x7FFFFF
		l = 31 - bits.LeadingZeros32(mant) - 149
	}
	e := l - 2
	if m > float32(math.Ldexp(6, l-2)) {
		e = l - 1
	}
	code := e + 127
	if code < 1 {
		code = 1
	}
	if code > 254 {
		code = 254
	}
	return uint8(code)
}

// e8m0Scale decodes an E8M0 code to its f32 scale (0 -> 0; 255 -> +Inf).
// All other codes are exact powers of two.
func e8m0Scale(code uint8) float32 {
	switch code {
	case 0:
		return 0
	case 255:
		return float32(math.Inf(1))
	}
	return float32(math.Ldexp(1, int(code)-127))
}

// ---------- shared 4-bit nibble packing (e2m1 and int4) ----------

// packNibblesInto packs 4-bit values into the caller-provided dst and
// returns the filled prefix, with the same layout packNibbles produces:
// two values per byte, element 2i in the low nibble (bits 0-3) and
// element 2i+1 in the high nibble (bits 4-7), an odd count padded with a
// trailing zero nibble. dst must hold (len(vals)+1)/2 bytes; the streaming
// passes reuse one such buffer per pass instead of allocating per block.
// The used prefix is zeroed first so a reused dst cannot leak stale
// nibbles from a previous pack (matters for odd-length packs, whose last
// byte's high nibble is only the zero pad).
func packNibblesInto(dst []byte, vals []uint8) []byte {
	n := (len(vals) + 1) / 2
	for i := 0; i < n; i++ {
		dst[i] = 0
	}
	for i, v := range vals {
		b := i / 2
		if i%2 == 0 {
			dst[b] = v & 0x0F
		} else {
			dst[b] |= (v & 0x0F) << 4
		}
	}
	return dst[:n]
}

// packNibbles packs 4-bit values into bytes, two per byte, little-endian
// element order: element 2i goes to the low nibble (bits 0-3) and element
// 2i+1 to the high nibble (bits 4-7). An odd count pads a trailing zero
// nibble, so the result is always (len(vals)+1)/2 bytes. The streaming
// passes use packNibblesInto with a reused buffer; this wrapper remains
// for one-shot callers (and the tests).
func packNibbles(vals []uint8) []byte {
	return packNibblesInto(make([]byte, (len(vals)+1)/2), vals)
}
