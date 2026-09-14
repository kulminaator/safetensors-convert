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

func f16ToF32(bits uint16) float32 {
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
func f32ToF8E4M3(f float32) uint8 {
	sign := uint8(0)
	if math.Signbit(float64(f)) {
		sign = 0x80
	}
	af := math.Abs(float64(f))

	if math.IsNaN(float64(f)) {
		return sign | 0x7F
	}
	if math.IsInf(float64(f), 0) {
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
		val := af / math.Pow(2, 1-bias) * denom
		man := int32(math.Round(val))
		if man <= 0 {
			return sign
		}
		if man >= denom {
			return sign | (1 << manBits) // rounds up into smallest normal
		}
		return sign | uint8(man)
	}

	manF := math.Round((m - 1) * denom)
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

func f32ToF8E5M2(f float32) uint8 {
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
		man := int32(math.Round(val))
		if man <= 0 {
			return sign
		}
		if man >= denom {
			return sign | (1 << manBits)
		}
		return sign | uint8(man)
	}

	manF := math.Round((m - 1) * denom)
	if manF >= denom {
		manF = 0
		biasedExp++
	}
	if biasedExp >= 0x1F {
		return sign | 0x7C // overflow -> Inf
	}
	return sign | uint8(biasedExp)<<manBits | uint8(manF)
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
// in [-7,7], using true round-to-nearest-even (RNE). It is deliberately
// distinct from f32ToInt8's math.Round (half-away-from-zero): a quotient of
// exactly 2.5 rounds to 2 here, not 3. The quotient v = f/scale is f32-
// rounded once, exactly like the int8 path, then its magnitude is rounded:
//
//	q := int32(|v|); frac := |v| - float64(q)
//	frac > 0.5            -> q+1
//	frac == 0.5 && q odd  -> q+1
//
// Edge behavior: NaN -> 0, +Inf -> 7, -Inf -> -7, scale == 0 -> 0, and
// magnitudes >= 7.5 clamp to 7 - clamped in float space before the integer
// conversion, so out-of-int32-range quotients clamp too instead of
// overflowing.
func f32ToInt4RNE(f, scale float32) int8 {
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
