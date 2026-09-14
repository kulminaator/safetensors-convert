package stconv

import "math"

// This file holds the test-only round-trip/encoder counterparts of the
// production dtype conversions in dtype.go: f32ToF16 and f32ToBF16 encode
// f32 values to float16/bfloat16 for building test inputs, int8ToF32,
// int4ToF32 and e2m1ToF32 decode quantized codes back to f32 for round-trip
// checks, and unpackNibbles inverts packNibbles. They are referenced only
// from _test.go files (testdata/gen carries its own copies), so they live
// here instead of in production code.

func f32ToF16(f float32) uint16 {
	sign := uint16(0)
	if math.Signbit(float64(f)) {
		sign = 0x8000
	}
	af := math.Abs(float64(f))

	switch {
	case math.IsNaN(float64(f)):
		return sign | 0x7E00
	case math.IsInf(float64(f), 0):
		return sign | 0x7C00
	case af == 0:
		return sign
	}

	frac, exp := math.Frexp(af) // af = frac * 2^exp, frac in [0.5,1)
	m := frac * 2               // in [1,2)
	e := int32(exp) - 1
	biasedExp := e + 15

	const manBits = 10
	const denom = 1 << manBits // 1024

	if biasedExp < 1 {
		// Subnormal or underflow. Smallest normal is 2^-14.
		val := af / math.Pow(2, -14) * denom // value in units of 2^-14/1024
		man := int32(math.Round(val))
		if man <= 0 {
			return sign
		}
		if man >= denom {
			// Rounds up into the smallest normal.
			return sign | (1 << manBits)
		}
		return sign | uint16(man)
	}

	manF := math.Round((m - 1) * denom)
	if manF >= denom {
		manF = 0
		biasedExp++
	}
	if biasedExp >= 0x1F {
		return sign | 0x7C00 // overflow -> Inf
	}
	return sign | uint16(biasedExp)<<manBits | uint16(manF)
}

func f32ToBF16(f float32) uint16 {
	bits := math.Float32bits(f)
	if math.IsNaN(float64(f)) {
		// Preserve NaN-ness; force a set mantissa bit so it can't decay to Inf.
		return uint16(bits>>16) | 0x0040
	}
	// Round to nearest, ties to even.
	roundBias := uint32(0x7FFF) + ((bits >> 16) & 1)
	bits += roundBias
	return uint16(bits >> 16)
}

func int8ToF32(q int8, scale float32) float32 {
	return float32(q) * scale
}

// int4ToF32 decodes a two's-complement int4 nibble (stored as v & 0x0F, so
// nibble >= 8 is negative: value - 16) to f32 scaled by scale.
func int4ToF32(nibble uint8, scale float32) float32 {
	v := int32(nibble & 0x0F)
	if v >= 8 {
		v -= 16
	}
	return float32(v) * scale
}

func e2m1ToF32(bits uint8) float32 {
	mag := e2m1Grid[bits&0x07]
	if bits&0x08 != 0 {
		mag = -mag
	}
	return mag
}

// unpackNibbles is the inverse of packNibbles: each byte yields its low
// nibble first, then its high nibble. (A trailing pad nibble from an odd
// element count is returned as a zero value; the caller knows the true
// element count.)
func unpackNibbles(b []byte) []uint8 {
	out := make([]uint8, len(b)*2)
	for i, x := range b {
		out[2*i] = x & 0x0F
		out[2*i+1] = x >> 4
	}
	return out
}
