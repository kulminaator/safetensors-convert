package stconv

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// DType is a safetensors dtype string, e.g. "F32", "F16", "BF16", "I8", "F8_E4M3", "F8_E5M2".
type DType string

const (
	DTypeF64    DType = "F64"
	DTypeF32    DType = "F32"
	DTypeF16    DType = "F16"
	DTypeBF16   DType = "BF16"
	DTypeI64    DType = "I64"
	DTypeI32    DType = "I32"
	DTypeI16    DType = "I16"
	DTypeI8     DType = "I8"
	DTypeU8     DType = "U8"
	DTypeBool   DType = "BOOL"
	DTypeF8E4M3 DType = "F8_E4M3"
	DTypeF8E5M2 DType = "F8_E5M2"
)

// ByteSize returns the size in bytes of a single element of the given dtype.
func (d DType) ByteSize() (int, error) {
	switch d {
	case DTypeF64, DTypeI64:
		return 8, nil
	case DTypeF32, DTypeI32:
		return 4, nil
	case DTypeF16, DTypeBF16, DTypeI16:
		return 2, nil
	case DTypeI8, DTypeU8, DTypeBool, DTypeF8E4M3, DTypeF8E5M2:
		return 1, nil
	default:
		return 0, fmt.Errorf("unknown dtype %q", d)
	}
}

// TensorInfo mirrors one entry in the safetensors JSON header.
type TensorInfo struct {
	DType       DType    `json:"dtype"`
	Shape       []int64  `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"` // relative to start of the data block, [begin, end)
}

// TensorEntry keeps a tensor's name alongside its info, so we can preserve
// header order instead of relying on Go's unordered maps.
type TensorEntry struct {
	Name string
	Info TensorInfo
}

// Header represents a fully parsed safetensors file header.
type Header struct {
	Tensors  []TensorEntry     // in original file order
	Metadata map[string]string // from the "__metadata__" key, may be nil
}

// ReadHeader reads and parses the safetensors header from r.
// It returns the parsed header plus the absolute byte offset in the file
// where the raw tensor data block begins (dataStart), which every
// TensorInfo.DataOffsets value is relative to.
func ReadHeader(r io.ReadSeeker) (*Header, int64, error) {
	var headerLen uint64
	if err := binary.Read(r, binary.LittleEndian, &headerLen); err != nil {
		return nil, 0, fmt.Errorf("reading header length: %w", err)
	}
	if headerLen == 0 || headerLen > 1<<32 {
		return nil, 0, fmt.Errorf("implausible header length %d", headerLen)
	}

	raw := make([]byte, headerLen)
	if _, err := io.ReadFull(r, raw); err != nil {
		return nil, 0, fmt.Errorf("reading header bytes: %w", err)
	}

	h, err := parseHeaderJSON(raw)
	if err != nil {
		return nil, 0, err
	}

	dataStart := int64(8) + int64(headerLen)
	return h, dataStart, nil
}

// parseHeaderJSON parses the header JSON object while preserving key order,
// which encoding/json's map decoding does not do on its own.
func parseHeaderJSON(raw []byte) (*Header, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()

	tok, err := dec.Token()
	if err != nil {
		return nil, fmt.Errorf("parsing header JSON: %w", err)
	}
	if delim, ok := tok.(json.Delim); !ok || delim != '{' {
		return nil, fmt.Errorf("header JSON does not start with an object")
	}

	h := &Header{}

	for dec.More() {
		keyTok, err := dec.Token()
		if err != nil {
			return nil, fmt.Errorf("parsing header key: %w", err)
		}
		key, ok := keyTok.(string)
		if !ok {
			return nil, fmt.Errorf("expected string key in header, got %v", keyTok)
		}

		if key == "__metadata__" {
			var md map[string]string
			if err := dec.Decode(&md); err != nil {
				return nil, fmt.Errorf("parsing __metadata__: %w", err)
			}
			h.Metadata = md
			continue
		}

		var raw json.RawMessage
		if err := dec.Decode(&raw); err != nil {
			return nil, fmt.Errorf("parsing entry for %q: %w", key, err)
		}
		var info TensorInfo
		if err := json.Unmarshal(raw, &info); err != nil {
			return nil, fmt.Errorf("parsing tensor info for %q: %w", key, err)
		}
		h.Tensors = append(h.Tensors, TensorEntry{Name: key, Info: info})
	}

	return h, nil
}

// MarshalJSON renders the header back into safetensors' expected JSON form,
// preserving the original tensor order and re-emitting metadata if present.
func (h *Header) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer
	buf.WriteByte('{')

	first := true
	writeComma := func() {
		if !first {
			buf.WriteByte(',')
		}
		first = false
	}

	if h.Metadata != nil {
		writeComma()
		buf.WriteString(`"__metadata__":`)
		b, err := json.Marshal(h.Metadata)
		if err != nil {
			return nil, err
		}
		buf.Write(b)
	}

	for _, e := range h.Tensors {
		writeComma()
		keyB, err := json.Marshal(e.Name)
		if err != nil {
			return nil, err
		}
		buf.Write(keyB)
		buf.WriteByte(':')
		infoB, err := json.Marshal(e.Info)
		if err != nil {
			return nil, err
		}
		buf.Write(infoB)
	}

	buf.WriteByte('}')
	return buf.Bytes(), nil
}

// CreateOutputFile creates path, writes the 8-byte header length plus the
// (padded) JSON header for h, and returns the still-open file positioned
// exactly at the start of the data section.
//
// Deliberately does NOT take the tensor data as a parameter: callers are
// expected to stream tensor bytes into the returned file afterward, in the
// same order used to compute h's DataOffsets, so that a multi-gigabyte
// model never has to be materialized in memory as a single byte slice.
func CreateOutputFile(path string, h *Header) (*os.File, error) {
	headerJSON, err := h.MarshalJSON()
	if err != nil {
		return nil, fmt.Errorf("marshaling header: %w", err)
	}

	// safetensors pads the header with spaces so the data section starts
	// on an 8-byte boundary. Not strictly required by the spec, but it's
	// what the reference implementation does, and it's harmless.
	pad := (8 - (len(headerJSON)+8)%8) % 8
	if pad > 0 {
		padded := make([]byte, len(headerJSON)+pad)
		copy(padded, headerJSON)
		for i := len(headerJSON); i < len(padded); i++ {
			padded[i] = ' '
		}
		headerJSON = padded
	}

	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}

	if err := binary.Write(f, binary.LittleEndian, uint64(len(headerJSON))); err != nil {
		f.Close()
		return nil, err
	}
	if _, err := f.Write(headerJSON); err != nil {
		f.Close()
		return nil, err
	}
	return f, nil
}
