package stconv

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
)

// Config lets a user say "keep this layer as-is" or "quantize that layer
// to int8" without editing code. It's intentionally simple today (exact
// name match or regex pattern match, first match wins, in list order),
// so it's easy to extend later (e.g. per-layer scale strategy, group-wise
// quantization, skip lists by substring, etc.) without a breaking format
// change to the JSON below.
//
// Example config.json:
//
//	{
//	  "default": "fp8_e4m3",
//	  "rules": [
//	    { "pattern": "^lm_head\\.", "dtype": "none" },
//	    { "pattern": "^model\\.embed_tokens\\.", "dtype": "none" },
//	    { "pattern": "\\.norm\\.weight$", "dtype": "none" },
//	    { "match": "model.layers.0.mlp.down_proj.weight", "dtype": "int8" }
//	  ]
//	}
//
// "dtype" values: "fp8_e4m3", "fp8_e5m2", "int8", "int8_convrot",
// "mxfp4", "nvfp4", "int4", or "none" (leave untouched, copied through
// at its original dtype).
type Config struct {
	Default string       `json:"default"`
	Rules   []ConfigRule `json:"rules"`
}

type ConfigRule struct {
	Match   string `json:"match,omitempty"`   // exact tensor name
	Pattern string `json:"pattern,omitempty"` // regexp, matched against tensor name
	DType   string `json:"dtype"`             // "fp8_e4m3" | "fp8_e5m2" | "int8" | "int8_convrot" | "mxfp4" | "nvfp4" | "int4" | "none"

	compiled *regexp.Regexp
}

// TargetKind is the normalized conversion target for a tensor.
type TargetKind int

const (
	TargetNone TargetKind = iota // leave the tensor exactly as it is
	TargetFP8E4M3
	TargetFP8E5M2
	TargetInt8
	TargetInt8ConvRot // Hadamard-rotated, per-row-scaled int8
	TargetMxFP4       // OCP microscaling: E2M1 elements + E8M0 32-element block scales
	TargetNVFP4       // NVIDIA FP4: E2M1 elements + E4M3 16-element block scales + F32 global scale
	TargetInt4        // naive per-tensor symmetric round-to-nearest-even int4
)

func ParseTargetKind(s string) (TargetKind, error) {
	switch s {
	case "none", "":
		return TargetNone, nil
	case "fp8_e4m3", "fp8":
		return TargetFP8E4M3, nil
	case "fp8_e5m2":
		return TargetFP8E5M2, nil
	case "int8":
		return TargetInt8, nil
	case "int8_convrot":
		return TargetInt8ConvRot, nil
	case "mxfp4":
		return TargetMxFP4, nil
	case "nvfp4":
		return TargetNVFP4, nil
	case "int4":
		return TargetInt4, nil
	default:
		return TargetNone, fmt.Errorf("unknown target dtype %q (expected fp8_e4m3, fp8_e5m2, int8, int8_convrot, mxfp4, nvfp4, int4, or none)", s)
	}
}

// LoadConfig reads and validates a JSON config file, compiling any regex
// patterns up front so errors surface immediately instead of mid-conversion.
func LoadConfig(path string) (*Config, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading config %s: %w", path, err)
	}
	var cfg Config
	if err := json.Unmarshal(b, &cfg); err != nil {
		return nil, fmt.Errorf("parsing config %s: %w", path, err)
	}
	if cfg.Default != "" {
		if _, err := ParseTargetKind(cfg.Default); err != nil {
			return nil, fmt.Errorf("config default: %w", err)
		}
	}
	for i := range cfg.Rules {
		r := &cfg.Rules[i]
		if _, err := ParseTargetKind(r.DType); err != nil {
			return nil, fmt.Errorf("config rule %d: %w", i, err)
		}
		if r.Match == "" && r.Pattern == "" {
			return nil, fmt.Errorf("config rule %d: must set \"match\" or \"pattern\"", i)
		}
		if r.Pattern != "" {
			re, err := regexp.Compile(r.Pattern)
			if err != nil {
				return nil, fmt.Errorf("config rule %d: invalid pattern %q: %w", i, r.Pattern, err)
			}
			r.compiled = re
		}
	}
	return &cfg, nil
}

// TargetFor resolves the conversion target for a given tensor name: first
// matching rule wins, in file order; falls back to the config's default,
// or to fallbackDefault if the config has none set. The second return
// value reports whether a rule actually matched: the config default and
// fallbackDefault are bulk defaults, not explicit per-tensor intent, so
// built-in policy (e.g. tensor protection) must not yield to them.
func (c *Config) TargetFor(tensorName string, fallbackDefault TargetKind) (TargetKind, bool) {
	for _, r := range c.Rules {
		if r.Match != "" && r.Match == tensorName {
			k, _ := ParseTargetKind(r.DType)
			return k, true
		}
		if r.compiled != nil && r.compiled.MatchString(tensorName) {
			k, _ := ParseTargetKind(r.DType)
			return k, true
		}
	}
	if c.Default != "" {
		k, _ := ParseTargetKind(c.Default)
		return k, false
	}
	return fallbackDefault, false
}
