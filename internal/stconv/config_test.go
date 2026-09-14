package stconv

import (
	"os"
	"path/filepath"
	"testing"
)

func TestParseTargetKind(t *testing.T) {
	cases := []struct {
		in      string
		want    TargetKind
		wantErr bool
	}{
		{"", TargetNone, false},
		{"none", TargetNone, false},
		{"fp8_e4m3", TargetFP8E4M3, false},
		{"fp8", TargetFP8E4M3, false},
		{"fp8_e5m2", TargetFP8E5M2, false},
		{"int8", TargetInt8, false},
		{"int8_convrot", TargetInt8ConvRot, false},
		{"mxfp4", TargetMxFP4, false},
		{"nvfp4", TargetNVFP4, false},
		{"int4", TargetInt4, false},
		{"INT8", TargetNone, true},
		{"int5", TargetNone, true},
	}
	for _, c := range cases {
		got, err := ParseTargetKind(c.in)
		if c.wantErr {
			if err == nil {
				t.Errorf("ParseTargetKind(%q): expected error, got none", c.in)
			}
			continue
		}
		if err != nil {
			t.Errorf("ParseTargetKind(%q): unexpected error: %v", c.in, err)
			continue
		}
		if got != c.want {
			t.Errorf("ParseTargetKind(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestLoadConfigTargetForNewDtypes(t *testing.T) {
	cfgJSON := `{
		"default": "int4",
		"rules": [
			{ "match": "w.convrot", "dtype": "int8_convrot" },
			{ "match": "w.mx", "dtype": "mxfp4" },
			{ "match": "w.nv", "dtype": "nvfp4" },
			{ "pattern": "^skip\\.", "dtype": "none" }
		]
	}`
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(cfgJSON), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	cases := []struct {
		name string
		want TargetKind
	}{
		{"w.convrot", TargetInt8ConvRot},
		{"w.mx", TargetMxFP4},
		{"w.nv", TargetNVFP4},
		{"skip.x", TargetNone},
		{"other.weight", TargetInt4}, // default
	}
	for _, c := range cases {
		if got, _ := cfg.TargetFor(c.name, TargetFP8E4M3); got != c.want {
			t.Errorf("TargetFor(%q) = %v, want %v", c.name, got, c.want)
		}
	}
}

func TestTargetForExplicitness(t *testing.T) {
	// A rule match (exact or pattern) is explicit per-tensor intent;
	// the config default and the fallback are bulk defaults and must
	// not be reported as explicit.
	cfgJSON := `{
		"default": "int4",
		"rules": [
			{ "match": "model.norm.weight", "dtype": "int8" },
			{ "pattern": "^skip\\.", "dtype": "none" }
		]
	}`
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(cfgJSON), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	cases := []struct {
		name     string
		want     TargetKind
		wantExpl bool
	}{
		{"model.norm.weight", TargetInt8, true}, // exact match rule
		{"skip.x", TargetNone, true},            // pattern rule
		{"other.weight", TargetInt4, false},     // config default
	}
	for _, c := range cases {
		got, expl := cfg.TargetFor(c.name, TargetFP8E4M3)
		if got != c.want || expl != c.wantExpl {
			t.Errorf("TargetFor(%q) = (%v, %v), want (%v, %v)", c.name, got, expl, c.want, c.wantExpl)
		}
	}
	// Rules but no default: unmatched tensors fall back, never explicit.
	noDefault := &Config{Rules: cfg.Rules}
	if got, expl := noDefault.TargetFor("other.weight", TargetFP8E4M3); got != TargetFP8E4M3 || expl {
		t.Errorf("no-default TargetFor = (%v, %v), want (TargetFP8E4M3, false)", got, expl)
	}
	// No config at all: the fallback is never explicit either.
	empty := &Config{}
	if got, expl := empty.TargetFor("anything", TargetFP8E4M3); got != TargetFP8E4M3 || expl {
		t.Errorf("empty config TargetFor = (%v, %v), want (TargetFP8E4M3, false)", got, expl)
	}
}

// TestTargetForExplicitnessVsProtection re-verifies Step 2's
// explicit/implicit semantics end to end for protected names: only
// implicit resolutions (config default, fallback) are subject to the
// default protection policy; an explicit rule match is per-tensor user
// intent and always wins. This is the contract planTensor's protection
// check (and its override warning) relies on.
func TestTargetForExplicitnessVsProtection(t *testing.T) {
	cfg := &Config{
		Default: "fp8_e4m3",
		Rules:   []ConfigRule{{Match: "model.layers.0.input_layernorm.weight", DType: "int8"}},
	}
	cases := []struct {
		name          string
		wantTarget    TargetKind
		wantExplicit  bool
		wantProtected bool // whether planTensor would keep the tensor at its original dtype
	}{
		{"model.layers.0.input_layernorm.weight", TargetInt8, true, false},    // explicit rule wins
		{"model.layers.1.input_layernorm.weight", TargetFP8E4M3, false, true}, // config default, protected name
		{"model.layers.1.mlp.gate_proj.weight", TargetFP8E4M3, false, false},  // config default, unprotected name
	}
	for _, c := range cases {
		target, explicit := cfg.TargetFor(c.name, TargetFP8E4M3)
		if target != c.wantTarget || explicit != c.wantExplicit {
			t.Errorf("TargetFor(%q) = (%v, %v), want (%v, %v)", c.name, target, explicit, c.wantTarget, c.wantExplicit)
			continue
		}
		// Mirror of planTensor's decision: protection applies only to
		// non-explicit targets.
		protected := false
		if !explicit {
			protected, _ = ProtectDefault(c.name, target)
		}
		if protected != c.wantProtected {
			t.Errorf("protection applies to %q = %v, want %v", c.name, protected, c.wantProtected)
		}
	}
}

func TestLoadConfigRejectsUnknownDtype(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(`{"rules":[{"match":"w","dtype":"int5"}]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadConfig(path); err == nil {
		t.Error("LoadConfig: expected error for unknown dtype, got none")
	}
}
