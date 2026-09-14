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

func TestLoadConfigRejectsUnknownDtype(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(`{"rules":[{"match":"w","dtype":"int5"}]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadConfig(path); err == nil {
		t.Error("LoadConfig: expected error for unknown dtype, got none")
	}
}
