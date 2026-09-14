package stconv

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func writeIndex(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "model.safetensors.index.json")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("writing fixture index: %v", err)
	}
	return path
}

func TestLoadIndex(t *testing.T) {
	tests := []struct {
		name    string
		content string
		wantErr bool
	}{
		{
			name: "valid two-shard index with unknown extra keys",
			content: `{
			  "metadata": {"total_size": 12345},
			  "weight_map": {
			    "layers.2.mlp.weight": "model.safetensors-00002-of-00002.safetensors",
			    "layers.0.mlp.weight": "model.safetensors-00001-of-00002.safetensors",
			    "layers.1.mlp.weight": "model.safetensors-00001-of-00002.safetensors"
			  },
			  "some_unknown_key": [1, 2, 3]
			}`,
		},
		{
			name:    "missing weight_map",
			content: `{"metadata": {"total_size": 1}}`,
			wantErr: true,
		},
		{
			name:    "null weight_map",
			content: `{"weight_map": null}`,
			wantErr: true,
		},
		{
			name:    "empty weight_map",
			content: `{"weight_map": {}}`,
			wantErr: true,
		},
		{
			name:    "non-string weight_map value",
			content: `{"weight_map": {"a.weight": 5}}`,
			wantErr: true,
		},
		{
			name:    "malformed json",
			content: `{"weight_map": {`,
			wantErr: true,
		},
		{
			name:    "non-object root",
			content: `["weight_map"]`,
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := writeIndex(t, tt.content)
			idx, err := LoadIndex(path)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("expected error, got %+v", idx)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(idx.WeightMap) != 3 {
				t.Errorf("expected 3 weight_map entries, got %d", len(idx.WeightMap))
			}
			if got := idx.Metadata["total_size"]; got != float64(12345) {
				t.Errorf("metadata total_size = %v, want 12345 (kept raw)", got)
			}
		})
	}

	t.Run("nonexistent path", func(t *testing.T) {
		if _, err := LoadIndex(filepath.Join(t.TempDir(), "nope.index.json")); err == nil {
			t.Fatal("expected error for nonexistent path")
		}
	})
}

// weightMapKeyOrder returns the weight_map key order in raw index JSON,
// reading the JSON stream so the on-disk key order is preserved
// (encoding/json's map decoding loses it). Same token-reading technique as
// parseHeaderJSON.
func weightMapKeyOrder(t *testing.T, raw []byte) []string {
	t.Helper()
	dec := json.NewDecoder(bytes.NewReader(raw))
	tok, err := dec.Token()
	if err != nil {
		t.Fatalf("reading index root: %v", err)
	}
	if d, ok := tok.(json.Delim); !ok || d != '{' {
		t.Fatalf("index root is not an object: %v", tok)
	}
	for dec.More() {
		keyTok, err := dec.Token()
		if err != nil {
			t.Fatalf("reading index key: %v", err)
		}
		key, ok := keyTok.(string)
		if !ok {
			t.Fatalf("non-string index key: %v", keyTok)
		}
		if key != "weight_map" {
			var skip json.RawMessage
			if err := dec.Decode(&skip); err != nil {
				t.Fatalf("skipping %q value: %v", key, err)
			}
			continue
		}
		tok, err = dec.Token()
		if err != nil {
			t.Fatalf("reading weight_map: %v", err)
		}
		if d, ok := tok.(json.Delim); !ok || d != '{' {
			t.Fatalf("weight_map is not an object: %v", tok)
		}
		var order []string
		for dec.More() {
			k, err := dec.Token()
			if err != nil {
				t.Fatalf("reading weight_map key: %v", err)
			}
			order = append(order, k.(string))
			var v string
			if err := dec.Decode(&v); err != nil {
				t.Fatalf("reading weight_map value: %v", err)
			}
		}
		if _, err := dec.Token(); err != nil { // consume "}"
			t.Fatalf("closing weight_map: %v", err)
		}
		return order
	}
	t.Fatalf("no weight_map in index JSON")
	return nil
}

func TestIndexMarshalJSON(t *testing.T) {
	t.Run("hand-built index: deterministic, sorted fallback order", func(t *testing.T) {
		idx := &Index{
			Metadata:  map[string]any{"total_size": 10},
			WeightMap: map[string]string{"b.w": "s2", "a.w": "s1", "c.w": "s2"},
		}
		b1, err := json.Marshal(idx)
		if err != nil {
			t.Fatalf("Marshal: %v", err)
		}
		b2, err := json.Marshal(idx)
		if err != nil {
			t.Fatalf("Marshal: %v", err)
		}
		if !bytes.Equal(b1, b2) {
			t.Fatalf("MarshalJSON is not deterministic:\n%s\n%s", b1, b2)
		}
		// metadata is written first, weight_map keys in sorted fallback order.
		if !bytes.HasPrefix(b1, []byte(`{"metadata":`)) {
			t.Errorf("marshaled = %s, want metadata before weight_map", b1)
		}
		if got := weightMapKeyOrder(t, b1); !reflect.DeepEqual(got, []string{"a.w", "b.w", "c.w"}) {
			t.Errorf("key order = %v, want sorted [a.w b.w c.w]", got)
		}
		// Round-trip: the bytes parse back to the same index.
		var back Index
		if err := json.Unmarshal(b1, &back); err != nil {
			t.Fatalf("Unmarshal: %v", err)
		}
		if want := map[string]string{"b.w": "s2", "a.w": "s1", "c.w": "s2"}; !reflect.DeepEqual(back.WeightMap, want) {
			t.Errorf("round-trip weight_map = %v, want %v", back.WeightMap, want)
		}
		if got := back.Metadata["total_size"]; got != float64(10) {
			t.Errorf("round-trip total_size = %v, want 10", got)
		}
	})

	t.Run("carried order is preserved", func(t *testing.T) {
		// Deliberately not sorted: if MarshalJSON fell back to sorted keys,
		// the on-disk order would differ from the carried one.
		idx := &Index{
			WeightMap: map[string]string{
				"a.w": "s1", "a.w.scale": "s1", "b.w": "s2", "b.w.scale": "s2",
			},
			weightMapOrder: []string{"b.w", "a.w", "b.w.scale", "a.w.scale"},
		}
		b, err := json.Marshal(idx)
		if err != nil {
			t.Fatalf("Marshal: %v", err)
		}
		if got := weightMapKeyOrder(t, b); !reflect.DeepEqual(got, idx.weightMapOrder) {
			t.Errorf("key order = %v, want %v", got, idx.weightMapOrder)
		}
	})

	t.Run("no metadata omits the key", func(t *testing.T) {
		idx := &Index{WeightMap: map[string]string{"a.w": "s1"}, weightMapOrder: []string{"a.w"}}
		b, err := json.Marshal(idx)
		if err != nil {
			t.Fatalf("Marshal: %v", err)
		}
		if !bytes.HasPrefix(b, []byte(`{"weight_map":`)) {
			t.Errorf("marshaled = %s, want weight_map first when metadata is nil", b)
		}
	})
}

func TestShardFiles(t *testing.T) {
	idx, err := LoadIndex(writeIndex(t, `{
	  "weight_map": {
	    "t2": "model.safetensors-00003-of-00003.safetensors",
	    "t3": "model.safetensors-00001-of-00003.safetensors",
	    "t4": "model.safetensors-00003-of-00003.safetensors",
	    "t5": "model.safetensors-00002-of-00003.safetensors",
	    "t6": "model.safetensors-00001-of-00003.safetensors"
	  }
	}`))
	if err != nil {
		t.Fatalf("LoadIndex: %v", err)
	}
	want := []string{
		"model.safetensors-00001-of-00003.safetensors",
		"model.safetensors-00002-of-00003.safetensors",
		"model.safetensors-00003-of-00003.safetensors",
	}
	if got := idx.ShardFiles(); !reflect.DeepEqual(got, want) {
		t.Errorf("ShardFiles() = %v, want %v (deduped, sorted)", got, want)
	}
}

func TestDiscoverShards(t *testing.T) {
	const (
		shard1 = "model.safetensors-00001-of-00002.safetensors"
		shard2 = "model.safetensors-00002-of-00002.safetensors"
		shard3 = "model.safetensors-00003-of-00003.safetensors"
	)

	writeShard := func(dir, name string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(dir, name), []byte("fake shard data"), 0o644); err != nil {
			t.Fatalf("writing shard %s: %v", name, err)
		}
	}

	tests := []struct {
		name       string
		files      []string // shard files present in the dir
		indexJSON  string   // content of model.safetensors.index.json; "" = no index
		wantShards []string // expected shard basenames, in processing order
		wantErr    string   // substring expected in the error; "" = expect success
	}{
		{
			name:       "index drives shard order",
			files:      []string{shard2, shard1},
			indexJSON:  `{"weight_map":{"b.w":"` + shard2 + `","a.w":"` + shard1 + `","c.w":"` + shard1 + `"}}`,
			wantShards: []string{shard1, shard2},
		},
		{
			name:      "index references missing shard",
			files:     []string{shard1},
			indexJSON: `{"weight_map":{"a.w":"` + shard1 + `","b.w":"` + shard2 + `"}}`,
			wantErr:   shard2,
		},
		{
			name:      "stray shard not referenced by index",
			files:     []string{shard1, shard2, shard3},
			indexJSON: `{"weight_map":{"a.w":"` + shard1 + `","b.w":"` + shard2 + `"}}`,
			wantErr:   shard3,
		},
		{
			name:       "no index falls back to sorted glob",
			files:      []string{shard2, shard1},
			wantShards: []string{shard1, shard2},
		},
		{
			name:    "empty dir",
			wantErr: "no .safetensors files",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			for _, f := range tt.files {
				writeShard(dir, f)
			}
			if tt.indexJSON != "" {
				if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), []byte(tt.indexJSON), 0o644); err != nil {
					t.Fatalf("writing index: %v", err)
				}
			}

			shards, index, err := DiscoverShards(dir)
			if tt.wantErr != "" {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil (shards=%v)", tt.wantErr, shards)
				}
				if !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("error %q does not contain %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if tt.indexJSON != "" && index == nil {
				t.Fatal("expected non-nil index")
			}
			if tt.indexJSON == "" && index != nil {
				t.Fatalf("expected nil index in glob fallback, got %+v", index)
			}
			want := make([]string, len(tt.wantShards))
			for i, s := range tt.wantShards {
				want[i] = filepath.Join(dir, s)
			}
			if !reflect.DeepEqual(shards, want) {
				t.Errorf("shards = %v, want %v", shards, want)
			}
		})
	}

	t.Run("multiple index files", func(t *testing.T) {
		dir := t.TempDir()
		writeShard(dir, shard1)
		for _, name := range []string{"model.safetensors.index.json", "alt.safetensors.index.json"} {
			if err := os.WriteFile(filepath.Join(dir, name), []byte(`{"weight_map":{"a.w":"`+shard1+`"}}`), 0o644); err != nil {
				t.Fatalf("writing index %s: %v", name, err)
			}
		}
		if _, _, err := DiscoverShards(dir); err == nil || !strings.Contains(err.Error(), "multiple index files") {
			t.Fatalf("expected multiple-index error, got %v", err)
		}
	})

	t.Run("nonexistent dir", func(t *testing.T) {
		if _, _, err := DiscoverShards(filepath.Join(t.TempDir(), "nope")); err == nil {
			t.Fatal("expected error for nonexistent dir")
		}
	})
}

func TestResolveInput(t *testing.T) {
	const (
		shard1 = "model.safetensors-00001-of-00002.safetensors"
		shard2 = "model.safetensors-00002-of-00002.safetensors"
	)

	writeShard := func(t *testing.T, dir, name string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(dir, name), []byte("fake shard data"), 0o644); err != nil {
			t.Fatalf("writing shard %s: %v", name, err)
		}
	}

	tests := []struct {
		name      string
		setup     func(t *testing.T, dir string) // create the inputs in dir
		target    func(dir string) string        // path to pass to ResolveInput
		wantBase  []string                       // expected shard basenames, in processing order
		wantIndex bool                           // expect a non-nil index
		wantIsDir bool                           // expect the input to be classified as a directory
		wantErr   string                         // substring expected in the error; "" = expect success
	}{
		{
			name:     "existing file",
			setup:    func(t *testing.T, dir string) { writeShard(t, dir, "model.safetensors") },
			target:   func(dir string) string { return filepath.Join(dir, "model.safetensors") },
			wantBase: []string{"model.safetensors"},
		},
		{
			name: "two-shard dir with index",
			setup: func(t *testing.T, dir string) {
				writeShard(t, dir, shard2)
				writeShard(t, dir, shard1)
				if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"),
					[]byte(`{"weight_map":{"b.w":"`+shard2+`","a.w":"`+shard1+`"}}`), 0o644); err != nil {
					t.Fatalf("writing index: %v", err)
				}
			},
			target:    func(dir string) string { return dir },
			wantBase:  []string{shard1, shard2},
			wantIndex: true,
			wantIsDir: true,
		},
		{
			name:      "dir without index",
			setup:     func(t *testing.T, dir string) { writeShard(t, dir, shard2); writeShard(t, dir, shard1) },
			target:    func(dir string) string { return dir },
			wantBase:  []string{shard1, shard2},
			wantIsDir: true,
		},
		{
			name:    "nonexistent path",
			setup:   func(t *testing.T, dir string) {},
			target:  func(dir string) string { return filepath.Join(dir, "nope.safetensors") },
			wantErr: "nope.safetensors",
		},
		{
			name:    "dir with no safetensors files",
			setup:   func(t *testing.T, dir string) { writeShard(t, dir, "README.txt") },
			target:  func(dir string) string { return dir },
			wantErr: "no .safetensors files",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			tt.setup(t, dir)

			got, idx, isDir, err := ResolveInput(tt.target(dir))
			if tt.wantErr != "" {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil (shards=%v)", tt.wantErr, got)
				}
				if !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("error %q does not contain %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if (idx != nil) != tt.wantIndex {
				t.Errorf("index non-nil = %v, want %v", idx != nil, tt.wantIndex)
			}
			if isDir != tt.wantIsDir {
				t.Errorf("isDir = %v, want %v", isDir, tt.wantIsDir)
			}

			absDir, err := filepath.Abs(dir)
			if err != nil {
				t.Fatalf("abs dir: %v", err)
			}
			want := make([]string, len(tt.wantBase))
			for i, b := range tt.wantBase {
				want[i] = filepath.Join(absDir, b)
			}
			if !reflect.DeepEqual(got, want) {
				t.Errorf("shards = %v, want %v", got, want)
			}
		})
	}

	t.Run("neither file nor directory", func(t *testing.T) {
		fifo := filepath.Join(t.TempDir(), "pipe")
		if out, err := exec.Command("mkfifo", fifo).CombinedOutput(); err != nil {
			t.Fatalf("mkfifo %s: %v: %s", fifo, err, out)
		}
		if _, _, _, err := ResolveInput(fifo); err == nil || !strings.Contains(err.Error(), "not a regular file or directory") {
			t.Fatalf("expected not-file-or-dir error, got %v", err)
		}
	})
}

func TestResolveOutput(t *testing.T) {
	tests := []struct {
		name       string
		setup      func(t *testing.T, dir string) // create the output path in dir
		target     func(dir string) string        // path to pass to ResolveOutput
		inputIsDir bool                           // whether -in was a model directory
		wantMode   string                         // expected mode; "" when wantErr != ""
		wantErr    string                         // substring expected in the error; "" = expect success
	}{
		{
			name:       "new file path",
			target:     func(dir string) string { return filepath.Join(dir, "out.safetensors") },
			inputIsDir: false,
			wantMode:   OutputSingle,
		},
		{
			name:       "nonexistent dir-looking path stays single-file",
			target:     func(dir string) string { return filepath.Join(dir, "out-dir") },
			inputIsDir: true,
			wantMode:   OutputSingle,
		},
		{
			name: "existing file is refused",
			setup: func(t *testing.T, dir string) {
				if err := os.WriteFile(filepath.Join(dir, "out.safetensors"), []byte("old"), 0o644); err != nil {
					t.Fatalf("writing file: %v", err)
				}
			},
			target:     func(dir string) string { return filepath.Join(dir, "out.safetensors") },
			inputIsDir: true,
			wantErr:    "refusing to overwrite",
		},
		{
			name: "existing dir with model-directory input",
			setup: func(t *testing.T, dir string) {
				if err := os.MkdirAll(filepath.Join(dir, "out-dir"), 0o755); err != nil {
					t.Fatalf("making dir: %v", err)
				}
			},
			target:     func(dir string) string { return filepath.Join(dir, "out-dir") },
			inputIsDir: true,
			wantMode:   OutputMulti,
		},
		{
			name: "existing dir with single-file input",
			setup: func(t *testing.T, dir string) {
				if err := os.MkdirAll(filepath.Join(dir, "out-dir"), 0o755); err != nil {
					t.Fatalf("making dir: %v", err)
				}
			},
			target:     func(dir string) string { return filepath.Join(dir, "out-dir") },
			inputIsDir: false,
			wantErr:    "multi-file output requires a model-directory input",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			if tt.setup != nil {
				tt.setup(t, dir)
			}

			mode, target, err := ResolveOutput(tt.target(dir), tt.inputIsDir)
			if tt.wantErr != "" {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil (mode=%s target=%s)", tt.wantErr, mode, target)
				}
				if !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("error %q does not contain %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if mode != tt.wantMode {
				t.Errorf("mode = %q, want %q", mode, tt.wantMode)
			}
			if target != tt.target(dir) {
				t.Errorf("target = %q, want %q", target, tt.target(dir))
			}
		})
	}

	t.Run("neither file nor directory", func(t *testing.T) {
		fifo := filepath.Join(t.TempDir(), "pipe")
		if out, err := exec.Command("mkfifo", fifo).CombinedOutput(); err != nil {
			t.Fatalf("mkfifo %s: %v: %s", fifo, err, out)
		}
		if _, _, err := ResolveOutput(fifo, true); err == nil || !strings.Contains(err.Error(), "not a regular file or directory") {
			t.Fatalf("expected not-file-or-dir error, got %v", err)
		}
	})
}
