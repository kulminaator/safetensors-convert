package stconv

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// fixtureIndexName is the index file inside the committed 2-shard fixture.
const fixtureIndexName = "model.safetensors.index.json"

// checkContiguous verifies a shard header's data offsets start at 0 and
// are contiguous (each entry starts where the previous one ends).
func checkContiguous(t *testing.T, h *Header) {
	t.Helper()
	var end int64
	for i, e := range h.Tensors {
		if e.Info.DataOffsets[0] != end {
			t.Errorf("header[%d] (%s) starts at %d, want %d (offsets must start at 0 and be contiguous)",
				i, e.Name, e.Info.DataOffsets[0], end)
		}
		end = e.Info.DataOffsets[1]
	}
}

func TestPlanShardOutput(t *testing.T) {
	inIdx, err := LoadIndex(filepath.Join(fixtureDir, fixtureIndexName))
	if err != nil {
		t.Fatalf("LoadIndex: %v", err)
	}
	outDir := t.TempDir()

	cases := []struct {
		name    string
		def     TargetKind
		want1   []outEntry // expected shard 1 header
		want2   []outEntry // expected shard 2 header
		wantTot int64      // expected output index total_size
	}{
		{
			name: "int8 (scales follow owners)",
			def:  TargetInt8,
			want1: []outEntry{
				{nameUpProj, DTypeI8, [2]int64{0, 8}},
				{nameUpProj + ".scale", DTypeF32, [2]int64{8, 12}},
				{nameQProj, DTypeI8, [2]int64{12, 16}},
				{nameQProj + ".scale", DTypeF32, [2]int64{16, 20}},
			},
			want2: []outEntry{
				{nameNorm, DTypeI8, [2]int64{0, 4}},
				{nameNorm + ".scale", DTypeF32, [2]int64{4, 8}},
				{namePosIDs, DTypeI32, [2]int64{8, 24}},
			},
			wantTot: 44,
		},
		{
			name: "fp8_e4m3 (no scales)",
			def:  TargetFP8E4M3,
			want1: []outEntry{
				{nameUpProj, DTypeF8E4M3, [2]int64{0, 8}},
				{nameQProj, DTypeF8E4M3, [2]int64{8, 12}},
			},
			want2: []outEntry{
				{nameNorm, DTypeF8E4M3, [2]int64{0, 4}},
				{namePosIDs, DTypeI32, [2]int64{4, 20}},
			},
			wantTot: 32,
		},
		{
			name: "int4 (scalar scale follows owner)",
			def:  TargetInt4,
			want1: []outEntry{
				{nameUpProj, DTypeU8, [2]int64{0, 4}},
				{nameUpProj + ".scale", DTypeF32, [2]int64{4, 8}},
				{nameQProj, DTypeU8, [2]int64{8, 10}},
				{nameQProj + ".scale", DTypeF32, [2]int64{10, 14}},
			},
			want2: []outEntry{
				{nameNorm, DTypeU8, [2]int64{0, 2}},
				{nameNorm + ".scale", DTypeF32, [2]int64{2, 6}},
				{namePosIDs, DTypeI32, [2]int64{6, 22}},
			},
			wantTot: 36,
		},
		{
			name: "mxfp4 (block scale before owner)",
			def:  TargetMxFP4,
			want1: []outEntry{
				{nameUpProj + ".block_scale", DTypeU8, [2]int64{0, 1}},
				{nameUpProj, DTypeU8, [2]int64{1, 5}},
				{nameQProj + ".block_scale", DTypeU8, [2]int64{5, 6}},
				{nameQProj, DTypeU8, [2]int64{6, 8}},
			},
			want2: []outEntry{
				{nameNorm + ".block_scale", DTypeU8, [2]int64{0, 1}},
				{nameNorm, DTypeU8, [2]int64{1, 3}},
				{namePosIDs, DTypeI32, [2]int64{3, 19}},
			},
			wantTot: 27,
		},
		{
			name: "nvfp4 (global + block scales before owner)",
			def:  TargetNVFP4,
			want1: []outEntry{
				{nameUpProj + ".global_scale", DTypeF32, [2]int64{0, 4}},
				{nameUpProj + ".block_scale", DTypeF8E4M3, [2]int64{4, 5}},
				{nameUpProj, DTypeU8, [2]int64{5, 9}},
				{nameQProj + ".global_scale", DTypeF32, [2]int64{9, 13}},
				{nameQProj + ".block_scale", DTypeF8E4M3, [2]int64{13, 14}},
				{nameQProj, DTypeU8, [2]int64{14, 16}},
			},
			want2: []outEntry{
				{nameNorm + ".global_scale", DTypeF32, [2]int64{0, 4}},
				{nameNorm + ".block_scale", DTypeF8E4M3, [2]int64{4, 5}},
				{nameNorm, DTypeU8, [2]int64{5, 7}},
				{namePosIDs, DTypeI32, [2]int64{7, 23}},
			},
			wantTot: 39,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			plans, _, err := planModel(ConvertOptions{InputShards: multiShardPaths(), Default: tc.def})
			if err != nil {
				t.Fatalf("planModel: %v", err)
			}
			shards, outIdx, err := PlanShardOutput(plans, inIdx, outDir)
			if err != nil {
				t.Fatalf("PlanShardOutput: %v", err)
			}

			// Shard count and names match the input, in input order.
			if len(shards) != 2 {
				t.Fatalf("got %d shards, want 2", len(shards))
			}
			if shards[0].Name != multiShard1 || shards[1].Name != multiShard2 {
				t.Fatalf("shard names = %q, %q; want %q, %q",
					shards[0].Name, shards[1].Name, multiShard1, multiShard2)
			}

			// Per-shard headers: pinned entries, offsets start at 0 and are
			// contiguous, and no __metadata__ (the fixture shards have none).
			checkOutHeader(t, shards[0].Header, tc.want1)
			checkOutHeader(t, shards[1].Header, tc.want2)
			checkContiguous(t, shards[0].Header)
			checkContiguous(t, shards[1].Header)
			if shards[0].Header.Metadata != nil || shards[1].Header.Metadata != nil {
				t.Errorf("shard metadata = %v, %v; want nil (fixture has none)",
					shards[0].Header.Metadata, shards[1].Header.Metadata)
			}

			// Every planned tensor (incl. ".scale" siblings) appears in
			// exactly one shard's header, and no others do.
			seen := make(map[string]int)
			for _, sh := range shards {
				for _, e := range sh.Header.Tensors {
					seen[e.Name]++
				}
			}
			wantCount := 0
			for _, p := range plans {
				if seen[p.name] != 1 {
					t.Errorf("planned tensor %s appears in %d shard headers, want 1", p.name, seen[p.name])
				}
				wantCount++
				for _, s := range p.sibs {
					if seen[p.name+s.suffix] != 1 {
						t.Errorf("planned sibling %s%s appears in %d shard headers, want 1", p.name, s.suffix, seen[p.name+s.suffix])
					}
					wantCount++
				}
			}
			if len(seen) != wantCount {
				t.Errorf("headers hold %d distinct tensors, want %d (all planned, no extras)", len(seen), wantCount)
			}

			// weight_map: same assignments as the input index for every
			// planned tensor; a scale sibling shares its owner's shard; the
			// per-shard maps are the subsets of the output index's map.
			want := make(map[string]string)
			for _, p := range plans {
				want[p.name] = inIdx.WeightMap[p.name]
				for _, s := range p.sibs {
					want[p.name+s.suffix] = inIdx.WeightMap[p.name]
				}
			}
			if !reflect.DeepEqual(outIdx.WeightMap, want) {
				t.Errorf("output weight_map = %v, want %v", outIdx.WeightMap, want)
			}
			for _, sh := range shards {
				if len(sh.WeightMap) != len(sh.Header.Tensors) {
					t.Errorf("shard %s: %d weight_map entries, want %d (one per header tensor)",
						sh.Name, len(sh.WeightMap), len(sh.Header.Tensors))
				}
				for name, shardName := range sh.WeightMap {
					if shardName != sh.Name {
						t.Errorf("shard %s maps %s to %s, want itself", sh.Name, name, shardName)
					}
					if outIdx.WeightMap[name] != shardName {
						t.Errorf("shard %s weight_map[%s] = %s, output index says %s",
							sh.Name, name, shardName, outIdx.WeightMap[name])
					}
				}
			}

			// total_size equals the sum of the per-shard header byte
			// lengths, i.e. is computed from the plan. The input index's
			// total_size is 48 in both cases, so a match with wantTot also
			// proves it was not copied.
			var sum int64
			for _, sh := range shards {
				sum += sh.Header.Tensors[len(sh.Header.Tensors)-1].Info.DataOffsets[1]
			}
			if got := outIdx.Metadata["total_size"]; got != sum {
				t.Errorf("total_size = %v, want %d (sum of header byte lengths)", got, sum)
			}
			if got := outIdx.Metadata["total_size"]; got != tc.wantTot {
				t.Errorf("total_size = %v, want %d", got, tc.wantTot)
			}

			// The output index lands next to the shards, under the input
			// index's basename.
			if want := filepath.Join(outDir, fixtureIndexName); outIdx.Path != want {
				t.Errorf("output index path = %q, want %q", outIdx.Path, want)
			}
		})
	}
}

// TestPlanShardOutputIndexKeyOrder pins that the output index's weight_map
// key order follows the on-disk file layout: within each owner, before-sibs
// come before the owner and after-sibs after it, in shard order.
func TestPlanShardOutputIndexKeyOrder(t *testing.T) {
	inIdx, err := LoadIndex(filepath.Join(fixtureDir, fixtureIndexName))
	if err != nil {
		t.Fatalf("LoadIndex: %v", err)
	}

	cases := []struct {
		name string
		def  TargetKind
		want []string
	}{
		{
			"int8 (scale after owner)", TargetInt8,
			[]string{nameUpProj, nameUpProj + ".scale", nameQProj, nameQProj + ".scale",
				nameNorm, nameNorm + ".scale", namePosIDs},
		},
		{
			"mxfp4 (block scale before owner)", TargetMxFP4,
			[]string{nameUpProj + ".block_scale", nameUpProj, nameQProj + ".block_scale", nameQProj,
				nameNorm + ".block_scale", nameNorm, namePosIDs},
		},
		{
			"nvfp4 (global + block scales before owner)", TargetNVFP4,
			[]string{nameUpProj + ".global_scale", nameUpProj + ".block_scale", nameUpProj,
				nameQProj + ".global_scale", nameQProj + ".block_scale", nameQProj,
				nameNorm + ".global_scale", nameNorm + ".block_scale", nameNorm, namePosIDs},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			plans, _, err := planModel(ConvertOptions{InputShards: multiShardPaths(), Default: tc.def})
			if err != nil {
				t.Fatalf("planModel: %v", err)
			}
			_, outIdx, err := PlanShardOutput(plans, inIdx, t.TempDir())
			if err != nil {
				t.Fatalf("PlanShardOutput: %v", err)
			}
			if !reflect.DeepEqual(outIdx.weightMapOrder, tc.want) {
				t.Errorf("weight_map key order = %v, want %v", outIdx.weightMapOrder, tc.want)
			}
		})
	}
}

func TestPlanShardOutputErrors(t *testing.T) {
	plans, _, err := planModel(ConvertOptions{InputShards: multiShardPaths(), Default: TargetNone})
	if err != nil {
		t.Fatalf("planModel: %v", err)
	}
	inIdx, err := LoadIndex(filepath.Join(fixtureDir, fixtureIndexName))
	if err != nil {
		t.Fatalf("LoadIndex: %v", err)
	}
	outDir := t.TempDir()

	t.Run("no input index", func(t *testing.T) {
		if _, _, err := PlanShardOutput(plans, nil, outDir); err == nil || !strings.Contains(err.Error(), "requires an input index") {
			t.Fatalf("expected missing-index error, got %v", err)
		}
	})
	t.Run("index without path", func(t *testing.T) {
		if _, _, err := PlanShardOutput(plans, &Index{WeightMap: map[string]string{}}, outDir); err == nil || !strings.Contains(err.Error(), "no file path") {
			t.Fatalf("expected no-path error, got %v", err)
		}
	})
	t.Run("no planned tensors", func(t *testing.T) {
		if _, _, err := PlanShardOutput(nil, inIdx, outDir); err == nil || !strings.Contains(err.Error(), "no tensors planned") {
			t.Fatalf("expected empty-plans error, got %v", err)
		}
	})
}

func TestPlanShardOutputPerShardMetadata(t *testing.T) {
	dir := t.TempDir()
	writeMetaShard(t, filepath.Join(dir, "a.safetensors"), map[string]string{"origin": "a"}, "ta")
	writeMetaShard(t, filepath.Join(dir, "b.safetensors"), map[string]string{"origin": "b"}, "tb")

	loadIndex := func(name, content string) *Index {
		t.Helper()
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatalf("writing %s: %v", name, err)
		}
		idx, err := LoadIndex(path)
		if err != nil {
			t.Fatalf("LoadIndex(%s): %v", name, err)
		}
		return idx
	}

	plans, _, err := planModel(ConvertOptions{
		InputShards: []string{filepath.Join(dir, "a.safetensors"), filepath.Join(dir, "b.safetensors")},
		Default:     TargetNone,
	})
	if err != nil {
		t.Fatalf("planModel: %v", err)
	}

	t.Run("each shard keeps its own __metadata__", func(t *testing.T) {
		inIdx := loadIndex("model.safetensors.index.json",
			`{"weight_map":{"ta":"a.safetensors","tb":"b.safetensors"}}`)
		shards, _, err := PlanShardOutput(plans, inIdx, t.TempDir())
		if err != nil {
			t.Fatalf("PlanShardOutput: %v", err)
		}
		if want := map[string]string{"origin": "a"}; !reflect.DeepEqual(shards[0].Header.Metadata, want) {
			t.Errorf("shard 0 metadata = %v, want %v (its input shard's block)", shards[0].Header.Metadata, want)
		}
		if want := map[string]string{"origin": "b"}; !reflect.DeepEqual(shards[1].Header.Metadata, want) {
			t.Errorf("shard 1 metadata = %v, want %v (its input shard's block)", shards[1].Header.Metadata, want)
		}
	})

	t.Run("tensor missing from input index falls back to its shard", func(t *testing.T) {
		inIdx := loadIndex("partial.index.json", `{"weight_map":{"ta":"a.safetensors"}}`)
		_, outIdx, err := PlanShardOutput(plans, inIdx, t.TempDir())
		if err != nil {
			t.Fatalf("PlanShardOutput: %v", err)
		}
		if got := outIdx.WeightMap["tb"]; got != "b.safetensors" {
			t.Errorf("tb weight_map = %q, want %q (the shard that holds it)", got, "b.safetensors")
		}
	})
}
