package stconv

import (
	"fmt"
	"path/filepath"
)

// ShardOut is the planned output for one input shard: the output file's
// name, its complete header (data offsets relative to this shard's data
// block, starting at 0), and the weight_map entries of the tensors it
// holds.
type ShardOut struct {
	Name      string
	Header    *Header
	WeightMap map[string]string
}

// PlanShardOutput splits a global conversion plan into one output shard
// per input shard, plus the index that ties them together.
//
// Each input shard's tensors (and their ".scale" siblings) keep their
// relative order in the output shard, output offsets restart at 0 per
// shard, and each shard's header carries that input shard's __metadata__
// block (the per-shard analogue of the merged file's first-shard rule).
// Output shard filenames are the input shards' basenames, so a converted
// model directory is predictable and diffable against the input.
//
// The returned output index is planned next to the shards, under the
// input index's basename. Its weight_map keeps the input index's
// tensor-to-shard assignment for every planned tensor (a ".scale" sibling
// gets its owner's shard; a tensor missing from the input index -
// defensive, a consistent index names every tensor - goes to the shard
// that holds it). Its metadata is a copy of the input index's metadata
// (e.g. HF's "architectures"), overriding only total_size, which is
// recomputed from the plan: the input's total_size describes the input
// and would be wrong for the output.
//
// inIndex is required: multi-file output exists to replicate a
// model-directory input's sharding, and the tensor-to-shard assignments
// come from its weight_map.
func PlanShardOutput(plans []tensorPlan, inIndex *Index, outDir string) ([]ShardOut, *Index, error) {
	if inIndex == nil {
		return nil, nil, fmt.Errorf("multi-file output requires an input index")
	}
	if inIndex.Path == "" {
		return nil, nil, fmt.Errorf("input index has no file path; load it with LoadIndex")
	}
	if len(plans) == 0 {
		return nil, nil, fmt.Errorf("multi-file output: no tensors planned")
	}

	// Plans are in shard order, then in-shard header order, so grouping by
	// first appearance of srcShard preserves the input shard order.
	var order []int
	byShard := make(map[int][]int)
	for i, p := range plans {
		if _, seen := byShard[p.srcShard]; !seen {
			order = append(order, p.srcShard)
		}
		byShard[p.srcShard] = append(byShard[p.srcShard], i)
	}

	outIndex := &Index{
		Path:      filepath.Join(outDir, filepath.Base(inIndex.Path)),
		WeightMap: make(map[string]string, len(plans)),
		// The output index's weight_map is written in file order (shard
		// order, then within each owner: before-sibs, owner, after-sibs -
		// consistent with the on-disk header layout).
		weightMapOrder: make([]string, 0, len(plans)),
	}
	shards := make([]ShardOut, 0, len(order))
	seenNames := make(map[string]bool, len(order))

	var total int64
	for _, src := range order {
		first := byShard[src][0]
		sh := ShardOut{
			Name:      plans[first].srcName,
			Header:    &Header{Metadata: plans[first].srcMetadata},
			WeightMap: make(map[string]string, len(byShard[src])),
		}
		if seenNames[sh.Name] {
			return nil, nil, fmt.Errorf("duplicate output shard name %q (input shards in different directories share a basename)", sh.Name)
		}
		seenNames[sh.Name] = true

		for _, i := range byShard[src] {
			p := plans[i]
			// A sibling's weight_map entry is its owner's shard (defensive
			// rule: a consistent index names the owner, and the sibling
			// lives with it).
			shardName, ok := inIndex.WeightMap[p.name]
			if !ok {
				shardName = sh.Name
			}

			// Emit in file order: before-sibs, then the owner, then
			// after-sibs. Every one gets a weight_map entry, and each
			// sibling's bytes count toward this shard's total_size.
			for _, s := range p.sibs {
				if s.before {
					addSibling(&sh, outIndex, p.name, s, shardName)
					total += s.byteLen()
				}
			}
			appendHeaderEntry(sh.Header, p.name, p.outDType, p.srcInfo.Shape, p.outLen)
			total += p.outLen
			sh.WeightMap[p.name] = shardName
			outIndex.WeightMap[p.name] = shardName
			outIndex.weightMapOrder = append(outIndex.weightMapOrder, p.name)
			for _, s := range p.sibs {
				if !s.before {
					addSibling(&sh, outIndex, p.name, s, shardName)
					total += s.byteLen()
				}
			}
		}
		shards = append(shards, sh)
	}

	// Merge the input index's metadata, overriding only total_size (which
	// is computed from the plan, never copied: the input's value describes
	// the input). A shallow top-level copy is enough; the values are
	// read-only here. Memory is O(index metadata size), trivially bounded.
	outIndex.Metadata = make(map[string]any, len(inIndex.Metadata)+1)
	for k, v := range inIndex.Metadata {
		outIndex.Metadata[k] = v
	}
	outIndex.Metadata["total_size"] = total
	return shards, outIndex, nil
}

// addSibling appends one sibling's header entry to a shard (in file order)
// and records its weight_map entry in both the shard and the output index.
// Every sibling maps to its owner's shard (ownerShard).
func addSibling(sh *ShardOut, outIndex *Index, owner string, s sib, ownerShard string) {
	sibName := owner + s.suffix
	appendHeaderEntry(sh.Header, sibName, s.dtype, sibShape(s), s.byteLen())
	sh.WeightMap[sibName] = ownerShard
	outIndex.WeightMap[sibName] = ownerShard
	outIndex.weightMapOrder = append(outIndex.weightMapOrder, sibName)
}
