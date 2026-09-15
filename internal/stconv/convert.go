package stconv

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
)

// DefaultChunkElems bounds how many source elements are held in memory at
// once while converting a tensor. Peak memory for the conversion path is
// O(DefaultChunkElems), independent of tensor size or file size - a
// multi-hundred-GB model is processed the same way as a 1KB one.
const DefaultChunkElems = 1 << 20 // e.g. 2MB per chunk for fp16/bf16 input

// ConvertOptions controls a conversion run. The input is either a single
// file (InputPath, via ConvertFile) or an ordered list of shard files
// (InputShards, via ConvertModel); every shard is a self-contained
// safetensors file with its own header. The output is either a single file
// (OutputPath) or a directory of shard files plus an index (OutputDir),
// which is mutually exclusive with OutputPath.
type ConvertOptions struct {
	InputPath   string     // input file for ConvertFile
	InputShards []string   // ordered input shard files for ConvertModel
	InputIndex  *Index     // input model index; required when OutputDir is set
	OutputPath  string     // single-file output (mutually exclusive with OutputDir)
	OutputDir   string     // multi-file output directory (mutually exclusive with OutputPath)
	Config      *Config    // may be nil
	Default     TargetKind // used when Config is nil, or Config has no default and no rule matches
	Protect     bool       // apply the built-in default precision policy (protect.go); the CLI sets this explicitly per run
	MinElems    int        // tensors with fewer elements than this are never converted (0 disables)
	ChunkElems  int        // 0 -> DefaultChunkElems
	Progress    *Progress  // live progress reporter (nil = no progress output); the CLI sets it when -quiet is off
}

// TensorStat reports what actually happened to one tensor during a run.
type TensorStat struct {
	Name       string
	Source     string // input shard filename; empty for single-file input
	FromDType  DType
	ToDType    DType
	NumElems   int64
	Scale      float32 // only meaningful for int8
	Note       string  // optional extra shown in the report when non-empty (e.g. convrot row count, nvfp4 global scale)
	SkippedWhy string  // non-empty if the tensor was left unconverted
	// ProtectOverride is true when an explicit config rule converted a
	// tensor the default policy would have protected. Not shown in the
	// report; the CLI counts these and prints a one-line stderr warning.
	ProtectOverride bool
}

// tensorPlan is decided entirely from the input header (dtype + shape),
// with no tensor data read. Because output byte length only depends on
// element count and target dtype, every output offset can be computed
// up front, which is what lets us write the header once and then stream
// data straight through in file order.
type tensorPlan struct {
	name            string
	srcShard        int               // index into ConvertOptions.InputShards; srcAbsOffset is relative to that shard file
	srcName         string            // input shard filename (basename); multi-file output keeps it as the output shard name
	srcMetadata     map[string]string // input shard's __metadata__ block (nil if none); shared by all plans from that shard
	srcInfo         TensorInfo
	srcAbsOffset    int64 // absolute offset of source tensor bytes in its source shard file
	srcLen          int64 // source tensor byte length
	numElems        int64
	target          TargetKind
	outDType        DType
	outShape        []int64 // output header shape (packed/halved for the packed 4-bit targets int4/nvfp4, source shape otherwise)
	outLen          int64   // planned output byte length of this tensor (sibling tensors not included)
	protectReason   string  // non-empty => the passthrough came from the default protection policy, not from config/default or a mechanical constraint
	protectOverride bool    // explicit config rule converted a tensor the policy would protect; surfaced in TensorStat for the CLI warning
	skippedWhy      string  // non-empty => passthrough copy, no conversion
	sibs            []sib   // sibling tensors this plan emits (nil for passthrough)
}

// sib is one sibling tensor emitted alongside a converted tensor's owner.
// suffix is appended to the owner's name; dtype/numElems fix its header
// entry; before controls file order (true => emitted before the owner).
//
// Scalar scales (int8, int4) keep the existing owner-then-.scale order
// (before=false); vector scales (convrot row scales, mxfp4/nvfp4 block
// scales) are emitted before the owner (before=true), because the
// streaming passes compute them in an earlier pass over the data - writing
// them after would require buffering a per-row/per-block vector that grows
// with tensor size, which the memory rules forbid.
type sib struct {
	suffix   string
	dtype    DType
	numElems int64
	before   bool
}

// byteLen is the sibling's total byte length (numElems x element size).
func (s sib) byteLen() int64 {
	bs, _ := s.dtype.ByteSize()
	return s.numElems * int64(bs)
}

// sibShape is the flat header shape for a sibling: a scalar is
// []int64{} (like the existing int8 ".scale"), a row/block vector is
// []int64{numElems}.
func sibShape(s sib) []int64 {
	if s.numElems == 1 {
		return []int64{}
	}
	return []int64{s.numElems}
}

// sibBytes is the total byte length of all of p's sibling tensors.
func (p tensorPlan) sibBytes() int64 {
	var total int64
	for _, s := range p.sibs {
		total += s.byteLen()
	}
	return total
}

// totalOutLen is p's full output byte length: the owner plus all siblings.
func (p tensorPlan) totalOutLen() int64 {
	return p.outLen + p.sibBytes()
}

// planSiblings is the single source of truth for which sibling tensors a
// converting target emits, with their dtype, element count, and file order.
// It is pure: decided entirely from the target and the input shape /
// element count, with no tensor data read.
func planSiblings(target TargetKind, shape []int64, numElems int64) []sib {
	switch target {
	case TargetInt8, TargetInt4:
		return []sib{{suffix: ".scale", dtype: DTypeF32, numElems: 1, before: false}}
	case TargetInt8ConvRot:
		// One F32 scale per row (first dimension); a 1-D tensor is one row.
		rows := int64(1)
		if len(shape) >= 2 {
			rows = shape[0]
		}
		return []sib{{suffix: ".scale", dtype: DTypeF32, numElems: rows, before: true}}
	case TargetMxFP4:
		return []sib{{suffix: ".block_scale", dtype: DTypeU8, numElems: (numElems + 31) / 32, before: true}}
	case TargetNVFP4:
		return []sib{
			{suffix: ".global_scale", dtype: DTypeF32, numElems: 1, before: true},
			{suffix: ".block_scale", dtype: DTypeF8E4M3, numElems: (numElems + 15) / 16, before: true},
		}
	}
	return nil
}

// packedOwnerShape returns the header shape for a packed 4-bit owner
// tensor (int4/nvfp4) whose data stores 2 elements per U8 byte:
// prod(result) == (numElems+1)/2, the spec byte-count invariant.
//
// The rule is GPTQ/AWQ-style: a zero-element tensor keeps its shape
// (prod = 0 = packed byte count); a 0-D tensor (1 element) becomes [1];
// otherwise the last dimension is halved when it is even (the usual
// case, where packing is contiguous along the last dim), else the
// nearest even dimension scanning backwards is halved (e.g. [32,1] ->
// [16,1]); if every dimension is odd the element count is odd, so the
// result is the 1-D ceil form [(numElems+1)/2] (e.g. [3,3] -> [5]).
// The packed byte stream is identical regardless of which dimension is
// halved - packing is flat row-major, low nibble = element 2i - so only
// the header shape label changes and the rule only needs to satisfy the
// byte-count invariant.
func packedOwnerShape(shape []int64, numElems int64) []int64 {
	out := make([]int64, len(shape))
	copy(out, shape)
	if numElems == 0 {
		return out
	}
	if len(shape) == 0 {
		return []int64{1}
	}
	if shape[len(shape)-1]%2 == 0 {
		out[len(shape)-1] /= 2
		return out
	}
	for i := len(shape) - 2; i >= 0; i-- {
		if shape[i]%2 == 0 {
			out[i] /= 2
			return out
		}
	}
	return []int64{(numElems + 1) / 2}
}

// ConvertFile reads a single safetensors file, converts tensors according
// to opts, and writes a new safetensors file. It delegates to
// ConvertModel with a one-shard input list, so single-file behavior is
// unchanged by the multi-shard support.
func ConvertFile(opts ConvertOptions) ([]TensorStat, error) {
	opts.InputShards = []string{opts.InputPath}
	return ConvertModel(opts)
}

// ConvertModel converts one or more input shards into either a single
// merged output safetensors file (OutputPath) or a directory of shard
// files plus an index (OutputDir, which requires InputIndex). Tensors are
// planned and emitted in shard order, then in-shard header order; since
// each shard is a self-contained safetensors file, a tensor's source
// offset is relative to its own shard's data block (see
// tensorPlan.srcShard). Tensor data is streamed in bounded chunks in both
// directions - nothing near the size of the full model is ever held in
// memory at once.
//
// A failure after the output files are created closes and removes the
// partially written outputs (the single file, or every output shard plus
// the index if it was written), so a failed run leaves nothing loadable
// behind; input files are never touched.
func ConvertModel(opts ConvertOptions) ([]TensorStat, error) {
	if len(opts.InputShards) == 0 {
		return nil, fmt.Errorf("no input shards given")
	}
	// The output destination is validated here, in the one place both
	// ConvertFile and ConvertModel run through.
	if opts.OutputPath != "" && opts.OutputDir != "" {
		return nil, fmt.Errorf("OutputPath and OutputDir are mutually exclusive; set exactly one")
	}

	chunkElems := opts.ChunkElems
	if chunkElems <= 0 {
		chunkElems = DefaultChunkElems
	}

	// ---- Pass 1: plan everything from metadata alone (no data read). ----
	plans, outHeader, err := planModel(opts)
	if err != nil {
		return nil, err
	}

	// The progress reporter (if any) counts tensors, and the plan is the
	// one place the run's tensor count is known: set the total before the
	// first tensor's Begin.
	if opts.Progress != nil {
		opts.Progress.SetTotal(len(plans))
	}

	// Multi-shard runs tag each stat with its source shard filename so the
	// report can show where a tensor came from; single-file runs leave it
	// empty, matching the pre-multi-shard report.
	sources := make([]string, len(opts.InputShards))
	if len(opts.InputShards) > 1 {
		for i, path := range opts.InputShards {
			sources[i] = filepath.Base(path)
		}
	}

	if opts.OutputDir != "" {
		return convertModelToShards(opts, plans, sources, chunkElems)
	}

	// ---- Single-file output: write the header once, up front; data
	// offsets are already final. ----
	out, err := CreateOutputFile(opts.OutputPath, outHeader)
	if err != nil {
		return nil, fmt.Errorf("creating output: %w", err)
	}
	// A failed run must not leave a partial output file behind: on any
	// error after this point, close the file and remove it. success is set
	// only on the final return, so every error path cleans up.
	success := false
	defer func() {
		// Close before remove: the file must be closed before it can be
		// unlinked, and on the failure path the close error is not what
		// matters - the run's error is.
		out.Close()
		if !success {
			os.Remove(opts.OutputPath)
		}
	}()

	bw := bufio.NewWriterSize(out, 4<<20) // 4MB write buffer, independent of chunkElems

	// ---- Pass 2: stream each tensor's data straight through. ----
	shards, err := openShards(opts.InputShards)
	if err != nil {
		return nil, err
	}
	defer closeFiles(shards)

	// One pass scratch for the whole run (see passScratch): allocated
	// here, before the tensor loop, and reused for every tensor -
	// tensors are processed strictly one at a time, so its buffers never
	// overlap. Its sizes depend only on chunkElems.
	scratch := newPassScratch(chunkElems)

	stats := make([]TensorStat, 0, len(plans))
	for _, p := range plans {
		stat, err := convertTensor(p, shards[p.srcShard], bw, chunkElems, scratch, opts.Progress)
		if err != nil {
			return nil, err
		}
		stat.Source = sources[p.srcShard]
		stats = append(stats, stat)
	}

	if err := bw.Flush(); err != nil {
		return nil, fmt.Errorf("flushing output: %w", err)
	}

	success = true
	return stats, nil
}

// ensureNoOverwrite stats every planned multi-file output path (all
// output shards plus the index) and refuses the run if any already
// exists, matching the single-file never-overwrite policy that
// ResolveOutput enforces on -out. Every target is checked up front, before
// any file is created, so a refused run touches nothing in the output
// directory and keeps the fail-before-creating-any-output property the
// input-open ordering already has.
func ensureNoOverwrite(paths ...string) error {
	for _, path := range paths {
		_, err := os.Stat(path)
		if err == nil {
			return fmt.Errorf("refusing to overwrite existing output file %s", path)
		}
		if !os.IsNotExist(err) {
			return fmt.Errorf("checking output path %s: %w", path, err)
		}
	}
	return nil
}

// convertModelToShards is the multi-file output branch of ConvertModel:
// one output shard file per input shard, each with its own header written
// once, up front (the same invariant as single-file output), plus the
// index JSON that ties the shards together. Tensors stream to the output
// shard that mirrors their input shard, in the same global order as the
// single-file pass - nothing here buffers beyond the per-shard write
// buffers, and the open fd count is bounded by the shard count on both
// the input and the output side.
//
// Planned output shard and index paths that already exist are refused up
// front, before any file is created: the tool never overwrites in place,
// the same policy the single-file branch gets from ResolveOutput.
//
// Any error after the shard files are created (streaming, flush, or index
// write) closes and removes every created output shard and, if it was
// written, the index, so a failed run leaves nothing loadable behind.
func convertModelToShards(opts ConvertOptions, plans []tensorPlan, sources []string, chunkElems int) ([]TensorStat, error) {
	shardOuts, outIndex, err := PlanShardOutput(plans, opts.InputIndex, opts.OutputDir)
	if err != nil {
		return nil, err
	}

	// The tool never overwrites in place: refuse any planned output shard
	// or index path that already exists, before creating anything (the
	// single-file branch gets the same guarantee from ResolveOutput).
	outPaths := make([]string, 0, len(shardOuts)+1)
	for _, sh := range shardOuts {
		outPaths = append(outPaths, filepath.Join(opts.OutputDir, sh.Name))
	}
	outPaths = append(outPaths, outIndex.Path)
	if err := ensureNoOverwrite(outPaths...); err != nil {
		return nil, err
	}

	// Inputs first, so a missing shard fails before any output is created.
	shards, err := openShards(opts.InputShards)
	if err != nil {
		return nil, err
	}
	defer closeFiles(shards)

	// One output file per shard, each with its complete header written
	// once, up front, before any data. A failed run must not leave
	// partial shards - or an index pointing at them - behind: on any
	// error after this point, close and remove every created output file
	// (and the index, if it was written). success is set only on the final
	// return, so every error path cleans up.
	outFiles := make([]*os.File, len(shardOuts))
	success := false
	defer func() {
		// Close before remove: a file must be closed before it can be
		// unlinked, and on the failure path the close/remove errors are
		// not what matters - the run's first error is. Removing the index
		// is a no-op if it was never written.
		closeFiles(outFiles)
		if !success {
			for i, f := range outFiles {
				if f != nil {
					os.Remove(filepath.Join(opts.OutputDir, shardOuts[i].Name))
				}
			}
			os.Remove(outIndex.Path)
		}
	}()
	for i, sh := range shardOuts {
		f, err := CreateOutputFile(filepath.Join(opts.OutputDir, sh.Name), sh.Header)
		if err != nil {
			return nil, fmt.Errorf("creating output shard %q: %w", sh.Name, err)
		}
		outFiles[i] = f
	}

	// Each plan writes to the output shard mirroring its input shard. Plans
	// are in shard order, so first appearance of srcShard matches
	// PlanShardOutput's shard order.
	outShardOf := make(map[int]int, len(shardOuts))
	next := 0
	for _, p := range plans {
		if _, ok := outShardOf[p.srcShard]; !ok {
			outShardOf[p.srcShard] = next
			next++
		}
	}
	if next != len(shardOuts) {
		return nil, fmt.Errorf("internal error: %d output shards planned, but the plan order spans %d source shards", len(shardOuts), next)
	}

	// One write buffer per output shard, same size as the single-file pass.
	bws := make([]*bufio.Writer, len(outFiles))
	for i, f := range outFiles {
		bws[i] = bufio.NewWriterSize(f, 4<<20)
	}

	// One pass scratch for the whole run (see passScratch): allocated
	// here, before the tensor loop, and reused for every tensor -
	// tensors are processed strictly one at a time, so its buffers never
	// overlap. Its sizes depend only on chunkElems.
	scratch := newPassScratch(chunkElems)

	stats := make([]TensorStat, 0, len(plans))
	for _, p := range plans {
		stat, err := convertTensor(p, shards[p.srcShard], bws[outShardOf[p.srcShard]], chunkElems, scratch, opts.Progress)
		if err != nil {
			return nil, err
		}
		stat.Source = sources[p.srcShard]
		stats = append(stats, stat)
	}

	for i, bw := range bws {
		if err := bw.Flush(); err != nil {
			return nil, fmt.Errorf("flushing output shard %q: %w", shardOuts[i].Name, err)
		}
	}

	// The index goes last: only once every shard's data is on disk is the
	// model directory complete and loadable.
	idxJSON, err := json.Marshal(outIndex)
	if err != nil {
		return nil, fmt.Errorf("marshaling output index: %w", err)
	}
	if err := os.WriteFile(outIndex.Path, idxJSON, 0o644); err != nil {
		return nil, fmt.Errorf("writing output index %s: %w", outIndex.Path, err)
	}

	success = true
	return stats, nil
}

// openShards opens every input shard file and returns them in input order.
// Each shard is kept open for the whole run: the fd count equals the shard
// count (small and bounded - a model is split into a handful of shards),
// so the streaming pass never churns opens. On error, the shards opened so
// far are closed.
func openShards(paths []string) ([]*os.File, error) {
	shards := make([]*os.File, len(paths))
	for i, path := range paths {
		f, err := os.Open(path)
		if err != nil {
			closeFiles(shards[:i])
			return nil, fmt.Errorf("opening input shard %q: %w", path, err)
		}
		shards[i] = f
	}
	return shards, nil
}

// closeFiles closes fs in reverse order, ignoring errors: it runs on
// cleanup paths where the caller's first error is what matters.
func closeFiles(fs []*os.File) {
	for i := len(fs) - 1; i >= 0; i-- {
		if fs[i] != nil {
			fs[i].Close()
		}
	}
}

// planModel plans every tensor of every input shard from headers alone
// (no tensor data read), in shard order, then in-shard header order, with
// output offsets accumulating globally across shards.
func planModel(opts ConvertOptions) ([]tensorPlan, *Header, error) {
	outHeader := &Header{}
	plans := make([]tensorPlan, 0)
	// The merged output is one header, so tensor names must be unique
	// across the whole planning loop, not just within each shard: a model
	// directory without an index (the glob fallback) or a hand-assembled
	// shard set can carry the same name twice, and both entries would
	// otherwise land in the output header as duplicate JSON keys - an
	// invalid safetensors file with no diagnostic.
	seen := make(map[string]bool)

	for srcShard, path := range opts.InputShards {
		header, dataStart, err := readShardHeader(path)
		if err != nil {
			return nil, nil, err
		}

		// The header is known here, so grow plans to exactly this shard's
		// tensor count instead of letting append re-allocate as it doubles.
		// Still O(total tensor count) memory.
		plans = slices.Grow(plans, len(header.Tensors))

		if srcShard == 0 {
			// The merged output keeps the first shard's __metadata__ block;
			// merging metadata maps across shards is out of scope.
			outHeader.Metadata = header.Metadata
		}

		base := filepath.Base(path)
		for _, entry := range header.Tensors {
			if seen[entry.Name] {
				return nil, nil, fmt.Errorf("duplicate tensor name %q (in shard %q and earlier)", entry.Name, path)
			}
			seen[entry.Name] = true
			plan, err := planTensor(opts, entry.Name, entry.Info, srcShard, dataStart, outHeader)
			if err != nil {
				return nil, nil, err
			}
			plan.srcName = base
			plan.srcMetadata = header.Metadata
			plans = append(plans, plan)
		}
	}

	return plans, outHeader, nil
}

// readShardHeader reads and parses one shard's header, closing the file
// before returning. Planning needs only the header (metadata-only rule);
// pass 2 re-opens shards as it streams.
func readShardHeader(path string) (*Header, int64, error) {
	in, err := os.Open(path)
	if err != nil {
		return nil, 0, fmt.Errorf("opening input shard %q: %w", path, err)
	}
	header, dataStart, readErr := ReadHeader(in)
	closeErr := in.Close()
	if readErr != nil {
		return nil, 0, fmt.Errorf("reading safetensors header from %q: %w", path, readErr)
	}
	if closeErr != nil {
		return nil, 0, fmt.Errorf("closing input shard %q: %w", path, closeErr)
	}
	return header, dataStart, nil
}

// planTensor decides what happens to one input tensor from its header
// entry alone (target dtype or passthrough reason), then appends the
// tensor's output header entries - its siblings (in file order) plus the
// owner - to outHeader, advancing the running output offset.
func planTensor(opts ConvertOptions, name string, info TensorInfo, srcShard int, dataStart int64, outHeader *Header) (tensorPlan, error) {
	numElems := int64(1)
	for _, d := range info.Shape {
		numElems *= d
	}

	plan := tensorPlan{
		name:         name,
		srcShard:     srcShard,
		srcInfo:      info,
		srcAbsOffset: dataStart + info.DataOffsets[0],
		srcLen:       info.DataOffsets[1] - info.DataOffsets[0],
		numElems:     numElems,
		outShape:     info.Shape,
	}
	if plan.srcLen < 0 {
		return plan, fmt.Errorf("tensor %q: invalid data_offsets %v", name, info.DataOffsets)
	}
	// A negative dimension makes numElems negative, which would silently
	// no-op the streaming loops (their done < numElems bounds never run)
	// and plan a negative output length. Reject the header instead of
	// trusting it.
	for _, d := range info.Shape {
		if d < 0 {
			return plan, fmt.Errorf("tensor %q: negative dimension %d in shape %v", name, d, info.Shape)
		}
	}

	convertible := info.DType == DTypeF16 || info.DType == DTypeBF16 ||
		info.DType == DTypeF32 || info.DType == DTypeF64

	target := opts.Default
	explicit := false
	if opts.Config != nil {
		target, explicit = opts.Config.TargetFor(name, opts.Default)
	}

	// The default protection policy (protect.go) keeps precision-sensitive
	// tensors at their original dtype. It is a default, not a lock: an
	// explicit config rule (matched above) is per-tensor user intent and
	// wins, while the config default and the CLI -target are bulk defaults
	// the policy applies to. Protection is decided before the mechanical
	// passthrough checks below (MinElems, the convrot 256-multiple rule):
	// all three produce passthrough, so the only consequence of the order
	// is which reason string is reported - a policy decision takes
	// precedence over a mechanical constraint.
	if opts.Protect && !explicit {
		if protect, reason := ProtectDefault(name, target); protect {
			target = TargetNone
			plan.protectReason = reason
		}
	}

	switch {
	case !convertible:
		plan.outDType = info.DType
		plan.skippedWhy = "non-float dtype, copied as-is"
	case target == TargetNone:
		plan.outDType = info.DType
		if plan.protectReason != "" {
			plan.skippedWhy = plan.protectReason
		} else {
			plan.skippedWhy = "config/default says keep original"
		}
	case opts.MinElems > 0 && numElems < int64(opts.MinElems):
		plan.outDType = info.DType
		plan.skippedWhy = fmt.Sprintf("fewer than %d elements", opts.MinElems)
	case target == TargetInt8ConvRot && numElems%256 != 0:
		// Checked after the min-elems case, so -min-elems still wins.
		plan.outDType = info.DType
		plan.skippedWhy = "not a multiple of 256 (rotation group size)"
	default:
		plan.target = target
		// A converting tensor is read with numElems × srcElemSize bytes at
		// srcAbsOffset, so the header's data_offsets must agree with
		// shape × dtype. A mismatch would either read a neighboring
		// tensor's bytes silently or fail later with an opaque ReadAt error
		// after the outputs were already created. The check is
		// converting-path-only: the passthrough paths deliberately skip it
		// - see the srcLen-copy comment below for why.
		srcElemSize, err := info.DType.ByteSize()
		if err != nil {
			return plan, err
		}
		wantLen := numElems * int64(srcElemSize)
		if plan.srcLen != wantLen {
			return plan, fmt.Errorf("tensor %q: data_offsets span %d bytes, but shape %v of %s is %d elems x %d bytes = %d bytes",
				name, plan.srcLen, info.Shape, info.DType, numElems, srcElemSize, wantLen)
		}
		// An explicit rule converting a protected tensor wins over the
		// policy (quantization-advice.md: honor the request, but warn).
		// Only tensors actually converted are flagged - a rule that keeps
		// a protected name at its original dtype overrides nothing.
		if opts.Protect && explicit {
			plan.protectOverride, _ = ProtectDefault(name, target)
		}
		switch target {
		case TargetFP8E4M3:
			plan.outDType = DTypeF8E4M3
			plan.outLen = numElems // 1 byte/elem
		case TargetFP8E5M2:
			plan.outDType = DTypeF8E5M2
			plan.outLen = numElems // 1 byte/elem
		case TargetInt8:
			plan.outDType = DTypeI8
			plan.outLen = numElems // 1 byte/elem
		case TargetInt8ConvRot:
			plan.outDType = DTypeI8
			plan.outLen = numElems // 1 byte/elem
		case TargetInt4, TargetNVFP4:
			plan.outDType = DTypeU8
			plan.outLen = (numElems + 1) / 2 // 2 elements packed per byte
			plan.outShape = packedOwnerShape(info.Shape, numElems)
		case TargetMxFP4:
			// Unpacked, unlike int4/nvfp4: one E2M1 code per U8 byte, so
			// the owner's header keeps the SOURCE shape. Standard loaders
			// (transformers' from_pretrained) match each checkpoint tensor
			// against the model parameter's shape; a halved packed shape is
			// a size mismatch that aborts the load. The dequantizing
			// consumer applies the E2M1 LUT and the ".block_scale" sibling
			// per 32-element block: value = e2m1(code) * 2^(e8m0(scale)-127).
			plan.outDType = DTypeU8
			plan.outLen = numElems // 1 byte/elem (unpacked code)
		default:
			return plan, fmt.Errorf("tensor %q: unhandled target kind %v", name, target)
		}
		plan.sibs = planSiblings(target, info.Shape, numElems)
	}

	// Passthrough plans copy exactly the source's stored bytes (srcLen),
	// the same length copyRaw writes. For ordinary tensors srcLen ==
	// dtype-size × numElems, and this tool's own packed 4-bit outputs
	// satisfy the same agreement now that their header carries the
	// packed shape (see packedOwnerShape): srcLen == numElems ×
	// itemsize, so a -target none round-trip of such an output stays
	// byte-identical.
	//
	// That is exactly why the shape/offset byte-agreement check above
	// (srcLen == numElems × srcElemSize) is converting-path-only and is
	// NOT applied here: a passthrough must not be rejected for third-
	// party files written with a lenient packed convention, where the
	// header keeps the full semantic shape while only half the bytes are
	// stored, so srcLen is deliberately half of numElems × srcElemSize.
	// Such a tensor's dtype is U8, which is non-convertible anyway, so it
	// always lands on this path.
	if plan.skippedWhy != "" {
		plan.outLen = plan.srcLen
	}

	// Emit in file order: before-sibs, then the owner, then after-sibs.
	for _, s := range plan.sibs {
		if s.before {
			appendHeaderEntry(outHeader, name+s.suffix, s.dtype, sibShape(s), s.byteLen())
		}
	}
	appendHeaderEntry(outHeader, name, plan.outDType, plan.outShape, plan.outLen)
	for _, s := range plan.sibs {
		if !s.before {
			appendHeaderEntry(outHeader, name+s.suffix, s.dtype, sibShape(s), s.byteLen())
		}
	}

	return plan, nil
}

// countingWriter wraps an io.Writer and counts the bytes written through
// it. convertTensor uses one per tensor (O(1) state) to enforce the
// plan/stream invariant in checkWritten: the plan-then-stream design has
// exactly one failure mode it cannot tolerate - the bytes written
// disagreeing with the bytes planned - and without this, such a
// disagreement silently corrupts the output.
type countingWriter struct {
	w io.Writer
	n int64
}

func (cw *countingWriter) Write(p []byte) (int, error) {
	n, err := cw.w.Write(p)
	cw.n += int64(n)
	return n, err
}

// progressAdder reports each successful write to the progress reporter,
// so a pass without its own chunk loop (the passthrough's io.Copy) still
// reports one Add per copied chunk, like the converting passes.
type progressAdder struct {
	w    io.Writer
	prog *Progress
}

func (p *progressAdder) Write(b []byte) (int, error) {
	n, err := p.w.Write(b)
	if n > 0 {
		p.prog.Add(int64(n))
	}
	return n, err
}

// checkWritten is convertTensor's plan/stream guard, run before it
// returns success: the bytes actually written for p must equal
// p.totalOutLen() (owner plus all sibling tensors). A mismatch means the
// streaming pass and the plan disagree, so the output header's offsets no
// longer describe the file that was written - silent corruption turned
// into a hard error.
func checkWritten(p tensorPlan, cw *countingWriter) error {
	if cw.n != p.totalOutLen() {
		return fmt.Errorf("tensor %q: wrote %d bytes, planned %d", p.name, cw.n, p.totalOutLen())
	}
	return nil
}

// maxElementSize is the largest element size among the convertible
// source dtypes (F16/BF16/F32/F64). The per-run pass scratch is sized
// before any tensor is read, and the source dtype varies per tensor, so
// its raw buffers must hold the worst case.
const maxElementSize = 8

// passScratch is the working buffers of the streaming conversion passes,
// allocated once per run (in ConvertModel / convertModelToShards, before
// the tensor loop) and shared by every pass of every tensor. It exists
// because the passes used to make() their own chunk-sized buffers per
// tensor: on a real run (~376 tensors x several MB) that is multi-GB of
// transient allocation churn for buffers that are always the same size.
//
// The scratch is reused across tensors because tensors are processed
// strictly one at a time: the plan loop runs each tensor's passes to
// completion before starting the next tensor, and within a tensor the
// passes run sequentially (scan, then quantize, ...), so no two passes
// are ever in flight at once. The one-shot wrappers (toFloat32Slice,
// packNibbles) and the tests keep allocating their own buffers: they are
// not on the per-tensor hot path.
//
// Every size depends only on chunkElems (and the fixed format
// constants), never on the tensor or the file: the total is
// passScratchSize(chunkElems), O(chunk) x a small constant (memory
// rule). The raw buffers are sized for maxElementSize (8 bytes, F64) for
// the same reason as above.
type passScratch struct {
	// raw1/raw2 are the double-buffered raw input: the current chunk (or
	// convrot window) and the prefetch target of the next one (the S10
	// double-buffer pattern, +1 raw buffer over the serial path). Sized
	// for the max window at maxElementSize, so every pass's raw read
	// fits.
	raw1, raw2 []byte
	// fbuf is the f32 decode scratch of the per-element passes (scan,
	// fp8, int8, int4, mxfp4, nvfp4), sized for the max window.
	fbuf []float32
	// out is the fp8/int8 encoded output (1 byte per element), max window.
	out []byte
	// packedInt4 is the int4 packed output (2 elements per byte), max
	// window.
	packedInt4 []byte
	// mxfp4Scales / mxfp4Codes are the mxfp4 block passes' per-chunk
	// scale bytes (one E8M0 per 32-element block) and unpacked E2M1
	// codes (one byte per element - see the TargetMxFP4 case in
	// planTensor for why mxfp4 is not packed 2-per-byte).
	mxfp4Scales []byte
	mxfp4Codes  []byte
	// nvfp4Scales / nvfp4Packed / nvfp4Qs are the nvfp4 counterparts
	// (one E4M3 scale per 16-element block).
	nvfp4Scales []byte
	nvfp4Packed []byte
	nvfp4Qs     [][]uint8
	// convrotGroups holds one 256-element f32 buffer per rotation group
	// in the max window (the rotated values), the convrot pass' group
	// buffers.
	convrotGroups [][convrotGroup]float32
	// results is the scan pass' per-worker max: one float32 per fan-out
	// worker at the max window's parallelism (bounded by the CPU count).
	results []float32
}

// maxWindowElems is the largest element window any streaming pass reads
// for a given chunkElems: the convrot window rounding (up to the next
// multiple of 256, at least one group). A multiple of 256 is a multiple
// of the mxfp4 (32) and nvfp4 (16) block sizes and of 2 (int4's even
// rounding), and >= chunkElems itself, so it covers every pass's
// effective chunk for the same chunkElems.
func maxWindowElems(chunkElems int) int64 {
	return int64((chunkElems + convrotGroup - 1) / convrotGroup * convrotGroup)
}

// passScratchSize is the total byte size of a passScratch for a given
// chunkElems - the documented bound, O(chunk) x a small constant. With
// w = maxWindowElems(chunkElems):
//
//	2 * w * 8                          raw1 + raw2 (max element size 8)
//	+ w * 4                            fbuf (f32 decode)
//	+ w                                out (1 byte/elem)
//	+ (w+1)/2                          packedInt4 (2 elems/byte)
//	+ w/32 + w                         mxfp4Scales + mxfp4Codes (1 byte/elem)
//	+ w/16 + (w+1)/2                   nvfp4Scales + nvfp4Packed
//	+ max(1, parallelism(w/16)) * 16   nvfp4Qs (per worker, nvfp4Block)
//	+ w * 4                            convrotGroups (one f32/elem)
//	+ max(1, parallelism(w)) * 4       results (per worker, one f32)
//
// The parallelism terms are bounded by the CPU count, never the chunk or
// the tensor. chunkElems is the run's normalized chunk size
// (ConvertModel substitutes DefaultChunkElems for <= 0).
func passScratchSize(chunkElems int) int64 {
	w := maxWindowElems(chunkElems)
	return 2*w*maxElementSize +
		w*4 +
		w +
		(w+1)/2 +
		w/mxfp4Block + w +
		w/nvfp4Block + (w+1)/2 +
		int64(max(1, parallelism(int(w/nvfp4Block))))*nvfp4Block +
		w*4 +
		int64(max(1, parallelism(int(w))))*4
}

// newPassScratch allocates a passScratch for one run: every buffer at
// exactly the size passScratchSize accounts for (see the formula there),
// so the run's pass-buffer memory is the documented bound.
func newPassScratch(chunkElems int) *passScratch {
	w := maxWindowElems(chunkElems)
	sc := &passScratch{
		raw1:          make([]byte, w*maxElementSize),
		raw2:          make([]byte, w*maxElementSize),
		fbuf:          make([]float32, w),
		out:           make([]byte, w),
		packedInt4:    make([]byte, (w+1)/2),
		mxfp4Scales:   make([]byte, w/mxfp4Block),
		mxfp4Codes:    make([]byte, w),
		nvfp4Scales:   make([]byte, w/nvfp4Block),
		nvfp4Packed:   make([]byte, (w+1)/2),
		convrotGroups: make([][convrotGroup]float32, w/convrotGroup),
		results:       make([]float32, max(1, parallelism(int(w)))),
	}
	sc.nvfp4Qs = make([][]uint8, max(1, parallelism(int(w/nvfp4Block))))
	for i := range sc.nvfp4Qs {
		sc.nvfp4Qs[i] = make([]uint8, nvfp4Block)
	}
	return sc
}

// convertTensor streams one planned tensor from src - its source shard
// file, already open by the caller - to w, reading from the plan's
// shard-relative offset. Every byte of the tensor's output (owner plus
// siblings) goes through a countingWriter so checkWritten can verify the
// written count against the plan; on any plan/stream disagreement it
// fails the run instead of emitting a corrupt file. sc is the run's
// pass scratch (see passScratch): the passes use its buffers instead of
// allocating their own per tensor. prog is the run's progress reporter
// (nil = no progress output): convertTensor brackets the tensor with
// Begin/End, and the passes report one Add per chunk.
func convertTensor(p tensorPlan, src io.ReaderAt, w io.Writer, chunkElems int, sc *passScratch, prog *Progress) (TensorStat, error) {
	cw := &countingWriter{w: w}
	stat := TensorStat{Name: p.name, FromDType: p.srcInfo.DType, ToDType: p.outDType, NumElems: p.numElems, ProtectOverride: p.protectOverride}

	// Start the tensor's progress line before any of its bytes move, so
	// the line is live for the whole tensor. totalWork is the tensor's
	// total bytes of work (see tensorTotalWork) - a display-only
	// accounting that never affects the output bytes.
	if prog != nil {
		prog.Begin(p.name, p.srcInfo.DType, p.outDType, tensorTotalWork(p))
	}
	// End is called only on success: a failed tensor leaves its line
	// unfinished - the next tensor's Begin finalizes it (Progress's
	// defensive path), or the run ends and the line simply stops
	// updating. success is set only on the final return, so every error
	// path skips it.
	success := false
	defer func() {
		if success && prog != nil {
			prog.End()
		}
	}()

	if p.skippedWhy != "" {
		stat.SkippedWhy = p.skippedWhy
		if err := copyRaw(src, cw, p.srcAbsOffset, p.srcLen, prog); err != nil {
			return stat, fmt.Errorf("copying tensor %q: %w", p.name, err)
		}
		if err := checkWritten(p, cw); err != nil {
			return stat, err
		}
		success = true
		return stat, nil
	}

	switch p.target {
	case TargetFP8E4M3, TargetFP8E5M2:
		e4m3 := p.target == TargetFP8E4M3
		if err := streamConvertFP8(src, cw, p.srcAbsOffset, p.srcInfo.DType, p.numElems, chunkElems, e4m3, sc, prog); err != nil {
			return stat, fmt.Errorf("converting tensor %q: %w", p.name, err)
		}

	case TargetInt8:
		maxAbs, err := streamComputeMaxAbsScale(src, p.srcAbsOffset, p.srcInfo.DType, p.numElems, chunkElems, sc, prog)
		if err != nil {
			return stat, fmt.Errorf("scanning tensor %q: %w", p.name, err)
		}
		scale := int8Scale(maxAbs)
		if err := streamConvertInt8(src, cw, p.srcAbsOffset, p.srcInfo.DType, p.numElems, chunkElems, scale, sc, prog); err != nil {
			return stat, fmt.Errorf("converting tensor %q: %w", p.name, err)
		}
		stat.Scale = scale
		if err := writeF32Scale(cw, p.name, scale); err != nil {
			return stat, err
		}

	case TargetInt4:
		maxAbs, err := streamComputeMaxAbsScale(src, p.srcAbsOffset, p.srcInfo.DType, p.numElems, chunkElems, sc, prog)
		if err != nil {
			return stat, fmt.Errorf("scanning tensor %q: %w", p.name, err)
		}
		scale := int4Scale(maxAbs)
		if err := streamInt4Data(src, cw, p.srcAbsOffset, p.srcInfo.DType, p.numElems, scale, chunkElems, sc, prog); err != nil {
			return stat, fmt.Errorf("converting tensor %q: %w", p.name, err)
		}
		stat.Scale = scale
		if err := writeF32Scale(cw, p.name, scale); err != nil {
			return stat, err
		}

	case TargetInt8ConvRot:
		// The row-scale sibling precedes the owner in the file, so the
		// pass writes scales first, then the rotated I8 data, to cw.
		rows, err := streamConvRot(src, cw, p.srcAbsOffset, p.srcInfo.DType, p.srcInfo.Shape, p.numElems, chunkElems, sc, prog)
		if err != nil {
			return stat, fmt.Errorf("converting tensor %q: %w", p.name, err)
		}
		stat.Note = fmt.Sprintf("convrot rows=%d", rows)

	case TargetMxFP4:
		// The block-scale sibling precedes the owner in the file, so the
		// pass writes the E8M0 block scales first, then the packed E2M1
		// data, to cw.
		if err := streamMxFP4(src, cw, p.srcAbsOffset, p.srcInfo.DType, p.numElems, chunkElems, sc, prog); err != nil {
			return stat, fmt.Errorf("converting tensor %q: %w", p.name, err)
		}

	case TargetNVFP4:
		// The ".global_scale" and ".block_scale" siblings both precede
		// the owner in the file, so the pass writes the F32 global scale,
		// the E4M3 block scales, then the packed E2M1 data, to cw.
		alpha, err := streamNVFP4(src, cw, p.srcAbsOffset, p.srcInfo.DType, p.numElems, chunkElems, sc, prog)
		if err != nil {
			return stat, fmt.Errorf("converting tensor %q: %w", p.name, err)
		}
		stat.Note = fmt.Sprintf("nvfp4 scale=%g", alpha)

	default:
		return stat, fmt.Errorf("tensor %q: unhandled target kind %v", p.name, p.target)
	}

	if err := checkWritten(p, cw); err != nil {
		return stat, err
	}
	success = true
	return stat, nil
}

// tensorTotalWork is the tensor's total bytes of work for the progress
// line: the source bytes (srcLen) times the target's pass count. The
// multiplier is a display choice - it scales the percentage, rate, and
// eta only, and never the output bytes:
//
//	passthrough / none: srcLen x 1 (one copy)
//	fp8 e4m3 / e5m2:    srcLen x 1 (one pass)
//	int8, int4:         srcLen x 2 (max-abs scan + quantize)
//	int8_convrot:       srcLen x 3 (scale scan + row rescan + quantize;
//	                   each sweep walks every source byte, so this is a
//	                   conservative lower bound - a multi-window row's
//	                   quantize re-reads and re-rotates its head window,
//	                   which is not counted)
//	mxfp4:              srcLen x 2 (block-scale pass + data pass)
//	nvfp4:              srcLen x 3 (global-max scan + block-scale pass +
//	                   data pass)
func tensorTotalWork(p tensorPlan) int64 {
	if p.skippedWhy != "" {
		return p.srcLen
	}
	var mult int64
	switch p.target {
	case TargetInt8, TargetInt4, TargetMxFP4:
		mult = 2
	case TargetInt8ConvRot, TargetNVFP4:
		mult = 3
	default: // TargetFP8E4M3, TargetFP8E5M2
		mult = 1
	}
	return p.srcLen * mult
}

// appendHeaderEntry records one tensor's header entry given its already-
// known output byte length, advancing the running offset.
func appendHeaderEntry(h *Header, name string, dtype DType, shape []int64, byteLen int64) {
	begin := int64(0)
	if n := len(h.Tensors); n > 0 {
		begin = h.Tensors[n-1].Info.DataOffsets[1]
	}
	h.Tensors = append(h.Tensors, TensorEntry{
		Name: name,
		Info: TensorInfo{
			DType:       dtype,
			Shape:       shape,
			DataOffsets: [2]int64{begin, begin + byteLen},
		},
	})
}

// copyRaw streams length bytes starting at offset from r straight to w
// without materializing the whole region in memory (io.Copy uses an
// internal fixed-size buffer, not the section length).
func copyRaw(r io.ReaderAt, w io.Writer, offset, length int64, prog *Progress) error {
	if prog != nil {
		w = &progressAdder{w: w, prog: prog}
	}
	sr := io.NewSectionReader(r, offset, length)
	_, err := io.Copy(w, sr)
	return err
}

// streamComputeMaxAbsScale performs a chunked read-only pass over a tensor
// to find the raw max(abs(x)) - the value every symmetric quantization
// scale (int8Scale, int4Scale) is derived from - without holding the whole
// tensor in memory. NaN/Inf source values are ignored for the max (they'll
// be clamped/handled at quantization time).
//
// The per-chunk max loop is fanned out across cores (mapContiguous): each
// worker scans a disjoint [lo, hi) sub-range of the decoded chunk into a
// worker-local max, and the per-worker results are reduced after the join.
// The decode itself stays serial: it is the only step that reads the
// shared raw chunk, and the encode passes parallelize the same element
// partition over their own loops.
//
// The chunk loop double-buffers the raw input: the first chunk is read
// synchronously, and each later chunk is prefetched (in a goroutine) into
// a second chunk-sized buffer while the previous chunk is being computed,
// so the next chunk's read latency overlaps this chunk's CPU work. The
// prefetch changes only timing - the bytes read and the result are
// identical to the serial loop.
func streamComputeMaxAbsScale(r io.ReaderAt, offset int64, srcDType DType, numElems int64, chunkElems int, sc *passScratch, prog *Progress) (float32, error) {
	elemSize, err := srcDType.ByteSize()
	if err != nil {
		return 0, err
	}
	// The per-pass buffers come from the run's passScratch (see
	// passScratch): allocated once per run, sized to chunkElems, and
	// reused across tensors - no per-tensor or per-chunk allocation.
	buf := sc.raw1[:chunkElems*elemSize]
	// buf2 is the prefetch (double-buffer) counterpart of buf: while chunk
	// i is being scanned, the next chunk i+1 is read into buf2 in a
	// goroutine. +1 raw chunk buffer over the serial path, bounded by the
	// chunk (memory rule): both are chunk-sized, never tensor-sized.
	buf2 := sc.raw2[:chunkElems*elemSize]
	// One chunk-sized f32 scratch reused across all chunks: the decode
	// fills at most chunkElems slots, so peak memory stays O(chunk) and
	// no per-chunk allocation churn is left for the GC.
	fbuf := sc.fbuf[:chunkElems]
	// One float32 per fan-out worker for the worker-local max, reduced
	// after the join. Sized parallelism(chunkElems): bounded by the CPU
	// count, never by the chunk or tensor size (memory rule); parallelism
	// is monotone in its argument, so the scratch's slice (sized for the
	// max window) also covers every smaller final chunk.
	results := sc.results[:max(1, parallelism(chunkElems))]

	// The first chunk has no predecessor to overlap with, so it is read
	// synchronously; every later chunk arrives via the previous
	// iteration's prefetch (see below), already in buf. A zero-element
	// tensor issues no read at all (the loop below never runs), matching
	// the serial path.
	if numElems > 0 {
		firstN := int64(chunkElems)
		if numElems < firstN {
			firstN = numElems
		}
		if _, err := r.ReadAt(buf[:firstN*int64(elemSize)], offset); err != nil {
			return 0, err
		}
	}

	var maxAbs float32
	var done int64
	for done < numElems {
		n := int64(chunkElems)
		if numElems-done < n {
			n = numElems - done
		}
		chunkBuf := buf[:n*int64(elemSize)]
		// Prefetch the next chunk into buf2 before this chunk's scan, so
		// its read latency overlaps this chunk's CPU work. The last chunk
		// has no successor, so no prefetch is started for it.
		var nextErr chan error
		if done+n < numElems {
			nextN := int64(chunkElems)
			if rem := numElems - done - n; rem < nextN {
				nextN = rem
			}
			nextOff := offset + (done+n)*int64(elemSize)
			nextBuf := buf2
			nextErr = make(chan error, 1)
			go func() {
				_, e := r.ReadAt(nextBuf[:nextN*int64(elemSize)], nextOff)
				nextErr <- e
			}()
		}
		floats, err := toFloat32SliceInto(fbuf, chunkBuf, srcDType)
		if err != nil {
			return 0, err
		}
		// Fan the max out over the chunk's disjoint fbuf sub-ranges: each
		// worker touches only fbuf[lo:hi] and results[slot], so the shared
		// buffers stay race-free. All ranges are sub-slices of the existing
		// per-tensor buffers - no new allocations.
		mapContiguous(int(n), func(slot, lo, hi int) {
			var m float32
			for _, f := range floats[lo:hi] {
				if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
					continue
				}
				a := f
				if a < 0 {
					a = -a
				}
				if a > m {
					m = a
				}
			}
			results[slot] = m
		})
		for i := 0; i < parallelism(int(n)); i++ {
			if results[i] > maxAbs {
				maxAbs = results[i]
			}
		}
		// Join the prefetch before the next scan: a failed read fails the
		// run here (at this chunk boundary, one chunk later than the
		// serial path, with the same ReadAt error). Then swap the buffers
		// so the prefetched chunk becomes the current one.
		if nextErr != nil {
			if err := <-nextErr; err != nil {
				return 0, err
			}
			buf, buf2 = buf2, buf
		}
		done += n
		// Progress: this chunk's source bytes are scanned (the pass writes
		// nothing, so the scan is the chunk's work).
		if prog != nil {
			prog.Add(n * int64(elemSize))
		}
	}
	return maxAbs, nil
}

// writeF32Scale writes the 4-byte little-endian F32 payload of a ".scale"
// sibling (int8 and int4) after the owner's data on w. The 4-byte payload
// is a stack array, so the call allocates nothing.
func writeF32Scale(w io.Writer, name string, scale float32) error {
	var scaleBytes [4]byte
	binary.LittleEndian.PutUint32(scaleBytes[:], math.Float32bits(scale))
	if _, err := w.Write(scaleBytes[:]); err != nil {
		return fmt.Errorf("writing scale for tensor %q: %w", name, err)
	}
	return nil
}

// streamConvertFP8 reads a tensor in bounded chunks, converts each element
// to fp8, and writes the result to w - never holding more than chunkElems
// elements in memory regardless of the tensor's total size. The per-chunk
// encode loop is fanned out across cores (mapContiguous) over disjoint
// output sub-ranges; the write stays one per chunk. The chunk loop
// double-buffers the raw input (see streamComputeMaxAbsScale): the first
// chunk is read synchronously, each later chunk is prefetched into a
// second chunk-sized buffer while the previous chunk is encoded, so the
// next chunk's read latency overlaps this chunk's CPU work.
func streamConvertFP8(r io.ReaderAt, w io.Writer, offset int64, srcDType DType, numElems int64, chunkElems int, e4m3 bool, sc *passScratch, prog *Progress) error {
	elemSize, err := srcDType.ByteSize()
	if err != nil {
		return err
	}
	// The per-pass buffers come from the run's passScratch (see
	// passScratch): allocated once per run, sized to chunkElems, and
	// reused across tensors - no per-tensor or per-chunk allocation.
	inBuf := sc.raw1[:chunkElems*elemSize]
	// inBuf2 is the prefetch (double-buffer) counterpart of inBuf: while
	// chunk i is being encoded, the next chunk i+1 is read into inBuf2 in
	// a goroutine. +1 raw chunk buffer over the serial path, bounded by
	// the chunk (memory rule): both are chunk-sized, never tensor-sized.
	inBuf2 := sc.raw2[:chunkElems*elemSize]
	outBuf := sc.out[:chunkElems]
	// One chunk-sized f32 scratch reused across all chunks (see
	// streamComputeMaxAbsScale): O(chunk) memory, no per-chunk allocations.
	fbuf := sc.fbuf[:chunkElems]

	// The first chunk has no predecessor to overlap with, so it is read
	// synchronously; every later chunk arrives via the previous
	// iteration's prefetch (see below), already in inBuf. A zero-element
	// tensor issues no read at all (the loop below never runs), matching
	// the serial path.
	if numElems > 0 {
		firstN := int64(chunkElems)
		if numElems < firstN {
			firstN = numElems
		}
		if _, err := r.ReadAt(inBuf[:firstN*int64(elemSize)], offset); err != nil {
			return err
		}
	}

	var done int64
	for done < numElems {
		n := int64(chunkElems)
		if numElems-done < n {
			n = numElems - done
		}
		chunkIn := inBuf[:n*int64(elemSize)]
		// Prefetch the next chunk into inBuf2 before this chunk's encode,
		// so its read latency overlaps this chunk's CPU work. The last
		// chunk has no successor, so no prefetch is started for it.
		var nextErr chan error
		if done+n < numElems {
			nextN := int64(chunkElems)
			if rem := numElems - done - n; rem < nextN {
				nextN = rem
			}
			nextOff := offset + (done+n)*int64(elemSize)
			nextBuf := inBuf2
			nextErr = make(chan error, 1)
			go func() {
				_, e := r.ReadAt(nextBuf[:nextN*int64(elemSize)], nextOff)
				nextErr <- e
			}()
		}
		floats, err := toFloat32SliceInto(fbuf, chunkIn, srcDType)
		if err != nil {
			return err
		}
		chunkOut := outBuf[:n]
		// Fan the encode out over the chunk's disjoint outBuf sub-ranges:
		// each worker touches only chunkOut[lo:hi]. All ranges are
		// sub-slices of the existing per-tensor buffers - no new
		// allocations.
		mapContiguous(int(n), func(_, lo, hi int) {
			for i := lo; i < hi; i++ {
				if e4m3 {
					chunkOut[i] = f32ToF8E4M3(floats[i])
				} else {
					chunkOut[i] = f32ToF8E5M2(floats[i])
				}
			}
		})
		if _, err := w.Write(chunkOut); err != nil {
			return err
		}
		// Progress: this chunk's source bytes are converted and handed to
		// the writer.
		if prog != nil {
			prog.Add(n * int64(elemSize))
		}
		// Join the prefetch before the next encode: a failed read fails
		// the run here (at this chunk boundary, one chunk later than the
		// serial path, with the same ReadAt error). Then swap the buffers
		// so the prefetched chunk becomes the current one.
		if nextErr != nil {
			if err := <-nextErr; err != nil {
				return err
			}
			inBuf, inBuf2 = inBuf2, inBuf
		}
		done += n
	}
	return nil
}

// streamConvertInt8 is streamConvertFP8's counterpart for int8, applying a
// precomputed per-tensor scale to each chunk (and fanning the per-chunk
// encode out the same way - see streamConvertFP8). It shares the
// double-buffered chunk loop of streamConvertFP8: the first chunk is read
// synchronously, each later chunk is prefetched into a second chunk-sized
// buffer while the previous chunk is encoded, so the next chunk's read
// latency overlaps this chunk's CPU work.
func streamConvertInt8(r io.ReaderAt, w io.Writer, offset int64, srcDType DType, numElems int64, chunkElems int, scale float32, sc *passScratch, prog *Progress) error {
	elemSize, err := srcDType.ByteSize()
	if err != nil {
		return err
	}
	// The per-pass buffers come from the run's passScratch (see
	// passScratch): allocated once per run, sized to chunkElems, and
	// reused across tensors - no per-tensor or per-chunk allocation.
	inBuf := sc.raw1[:chunkElems*elemSize]
	// inBuf2 is the prefetch (double-buffer) counterpart of inBuf: while
	// chunk i is being encoded, the next chunk i+1 is read into inBuf2 in
	// a goroutine. +1 raw chunk buffer over the serial path, bounded by
	// the chunk (memory rule): both are chunk-sized, never tensor-sized.
	inBuf2 := sc.raw2[:chunkElems*elemSize]
	outBuf := sc.out[:chunkElems]
	// One chunk-sized f32 scratch reused across all chunks (see
	// streamComputeMaxAbsScale): O(chunk) memory, no per-chunk allocations.
	fbuf := sc.fbuf[:chunkElems]

	// The first chunk has no predecessor to overlap with, so it is read
	// synchronously; every later chunk arrives via the previous
	// iteration's prefetch (see below), already in inBuf. A zero-element
	// tensor issues no read at all (the loop below never runs), matching
	// the serial path.
	if numElems > 0 {
		firstN := int64(chunkElems)
		if numElems < firstN {
			firstN = numElems
		}
		if _, err := r.ReadAt(inBuf[:firstN*int64(elemSize)], offset); err != nil {
			return err
		}
	}

	var done int64
	for done < numElems {
		n := int64(chunkElems)
		if numElems-done < n {
			n = numElems - done
		}
		chunkIn := inBuf[:n*int64(elemSize)]
		// Prefetch the next chunk into inBuf2 before this chunk's encode,
		// so its read latency overlaps this chunk's CPU work. The last
		// chunk has no successor, so no prefetch is started for it.
		var nextErr chan error
		if done+n < numElems {
			nextN := int64(chunkElems)
			if rem := numElems - done - n; rem < nextN {
				nextN = rem
			}
			nextOff := offset + (done+n)*int64(elemSize)
			nextBuf := inBuf2
			nextErr = make(chan error, 1)
			go func() {
				_, e := r.ReadAt(nextBuf[:nextN*int64(elemSize)], nextOff)
				nextErr <- e
			}()
		}
		floats, err := toFloat32SliceInto(fbuf, chunkIn, srcDType)
		if err != nil {
			return err
		}
		chunkOut := outBuf[:n]
		// Fan the encode out over the chunk's disjoint outBuf sub-ranges
		// (see streamConvertFP8); each worker touches only
		// chunkOut[lo:hi]. All ranges are sub-slices of the existing
		// per-tensor buffers - no new allocations.
		mapContiguous(int(n), func(_, lo, hi int) {
			for i := lo; i < hi; i++ {
				chunkOut[i] = byte(f32ToInt8(floats[i], scale))
			}
		})
		if _, err := w.Write(chunkOut); err != nil {
			return err
		}
		// Progress: this chunk's source bytes are converted and handed to
		// the writer.
		if prog != nil {
			prog.Add(n * int64(elemSize))
		}
		// Join the prefetch before the next encode: a failed read fails
		// the run here (at this chunk boundary, one chunk later than the
		// serial path, with the same ReadAt error). Then swap the buffers
		// so the prefetched chunk becomes the current one.
		if nextErr != nil {
			if err := <-nextErr; err != nil {
				return err
			}
			inBuf, inBuf2 = inBuf2, inBuf
		}
		done += n
	}
	return nil
}

// toFloat32SliceInto decodes raw tensor bytes of the given source dtype
// into the caller-provided scratch dst and returns the filled prefix
// dst[:n]. The streaming passes pass one chunk-sized dst (len = chunkElems)
// and reuse it for every chunk, so the decode allocates nothing per chunk
// and the decoded values stay bounded by the caller's chunk size; the
// caller must provide len(dst) >= the decoded element count.
func toFloat32SliceInto(dst []float32, raw []byte, dtype DType) ([]float32, error) {
	size, err := dtype.ByteSize()
	if err != nil {
		return nil, err
	}
	if size == 0 || len(raw)%size != 0 {
		return nil, fmt.Errorf("raw byte length %d not a multiple of element size %d", len(raw), size)
	}
	n := len(raw) / size
	if len(dst) < n {
		return nil, fmt.Errorf("dst too small: %d float32 slots needed, have %d", n, len(dst))
	}
	switch dtype {
	case DTypeF32, DTypeF64, DTypeF16, DTypeBF16:
	default:
		return nil, fmt.Errorf("unsupported source float dtype %q", dtype)
	}
	return decodeFloat32Into(dst, raw, dtype, n), nil
}

// decodeFloat32Into decodes the first n elements of raw (each of dtype's
// element size) into dst[:n] and returns that prefix. It is the
// error-free core of toFloat32SliceInto: the dtype and the slice sizes
// are preconditions (toFloat32SliceInto validates them), so a caller
// that has checked them - the convrot window rotation, whose parallel
// workers cannot return an error - can call it directly.
func decodeFloat32Into(dst []float32, raw []byte, dtype DType, n int) []float32 {
	out := dst[:n]
	switch dtype {
	case DTypeF32:
		for i := 0; i < n; i++ {
			bits := binary.LittleEndian.Uint32(raw[i*4:])
			out[i] = math.Float32frombits(bits)
		}
	case DTypeF64:
		for i := 0; i < n; i++ {
			bits := binary.LittleEndian.Uint64(raw[i*8:])
			out[i] = float32(math.Float64frombits(bits))
		}
	case DTypeF16:
		for i := 0; i < n; i++ {
			bits := binary.LittleEndian.Uint16(raw[i*2:])
			out[i] = f16ToF32(bits)
		}
	case DTypeBF16:
		for i := 0; i < n; i++ {
			bits := binary.LittleEndian.Uint16(raw[i*2:])
			out[i] = bf16ToF32(bits)
		}
	}
	return out
}

// toFloat32Slice decodes raw tensor bytes of the given source dtype into a
// fresh []float32 for uniform downstream processing. The streaming passes
// use toFloat32SliceInto with a reused chunk-sized scratch instead; this
// wrapper remains for one-shot callers (and the tests).
func toFloat32Slice(raw []byte, dtype DType) ([]float32, error) {
	size, err := dtype.ByteSize()
	if err != nil {
		return nil, err
	}
	if size == 0 || len(raw)%size != 0 {
		return nil, fmt.Errorf("raw byte length %d not a multiple of element size %d", len(raw), size)
	}
	return toFloat32SliceInto(make([]float32, len(raw)/size), raw, dtype)
}
