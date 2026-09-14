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
	MinElems    int        // tensors with fewer elements than this are never converted (0 disables)
	ChunkElems  int        // 0 -> DefaultChunkElems
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
}

// tensorPlan is decided entirely from the input header (dtype + shape),
// with no tensor data read. Because output byte length only depends on
// element count and target dtype, every output offset can be computed
// up front, which is what lets us write the header once and then stream
// data straight through in file order.
type tensorPlan struct {
	name         string
	srcShard     int               // index into ConvertOptions.InputShards; srcAbsOffset is relative to that shard file
	srcName      string            // input shard filename (basename); multi-file output keeps it as the output shard name
	srcMetadata  map[string]string // input shard's __metadata__ block (nil if none); shared by all plans from that shard
	srcInfo      TensorInfo
	srcAbsOffset int64 // absolute offset of source tensor bytes in its source shard file
	srcLen       int64 // source tensor byte length
	numElems     int64
	target       TargetKind
	outDType     DType
	outLen       int64  // planned output byte length of this tensor (sibling tensors not included)
	skippedWhy   string // non-empty => passthrough copy, no conversion
	sibs         []sib  // sibling tensors this plan emits (nil for passthrough)
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
	defer out.Close()

	bw := bufio.NewWriterSize(out, 4<<20) // 4MB write buffer, independent of chunkElems

	// ---- Pass 2: stream each tensor's data straight through. ----
	shards, err := openShards(opts.InputShards)
	if err != nil {
		return nil, err
	}
	defer closeFiles(shards)

	stats := make([]TensorStat, 0, len(plans))
	for _, p := range plans {
		stat, err := convertTensor(p, shards[p.srcShard], bw, chunkElems)
		if err != nil {
			return nil, err
		}
		stat.Source = sources[p.srcShard]
		stats = append(stats, stat)
	}

	if err := bw.Flush(); err != nil {
		return nil, fmt.Errorf("flushing output: %w", err)
	}

	return stats, nil
}

// convertModelToShards is the multi-file output branch of ConvertModel:
// one output shard file per input shard, each with its own header written
// once, up front (the same invariant as single-file output), plus the
// index JSON that ties the shards together. Tensors stream to the output
// shard that mirrors their input shard, in the same global order as the
// single-file pass - nothing here buffers beyond the per-shard write
// buffers, and the open fd count is bounded by the shard count on both
// the input and the output side.
func convertModelToShards(opts ConvertOptions, plans []tensorPlan, sources []string, chunkElems int) ([]TensorStat, error) {
	shardOuts, outIndex, err := PlanShardOutput(plans, opts.InputIndex, opts.OutputDir)
	if err != nil {
		return nil, err
	}

	// Inputs first, so a missing shard fails before any output is created.
	shards, err := openShards(opts.InputShards)
	if err != nil {
		return nil, err
	}
	defer closeFiles(shards)

	// One output file per shard, each with its complete header written
	// once, up front, before any data.
	outFiles := make([]*os.File, len(shardOuts))
	for i, sh := range shardOuts {
		f, err := CreateOutputFile(filepath.Join(opts.OutputDir, sh.Name), sh.Header)
		if err != nil {
			closeFiles(outFiles[:i])
			return nil, fmt.Errorf("creating output shard %q: %w", sh.Name, err)
		}
		outFiles[i] = f
	}
	defer closeFiles(outFiles)

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

	stats := make([]TensorStat, 0, len(plans))
	for _, p := range plans {
		stat, err := convertTensor(p, shards[p.srcShard], bws[outShardOf[p.srcShard]], chunkElems)
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
	plans := make([]tensorPlan, 0, 8)

	for srcShard, path := range opts.InputShards {
		header, dataStart, err := readShardHeader(path)
		if err != nil {
			return nil, nil, err
		}

		if srcShard == 0 {
			// The merged output keeps the first shard's __metadata__ block;
			// merging metadata maps across shards is out of scope.
			outHeader.Metadata = header.Metadata
		}

		base := filepath.Base(path)
		for _, entry := range header.Tensors {
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
	}
	if plan.srcLen < 0 {
		return plan, fmt.Errorf("tensor %q: invalid data_offsets %v", name, info.DataOffsets)
	}

	convertible := info.DType == DTypeF16 || info.DType == DTypeBF16 ||
		info.DType == DTypeF32 || info.DType == DTypeF64

	target := opts.Default
	if opts.Config != nil {
		target = opts.Config.TargetFor(name, opts.Default)
	}

	switch {
	case !convertible:
		plan.outDType = info.DType
		plan.skippedWhy = "non-float dtype, copied as-is"
	case target == TargetNone:
		plan.outDType = info.DType
		plan.skippedWhy = "config/default says keep original"
	case opts.MinElems > 0 && numElems < int64(opts.MinElems):
		plan.outDType = info.DType
		plan.skippedWhy = fmt.Sprintf("fewer than %d elements", opts.MinElems)
	case target == TargetInt8ConvRot && numElems%256 != 0:
		// Checked after the min-elems case, so -min-elems still wins.
		plan.outDType = info.DType
		plan.skippedWhy = "not a multiple of 256 (rotation group size)"
	default:
		plan.target = target
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
		case TargetInt4, TargetMxFP4, TargetNVFP4:
			plan.outDType = DTypeU8
			plan.outLen = (numElems + 1) / 2 // 2 elements packed per byte
		default:
			return plan, fmt.Errorf("tensor %q: unhandled target kind %v", name, target)
		}
		plan.sibs = planSiblings(target, info.Shape, numElems)
	}

	// Passthrough plans copy exactly the source's stored bytes (srcLen),
	// the same length copyRaw writes. For ordinary tensors srcLen ==
	// dtype-size × numElems; for this tool's own packed 4-bit outputs the
	// header keeps the full semantic shape while only half the bytes are
	// stored, so srcLen is what keeps the re-emitted data_offsets
	// consistent with the data actually written (a -target none round-trip
	// of such an output stays byte-identical).
	if plan.skippedWhy != "" {
		plan.outLen = plan.srcLen
	}

	// Emit in file order: before-sibs, then the owner, then after-sibs.
	for _, s := range plan.sibs {
		if s.before {
			appendHeaderEntry(outHeader, name+s.suffix, s.dtype, sibShape(s), s.byteLen())
		}
	}
	appendHeaderEntry(outHeader, name, plan.outDType, info.Shape, plan.outLen)
	for _, s := range plan.sibs {
		if !s.before {
			appendHeaderEntry(outHeader, name+s.suffix, s.dtype, sibShape(s), s.byteLen())
		}
	}

	return plan, nil
}

// convertTensor streams one planned tensor from src - its source shard
// file, already open by the caller - to w, reading from the plan's
// shard-relative offset.
func convertTensor(p tensorPlan, src io.ReaderAt, w io.Writer, chunkElems int) (TensorStat, error) {
	stat := TensorStat{Name: p.name, FromDType: p.srcInfo.DType, ToDType: p.outDType, NumElems: p.numElems}

	if p.skippedWhy != "" {
		stat.SkippedWhy = p.skippedWhy
		if err := copyRaw(src, w, p.srcAbsOffset, p.srcLen); err != nil {
			return stat, fmt.Errorf("copying tensor %q: %w", p.name, err)
		}
		return stat, nil
	}

	switch p.target {
	case TargetFP8E4M3, TargetFP8E5M2:
		e4m3 := p.target == TargetFP8E4M3
		if err := streamConvertFP8(src, w, p.srcAbsOffset, p.srcInfo.DType, p.numElems, chunkElems, e4m3); err != nil {
			return stat, fmt.Errorf("converting tensor %q: %w", p.name, err)
		}

	case TargetInt8:
		maxAbs, err := streamComputeMaxAbsScale(src, p.srcAbsOffset, p.srcInfo.DType, p.numElems, chunkElems)
		if err != nil {
			return stat, fmt.Errorf("scanning tensor %q: %w", p.name, err)
		}
		scale := int8Scale(maxAbs)
		if err := streamConvertInt8(src, w, p.srcAbsOffset, p.srcInfo.DType, p.numElems, chunkElems, scale); err != nil {
			return stat, fmt.Errorf("converting tensor %q: %w", p.name, err)
		}
		stat.Scale = scale
		if err := writeF32Scale(w, p.name, scale); err != nil {
			return stat, err
		}

	case TargetInt4:
		maxAbs, err := streamComputeMaxAbsScale(src, p.srcAbsOffset, p.srcInfo.DType, p.numElems, chunkElems)
		if err != nil {
			return stat, fmt.Errorf("scanning tensor %q: %w", p.name, err)
		}
		scale := int4Scale(maxAbs)
		if err := streamInt4Data(src, w, p.srcAbsOffset, p.srcInfo.DType, p.numElems, scale, chunkElems); err != nil {
			return stat, fmt.Errorf("converting tensor %q: %w", p.name, err)
		}
		stat.Scale = scale
		if err := writeF32Scale(w, p.name, scale); err != nil {
			return stat, err
		}

	case TargetInt8ConvRot:
		// The row-scale sibling precedes the owner in the file, so the
		// pass writes scales first, then the rotated I8 data, to w.
		rows, err := streamConvRot(src, w, p.srcAbsOffset, p.srcInfo.DType, p.srcInfo.Shape, p.numElems, chunkElems)
		if err != nil {
			return stat, fmt.Errorf("converting tensor %q: %w", p.name, err)
		}
		stat.Note = fmt.Sprintf("convrot rows=%d", rows)

	case TargetMxFP4:
		// The block-scale sibling precedes the owner in the file, so the
		// pass writes the E8M0 block scales first, then the packed E2M1
		// data, to w.
		if err := streamMxFP4(src, w, p.srcAbsOffset, p.srcInfo.DType, p.numElems, chunkElems); err != nil {
			return stat, fmt.Errorf("converting tensor %q: %w", p.name, err)
		}

	case TargetNVFP4:
		// The ".global_scale" and ".block_scale" siblings both precede
		// the owner in the file, so the pass writes the F32 global scale,
		// the E4M3 block scales, then the packed E2M1 data, to w.
		alpha, err := streamNVFP4(src, w, p.srcAbsOffset, p.srcInfo.DType, p.numElems, chunkElems)
		if err != nil {
			return stat, fmt.Errorf("converting tensor %q: %w", p.name, err)
		}
		stat.Note = fmt.Sprintf("nvfp4 scale=%g", alpha)

	default:
		return stat, fmt.Errorf("tensor %q: unhandled target kind %v", p.name, p.target)
	}

	return stat, nil
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
func copyRaw(r io.ReaderAt, w io.Writer, offset, length int64) error {
	sr := io.NewSectionReader(r, offset, length)
	_, err := io.Copy(w, sr)
	return err
}

// streamComputeMaxAbsScale performs a chunked read-only pass over a tensor
// to find the raw max(abs(x)) - the value every symmetric quantization
// scale (int8Scale, int4Scale) is derived from - without holding the whole
// tensor in memory. NaN/Inf source values are ignored for the max (they'll
// be clamped/handled at quantization time).
func streamComputeMaxAbsScale(r io.ReaderAt, offset int64, srcDType DType, numElems int64, chunkElems int) (float32, error) {
	elemSize, err := srcDType.ByteSize()
	if err != nil {
		return 0, err
	}
	buf := make([]byte, chunkElems*elemSize)

	var maxAbs float32
	var done int64
	for done < numElems {
		n := int64(chunkElems)
		if numElems-done < n {
			n = numElems - done
		}
		chunkBuf := buf[:n*int64(elemSize)]
		if _, err := r.ReadAt(chunkBuf, offset+done*int64(elemSize)); err != nil {
			return 0, err
		}
		floats, err := toFloat32Slice(chunkBuf, srcDType)
		if err != nil {
			return 0, err
		}
		for _, f := range floats {
			if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
				continue
			}
			a := f
			if a < 0 {
				a = -a
			}
			if a > maxAbs {
				maxAbs = a
			}
		}
		done += n
	}
	return maxAbs, nil
}

// writeF32Scale writes the 4-byte little-endian F32 payload of a ".scale"
// sibling (int8 and int4) after the owner's data on w.
func writeF32Scale(w io.Writer, name string, scale float32) error {
	scaleBytes := make([]byte, 4)
	binary.LittleEndian.PutUint32(scaleBytes, math.Float32bits(scale))
	if _, err := w.Write(scaleBytes); err != nil {
		return fmt.Errorf("writing scale for tensor %q: %w", name, err)
	}
	return nil
}

// streamConvertFP8 reads a tensor in bounded chunks, converts each element
// to fp8, and writes the result to w - never holding more than chunkElems
// elements in memory regardless of the tensor's total size.
func streamConvertFP8(r io.ReaderAt, w io.Writer, offset int64, srcDType DType, numElems int64, chunkElems int, e4m3 bool) error {
	elemSize, err := srcDType.ByteSize()
	if err != nil {
		return err
	}
	inBuf := make([]byte, chunkElems*elemSize)
	outBuf := make([]byte, chunkElems)

	var done int64
	for done < numElems {
		n := int64(chunkElems)
		if numElems-done < n {
			n = numElems - done
		}
		chunkIn := inBuf[:n*int64(elemSize)]
		if _, err := r.ReadAt(chunkIn, offset+done*int64(elemSize)); err != nil {
			return err
		}
		floats, err := toFloat32Slice(chunkIn, srcDType)
		if err != nil {
			return err
		}
		chunkOut := outBuf[:n]
		for i, f := range floats {
			if e4m3 {
				chunkOut[i] = f32ToF8E4M3(f)
			} else {
				chunkOut[i] = f32ToF8E5M2(f)
			}
		}
		if _, err := w.Write(chunkOut); err != nil {
			return err
		}
		done += n
	}
	return nil
}

// streamConvertInt8 is streamConvertFP8's counterpart for int8, applying a
// precomputed per-tensor scale to each chunk.
func streamConvertInt8(r io.ReaderAt, w io.Writer, offset int64, srcDType DType, numElems int64, chunkElems int, scale float32) error {
	elemSize, err := srcDType.ByteSize()
	if err != nil {
		return err
	}
	inBuf := make([]byte, chunkElems*elemSize)
	outBuf := make([]byte, chunkElems)

	var done int64
	for done < numElems {
		n := int64(chunkElems)
		if numElems-done < n {
			n = numElems - done
		}
		chunkIn := inBuf[:n*int64(elemSize)]
		if _, err := r.ReadAt(chunkIn, offset+done*int64(elemSize)); err != nil {
			return err
		}
		floats, err := toFloat32Slice(chunkIn, srcDType)
		if err != nil {
			return err
		}
		chunkOut := outBuf[:n]
		for i, f := range floats {
			chunkOut[i] = byte(f32ToInt8(f, scale))
		}
		if _, err := w.Write(chunkOut); err != nil {
			return err
		}
		done += n
	}
	return nil
}

// toFloat32Slice decodes raw tensor bytes of the given source dtype into a
// []float32 for uniform downstream processing. Used per-chunk, not on
// whole tensors, so its allocation is bounded by the caller's chunk size.
func toFloat32Slice(raw []byte, dtype DType) ([]float32, error) {
	size, err := dtype.ByteSize()
	if err != nil {
		return nil, err
	}
	if size == 0 || len(raw)%size != 0 {
		return nil, fmt.Errorf("raw byte length %d not a multiple of element size %d", len(raw), size)
	}
	n := len(raw) / size
	out := make([]float32, n)

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
	default:
		return nil, fmt.Errorf("unsupported source float dtype %q", dtype)
	}
	return out, nil
}
