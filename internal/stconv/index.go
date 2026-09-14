package stconv

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Index is a Hugging Face safetensors index file
// (model.safetensors.index.json): it maps each tensor name to the shard
// file that contains it, plus optional model-level metadata.
//
// Metadata values (e.g. "total_size") are kept raw and are never trusted
// for planning: output sizes are computed from the conversion plan, and
// the input's total_size describes the input, not the output.
type Index struct {
	Path      string            `json:"-"` // path of the index file as loaded by LoadIndex; empty if constructed directly
	Metadata  map[string]any    `json:"metadata"`
	WeightMap map[string]string `json:"weight_map"`

	// weightMapOrder is the key order to use for WeightMap when
	// marshaling (encoding/json's map encoding has no order). PlanShardOutput
	// sets it to the plan's tensor order, so the written index keeps that
	// order on disk; a directly constructed index leaves it empty and falls
	// back to sorted keys.
	weightMapOrder []string
}

// LoadIndex reads and parses a safetensors index file. weight_map is
// required; unknown top-level keys are tolerated so that richer index
// files from other tools don't break us.
func LoadIndex(path string) (*Index, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading index %s: %w", path, err)
	}
	var idx Index
	if err := json.Unmarshal(b, &idx); err != nil {
		return nil, fmt.Errorf("parsing index %s: %w", path, err)
	}
	if len(idx.WeightMap) == 0 {
		return nil, fmt.Errorf("index %s: missing or empty \"weight_map\"", path)
	}
	idx.Path = path
	return &idx, nil
}

// MarshalJSON renders the index in the HF index format, preserving the
// weight_map key order the index carries (encoding/json's map encoding
// would emit the keys in arbitrary order). The manual buffer writing
// follows the same technique as Header.MarshalJSON.
func (i *Index) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer
	buf.WriteByte('{')

	if i.Metadata != nil {
		md, err := json.Marshal(i.Metadata)
		if err != nil {
			return nil, err
		}
		buf.WriteString(`"metadata":`)
		buf.Write(md)
		buf.WriteByte(',')
	}

	buf.WriteString(`"weight_map":{`)
	order := i.weightMapOrder
	if len(order) == 0 {
		order = make([]string, 0, len(i.WeightMap))
		for name := range i.WeightMap {
			order = append(order, name)
		}
		sort.Strings(order)
	}
	for n, name := range order {
		if n > 0 {
			buf.WriteByte(',')
		}
		keyB, err := json.Marshal(name)
		if err != nil {
			return nil, err
		}
		valB, err := json.Marshal(i.WeightMap[name])
		if err != nil {
			return nil, err
		}
		buf.Write(keyB)
		buf.WriteByte(':')
		buf.Write(valB)
	}
	buf.WriteString("}}")
	return buf.Bytes(), nil
}

// ShardFiles returns the deduplicated, lexicographically sorted list of
// shard filenames referenced by WeightMap. Shard filenames are
// zero-padded (model.safetensors-00001-of-00002.safetensors), so
// lexicographic order is numeric order.
func (i *Index) ShardFiles() []string {
	seen := make(map[string]bool, len(i.WeightMap))
	files := make([]string, 0, len(i.WeightMap))
	for _, f := range i.WeightMap {
		if !seen[f] {
			seen[f] = true
			files = append(files, f)
		}
	}
	sort.Strings(files)
	return files
}

// DiscoverShards locates the .safetensors shard files in a model directory.
//
// If dir holds exactly one *.safetensors.index.json, the shards are taken
// from that index's weight_map: every referenced shard must exist in dir
// (error naming the missing file), and any .safetensors file the index does
// not reference is an error (stray shard), so a shard can never be silently
// dropped. Without an index, all *.safetensors files in dir are used in
// lexicographic order, which is also numeric order for zero-padded names.
//
// The returned paths are in processing order; index is the parsed index,
// or nil in the glob fallback.
func DiscoverShards(dir string) (shards []string, index *Index, err error) {
	// os.ReadDir returns entries sorted by filename, which gives the
	// lexicographic glob order for the fallback case.
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, nil, fmt.Errorf("listing shards in %s: %w", dir, err)
	}

	var indexFiles, shardFiles []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		switch name := e.Name(); {
		case strings.HasSuffix(name, ".safetensors.index.json"):
			indexFiles = append(indexFiles, name)
		case strings.HasSuffix(name, ".safetensors"):
			shardFiles = append(shardFiles, name)
		}
	}

	switch len(indexFiles) {
	case 0:
		if len(shardFiles) == 0 {
			return nil, nil, fmt.Errorf("no .safetensors files found in %s", dir)
		}
		for _, name := range shardFiles {
			shards = append(shards, filepath.Join(dir, name))
		}
		return shards, nil, nil
	case 1:
		idx, err := LoadIndex(filepath.Join(dir, indexFiles[0]))
		if err != nil {
			return nil, nil, err
		}
		referenced := make(map[string]bool, len(idx.WeightMap))
		for _, name := range idx.ShardFiles() {
			referenced[name] = true
			if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
				return nil, nil, fmt.Errorf("index %s references missing shard file: %s", indexFiles[0], name)
			}
			shards = append(shards, filepath.Join(dir, name))
		}
		for _, name := range shardFiles {
			if !referenced[name] {
				return nil, nil, fmt.Errorf("stray shard not referenced by index %s: %s", indexFiles[0], name)
			}
		}
		return shards, idx, nil
	default:
		return nil, nil, fmt.Errorf("multiple index files in %s (ambiguous): %s", dir, strings.Join(indexFiles, ", "))
	}
}

// ResolveInput resolves the -in argument into the ordered list of input
// shard files to convert, the model index, and whether the input was a
// directory. The index is nil for single-file input and for model
// directories without an index (the glob fallback).
//
// A regular file is returned as-is, as a one-shard input. The extension is
// deliberately not checked: a file that is not a valid safetensors file is
// rejected when the conversion parses its header, with a more useful error
// than an extension filter would give. A directory is a model directory and
// is resolved with DiscoverShards; the returned shard paths are absolute so
// that later error messages and reports name files unambiguously. Anything
// that is neither a regular file nor a directory (fifo, socket, device)
// is an explicit error. As with os.Stat, symlinks are followed, so a link
// to a file or directory resolves to its target.
func ResolveInput(path string) (shards []string, index *Index, isDir bool, err error) {
	fi, err := os.Stat(path)
	if err != nil {
		return nil, nil, false, fmt.Errorf("resolving input %s: %w", path, err)
	}
	switch {
	case fi.IsDir():
		abs, err := filepath.Abs(path)
		if err != nil {
			return nil, nil, false, fmt.Errorf("resolving input %s: %w", path, err)
		}
		shards, index, err = DiscoverShards(abs)
		if err != nil {
			return nil, nil, false, err
		}
		return shards, index, true, nil
	case fi.Mode().IsRegular():
		return []string{path}, nil, false, nil
	default:
		return nil, nil, false, fmt.Errorf("input %s: not a regular file or directory", path)
	}
}

// Output modes returned by ResolveOutput.
const (
	// OutputSingle: -out is a new file path; single-file output.
	OutputSingle = "single"
	// OutputMulti: -out is an existing directory; multi-file output.
	OutputMulti = "multi"
)

// ResolveOutput decides how the -out argument is interpreted.
//
// An existing directory selects multi-file output (target = that
// directory) and requires a model-directory input: the output replicates
// the input's sharding, so a single-file input has no sharding to
// replicate. A path that does not exist - file-looking or dir-looking
// alike - is a new output file, the default single-file mode. An existing
// regular file is refused: the tool has never overwritten a file in
// place. As with os.Stat, symlinks are followed.
func ResolveOutput(path string, inputIsDir bool) (mode, target string, err error) {
	fi, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return OutputSingle, path, nil
		}
		return "", "", fmt.Errorf("resolving output %s: %w", path, err)
	}
	switch {
	case fi.IsDir():
		if !inputIsDir {
			return "", "", fmt.Errorf("multi-file output requires a model-directory input")
		}
		return OutputMulti, path, nil
	case fi.Mode().IsRegular():
		return "", "", fmt.Errorf("output %s: refusing to overwrite an existing file", path)
	default:
		return "", "", fmt.Errorf("output %s: not a regular file or directory", path)
	}
}
