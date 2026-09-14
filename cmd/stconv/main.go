// Command safetensors-convert reads a safetensors file or a directory of
// sharded safetensors files (fp16/bf16/fp32 weights) and writes a new
// safetensors file - or a directory of shards plus an index, when -out is
// an existing directory - with tensors converted to fp8 (e4m3 or e5m2) or
// int8, int8_convrot, mxfp4, nvfp4, or int4, using only the Go standard
// library.
//
// Usage:
//
//	safetensors-convert -in model.safetensors -out model.fp8.safetensors -target fp8_e4m3
//	safetensors-convert -in model-dir -out model.merged.fp8.safetensors -target fp8_e4m3
//	safetensors-convert -in model-dir -out out-dir -target fp8_e4m3
//	safetensors-convert -in model.safetensors -out model.int8.safetensors -target int8
//	safetensors-convert -in model.safetensors -out model.convrot.safetensors -target int8_convrot
//	safetensors-convert -in model.safetensors -out model.mxfp4.safetensors -target mxfp4
//	safetensors-convert -in model.safetensors -out model.nvfp4.safetensors -target nvfp4
//	safetensors-convert -in model.safetensors -out model.int4.safetensors -target int4
//	safetensors-convert -in model.safetensors -out model.mixed.safetensors -config config.json
//
// -in may be a single .safetensors file or a model directory of shards
// (see internal/stconv/index.go for discovery); a directory input is
// merged into the single output file, in shard order.
//
// -out is a new .safetensors file path (single-file output, the default)
// or an existing directory (multi-file output: one shard per input shard,
// same filenames, plus a new index - which requires a model-directory
// input). An existing file is refused; the tool never overwrites in place.
//
// See internal/stconv/config.go for the config file format, which lets you
// pick a different target (or "none" to leave it untouched) per tensor
// name or regex pattern.
package main

import (
	"flag"
	"fmt"
	"os"

	"safetensors-convert/internal/stconv"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run() error {
	inPath := flag.String("in", "", "input .safetensors file or model directory (required)")
	outPath := flag.String("out", "", "output .safetensors file, or an existing directory for multi-file output (required)")
	configPath := flag.String("config", "", "optional JSON config for per-layer dtype overrides")
	targetStr := flag.String("target", "fp8_e4m3", "default conversion target for tensors not covered by -config: fp8_e4m3, fp8_e5m2, int8, int8_convrot, mxfp4, nvfp4, int4, or none")
	minElems := flag.Int("min-elems", 0, "skip conversion for tensors with fewer elements than this (e.g. to leave small bias/norm vectors alone)")
	chunkElems := flag.Int("chunk-elems", stconv.DefaultChunkElems, "elements processed per streaming chunk; bounds peak memory regardless of tensor/file size")
	quiet := flag.Bool("quiet", false, "suppress per-tensor report")
	flag.Parse()

	if *inPath == "" || *outPath == "" {
		flag.Usage()
		return fmt.Errorf("-in and -out are required")
	}

	shards, inIndex, inputIsDir, err := stconv.ResolveInput(*inPath)
	if err != nil {
		return err
	}

	target, err := stconv.ParseTargetKind(*targetStr)
	if err != nil {
		return fmt.Errorf("-target: %w", err)
	}

	outMode, outTarget, err := stconv.ResolveOutput(*outPath, inputIsDir)
	if err != nil {
		return err
	}

	var cfg *stconv.Config
	if *configPath != "" {
		cfg, err = stconv.LoadConfig(*configPath)
		if err != nil {
			return err
		}
	}

	opts := stconv.ConvertOptions{
		InputShards: shards,
		InputIndex:  inIndex,
		Config:      cfg,
		Default:     target,
		MinElems:    *minElems,
		ChunkElems:  *chunkElems,
	}
	if outMode == stconv.OutputMulti {
		opts.OutputDir = outTarget
	} else {
		opts.OutputPath = outTarget
	}

	stats, err := stconv.ConvertModel(opts)
	if err != nil {
		return err
	}

	if !*quiet {
		printReport(stats)
	}
	return nil
}

func printReport(stats []stconv.TensorStat) {
	var converted, skipped int
	multiShard := false
	for _, s := range stats {
		if s.SkippedWhy == "" {
			converted++
		} else {
			skipped++
		}
		if s.Source != "" {
			multiShard = true
		}
	}
	fmt.Printf("%d tensors converted, %d left unchanged\n\n", converted, skipped)

	for _, s := range stats {
		if s.SkippedWhy != "" {
			fmt.Printf("  %-50s %-8s (unchanged: %s)%s\n", s.Name, s.FromDType, s.SkippedWhy, srcCol(s, multiShard))
			continue
		}
		note := ""
		if s.Note != "" {
			note = "  " + s.Note
		}
		// Scale is only shown when it was set: int8/int4 scales are never
		// zero (0 -> 1), while convrot's per-row scales live in its ".scale"
		// sibling and leave the per-tensor Scale unset.
		if s.ToDType == stconv.DTypeI8 && s.Scale != 0 {
			fmt.Printf("  %-50s %-8s -> %-8s  scale=%g  n=%d%s%s\n", s.Name, s.FromDType, s.ToDType, s.Scale, s.NumElems, note, srcCol(s, multiShard))
		} else {
			fmt.Printf("  %-50s %-8s -> %-8s  n=%d%s%s\n", s.Name, s.FromDType, s.ToDType, s.NumElems, note, srcCol(s, multiShard))
		}
	}
}

// srcCol renders the source shard column for multi-shard (directory) runs,
// so the report shows where each tensor came from. Single-file runs leave
// it empty, keeping the report byte-identical to the pre-multi-shard form.
func srcCol(s stconv.TensorStat, multiShard bool) string {
	if !multiShard {
		return ""
	}
	return "  from " + s.Source
}
