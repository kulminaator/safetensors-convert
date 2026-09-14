# safetensors-convert

A small, dependency-free Go tool that reads a `.safetensors` file (fp16,
bfloat16, fp32, or fp64 weights) and writes a new `.safetensors` file with
tensors converted to `fp8` (e4m3 or e5m2), `int8`, `int8_convrot`,
`mxfp4`, `nvfp4`, or `int4`. Uses only the Go standard library - no
third-party packages.

The tool is still under heavy development.

## Build

```
go build -o stconv ./cmd/stconv
```

## Basic usage

`-in` accepts a single `.safetensors` file or a model directory of shards
(see [Multi-file models](#multi-file-models) below); `-out` accepts a file
path - single-file output, the default, always one file - or an existing
directory, which selects explicit multi-file output.

Convert to fp8 (e4m3, the common default for LLM weights - precision-
sensitive tensors stay at their original dtype, see
[Default precision policy](#default-precision-policy)):

```
./stconv -in model.safetensors -out model.fp8.safetensors -target fp8_e4m3
```

Convert to int8:

```
./stconv -in model.safetensors -out model.int8.safetensors -target int8
```

Or to one of the 4-bit / rotated targets (`int8_convrot`, `mxfp4`, `nvfp4`,
`int4` - see [How conversion works](#how-conversion-works) for what each
does):

```
./stconv -in model.safetensors -out model.nvfp4.safetensors -target nvfp4
```

Per-layer selection of any of these targets works via `-config` as before
(see [Per-layer control via config file](#per-layer-control-via-config-file)).

Leave a run untouched (useful with `-config` to only convert specific
layers - see below):

```
./stconv -in model.safetensors -out model.copy.safetensors -target none
```

Skip tiny tensors (e.g. biases/norm vectors) regardless of target:

```
./stconv -in model.safetensors -out model.fp8.safetensors -target fp8_e4m3 -min-elems 1000
```

Tune the streaming chunk size (rarely needed - default is 2^20 elements,
about 2MB at a time for fp16/bf16 input; lower it if you're memory-constrained,
raise it for a bit more throughput on fast disks):

```
./stconv -in model.safetensors -out model.fp8.safetensors -target fp8_e4m3 -chunk-elems 262144
```

### Default precision policy

By default, not every tensor is converted. The tool keeps precision-
sensitive tensors at their original dtype, per `quantization-advice.md`
at the repo root (the reasoning for each category is there):

- **All norm weights** - LayerNorm/RMSNorm, including attention-internal
  `q_norm`/`k_norm` and numbered vision norms.
- **Token embeddings** (`embed_tokens` and family equivalents) and the
  **LM head / output projection**.
- **Attention projections** (`q/k/v/o_proj` and the fused `qkv`,
  `in_proj_qkv`, `out_proj` forms) **when the target is fp8** - this
  tool's fp8 applies no scaling, and these projections must not be fp8
  without per-channel scaling. Scaled targets (int8, int8_convrot, mxfp4,
  nvfp4, int4) convert them as normal.

Protected tensors are reported as unchanged with the policy reason
(e.g. `default policy: norm weights stay at original precision`). Two
escape hatches:

- `-no-protect` disables the policy and converts every tensor to the
  target dtype (the pre-policy behavior).
- An explicit `-config` rule wins over the policy for the tensors it
  matches - e.g. forcing `int8` on a norm weight converts it. When a run
  does this, the tool prints one stderr line after the report:
  `warning: N protected tensor(s) converted by explicit config rules
  (see quantization-advice.md)`.

The policy is best-effort name matching: the patterns are lowercase and
segment-based, so e.g. an uppercase `LayerNorm` segment is not matched (the
tensor gets converted) and a `denorm_*` name is over-matched (the tensor is
kept); both directions are visible in the per-tensor report - a conversion
or an `unchanged: default policy: ...` line - and `-no-protect`/config
rules remain the escape hatches.

### Multi-file models

`-in` also accepts a model directory of sharded `.safetensors` files: the
shards are discovered via the `*.safetensors.index.json` weight map when
present (otherwise in sorted filename order) and merged into the single
output file, in shard order, and the report shows each tensor's source
shard.

```
./stconv -in inputs/Qwen3.5-4B -out out/qwen35_4b.merged.fp8.safetensors -target fp8_e4m3
```

`-out` also accepts an existing directory for multi-file output: the model
is written as one shard per input shard (same filenames) plus a new
`*.safetensors.index.json` whose `weight_map` mirrors the input's
tensor-to-shard assignments. This requires a model-directory input, since
the input's index drives the output layout, and the tool never overwrites
in place: a `-out` path that is an existing file is refused, and so is any
planned shard filename or index name already present in the target
directory - the run is refused up front, before anything is written.

The output directory must exist beforehand - a missing `-out` path is a
single-file output, not a directory to be created:

```
mkdir -p out/qwen35_4b.multi.fp8
./stconv -in inputs/Qwen3.5-4B -out out/qwen35_4b.multi.fp8 -target fp8_e4m3
```

A few details worth knowing:

- **Scale siblings land in their owner's shard.** Each scale sibling
  (`.scale` for int8/int4/int8_convrot, `.block_scale` for mxfp4/nvfp4,
  `.global_scale` for nvfp4) is written into the same output shard as the
  weight it quantizes, and the output index's `weight_map` points it at
  that shard. Scalar scales (int8, int4) follow their owner in the shard;
  vector scales (convrot row scales, mxfp4/nvfp4 block scales) precede it.
  Either way, both live in the owner's shard.
- **`__metadata__` is per source shard.** The merged single-file output
  keeps the *first* shard's `__metadata__` block; in multi-file output,
  each output shard keeps the `__metadata__` block of the input shard it
  mirrors. Merging metadata maps across shards is out of scope.
- **`total_size` is computed, not copied.** The output index's
  `metadata.total_size` is the sum of the output shards' data byte lengths
  (including all scale-sibling bytes), derived from the conversion plan - the
  input's `total_size` describes the input and would be wrong for the
  output. Every other input index metadata key (`architectures` and
  friends) is carried over unchanged; `total_size` is the only key the
  output index overrides.
- **A failed run leaves nothing behind.** If a run fails after its output
  files were created (e.g. a read or write error mid-stream), the partially
  written outputs are closed and removed - the single output file, or every
  output shard plus the index if it was written - so a failed run never
  leaves a truncated model on disk that looks loadable. Input files are
  never touched.

## Per-layer control via config file

For "keep this layer at fp16, quantize that one," pass `-config`:

```
./stconv -in model.safetensors -out model.mixed.safetensors -config config.json
```

`config.json`:

```json
{
  "default": "fp8_e4m3",
  "rules": [
    { "match": "model.layers.0.mlp.down_proj.weight", "dtype": "int8" },
    { "pattern": "\\.q_norm\\.weight$", "dtype": "int8" }
  ]
}
```

- `default` is the target used for any tensor that no rule matches
  (falls back to `-target` on the command line if omitted).
- `rules` are checked in order, first match wins. Each rule matches by
  exact tensor `"match"` name or a regexp `"pattern"` and requires an
  explicit `"dtype"`.
- `"dtype"` is one of `fp8_e4m3`, `fp8_e5m2`, `int8`, `int8_convrot`,
  `mxfp4`, `nvfp4`, `int4`, or `none` (leave the tensor exactly as it is in
  the output).
- Rules take precedence over the [default precision policy](#default-precision-policy):
  no `none` rules are needed to keep norms, embeddings, or the output head
  at their original dtype - that is now the default. A rule that converts
  a protected tensor wins, and the run prints a one-line stderr warning
  for it (see above). The second rule in the example is such an override.

This is intentionally simple to keep the door open for richer per-layer
strategies later (block-wise scales, different int8 calibration,
skip-by-substring shortcuts, etc.) without changing the file format.

## Memory behavior

Model files are often tens or hundreds of GB, so this tool never reads a
whole tensor - let alone a whole file - into memory:

- **Header planning is metadata-only.** Output byte length for any tensor
  depends only on its element count and target dtype, never on the actual
  values, so every tensor's output offset is computed from the *input
  header* alone. With a multi-shard input, planning reads the header of
  *every* shard (metadata only) before any tensor data is touched, which
  is what lets the tool write the complete output header(s) once, up
  front, before touching any tensor data. The streaming pass then opens
  each shard once and keeps it open for the run - open fd count = shard
  count (small and bounded, never per tensor) - and peak memory stays
  `O(chunk size)`.
- **Data is streamed in fixed-size chunks** (`-chunk-elems`, default
  `2^20` elements - about 2MB at a time for fp16/bf16 input) via
  `io.ReaderAt`/`io.Writer`, so peak memory is `O(chunk size)`, not
  `O(tensor size)` or `O(file size)`. Passthrough tensors (kept at their
  original dtype) use `io.Copy` over an `io.SectionReader`, which has the
  same property.
- **Converting targets need multiple passes per tensor** (int8/int4: one to
  find `max(abs(x))` for the scale, one to quantize; int8_convrot/mxfp4/nvfp4:
  up to three - nvfp4 first takes the global max, then writes the block
  scales, then the data), but every pass is chunked the same way and **no
  state is buffered between passes** - row/block scales are *recomputed* on
  the re-read instead of stored (storing a per-row / per-block vector would
  grow with tensor size); the only things that survive are bounded scalars
  and fixed-size group buffers (256/32/16 elements). Peak memory stays
  `O(chunk)`.

Measured on a synthetic 1GB single-tensor file, peak RSS was ~27MB and
did not increase between a 200MB and a 1GB input - confirming memory use
is decoupled from file size in practice, not just in theory.

## Regression testing with the bundled model inputs

The `inputs/Qwen3.5-0.8B` and `inputs/Qwen3.5-4B` directories contain real Hugging Face model
checkouts to use as regression-test inputs (both are gitignored - they are
expected to exist locally but are not part of the repo):

- `inputs/Qwen3.5-0.8B` - Qwen3.5-0.8B, single-shard
  (`model.safetensors-00001-of-00001.safetensors`, ~1.7GB). Good for fast
  smoke/regression runs.
- `inputs/Qwen3.5-4B` - Qwen3.5-4B, two shards
  (`model.safetensors-00001-of-00002.safetensors`,
  `model.safetensors-00002-of-00002.safetensors`, ~9GB total). Good for
  multi-shard and larger-memory-footprint regression runs.

Always write conversion outputs into the `out/` folder (also gitignored),
so test outputs never show up in `git status`:

```bash
mkdir -p out
timeout 1800 ./stconv -in inputs/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors \
    -out out/qwen35_0.8b.fp8.safetensors -target fp8_e4m3
timeout 7200 ./stconv -in inputs/Qwen3.5-4B -out out/qwen35_4b.merged.fp8.safetensors -target fp8_e4m3
```

Note the `timeout` wrappers - per GUIDELINES.md, evaluation/CLI runs must
never be launched unbounded. A useful regression check: rerun a
conversion, compare `out/` file sizes and the per-tensor report between
runs, and watch peak RSS stay flat (see Memory behavior above).

## How conversion works

- **fp8 (e4m3 / e5m2)**: values are cast directly, bit-for-bit per the OCP
  8-bit float spec (the same layouts as PyTorch's `torch.float8_e4m3fn` /
  `torch.float8_e5m2`, and ONNX's `Float8E4M3FN` / `Float8E5M2`). No
  scaling is applied. The mantissa is rounded half-away-from-zero - this
  matches the PyTorch/ONNX *layouts* but not their cast *bytes* at exact
  mantissa midpoints, where torch rounds to nearest-even (RNE);
  `f32ToE4M3RNE` (used only for NVFP4 block scales) is the RNE variant.
  `e4m3` gives more mantissa precision and a max magnitude of 448, but
  has no Inf: magnitudes beyond 448 encode to `0x7F`, the e4m3fn NaN
  pattern (a dequantizing loader reads NaN, not 448), and tiny magnitudes
  flush to zero. `e5m2` gives more exponent range (max magnitude 57344,
  supports Inf) at the cost of precision; its overflow encodes to +/-Inf,
  which stays accurate.
- **int8**: quantized per-tensor, symmetric, using
  `scale = max(abs(tensor)) / 127`, `q = round(x / scale)` clamped to
  `[-127, 127]`. Since int8 has no implicit scale, a small sibling scalar
  tensor named `"<original_name>.scale"` (dtype `F32`) is added next to
  each quantized weight so it can be dequantized later. There's no
  standardized safetensors field for this - storing a scale tensor is the
  convention several existing quantization tools use - so treat this as
  a defined-but-not-universal convention for this tool specifically.
- **int8_convrot**: per `int8_convrot_guide.md` - each 256-element row-major
  group is rotated by the orthonormal regular Hadamard (y = H_256·x/16)
  **before** quantization; per-row symmetric int8 scale = `rowMax/127`
  (row = first dimension); values clamped to `[-127, 127]`. This is a
  **value-changing conversion**: the output weights are the *rotated*
  weights - a loader must apply the same Hadamard rotation to activations
  at inference time (serving-side pairing, out of scope for this tool).
  Tensors whose element count is not a multiple of 256 are left unchanged
  (reported as skipped).
- **mxfp4** (OCP microscaling): E2M1 4-bit elements (values
  0/0.5/1/1.5/2/3/4/6, max magnitude 6) in 32-element blocks; per-block
  E8M0 scale = smallest power of two >= `blockMax/6` (so the scaled block
  max lands in (3,6]); round-to-nearest-even element rounding.
- **nvfp4** (NVIDIA FP4): E2M1 elements in 16-element blocks; per-block
  E4M3 scale (round-to-nearest-even, saturating at 448) plus a per-tensor
  F32 global scale `α = max|x| / (6·448)`; dequant is `q·s·α`.
- **int4**: naive per-tensor symmetric int4, `scale = max|x| / 7`,
  round-to-nearest-even, clamped to `[-7, 7]`; a `.scale` sibling like
  int8's.
- **Storage convention for the 4-bit targets.** The safetensors spec has no
  4-bit dtype, so packed 4-bit data (2 elements per byte, element `2i` in
  the low nibble) and E8M0 block scales are stored as `U8`; NVFP4 block
  scales as `F8_E4M3`; all scalar/row scales as `F32`. Sibling names:
  `.scale` (int8, int4, int8_convrot), `.block_scale` (mxfp4, nvfp4),
  `.global_scale` (nvfp4). Scalar-scale siblings follow their owner in the
  file; vector-scale siblings (convrot row scales, mxfp4/nvfp4 block
  scales) precede it - in the file and in the output index's `weight_map`
  order.
- Non-float tensors (int/bool weights, buffers, etc.) are always copied
  through unchanged - quantizing already-integer tensors is out of scope.
- Tensor order and any `__metadata__` block from the input header are
  preserved in the output (multi-shard inputs: first shard's block in the
  merged file, per-shard in multi-file output - see Multi-file models).

## Project layout

- `cmd/stconv/main.go` - CLI flags and per-tensor report. A thin layer:
  all logic lives in `internal/stconv`.
- `internal/stconv/header.go` - safetensors header parsing/writing
  (order-preserving JSON handling, since Go maps don't preserve key
  order).
- `internal/stconv/dtype.go` - float32 <-> {float16, bfloat16,
  float8_e4m3fn, float8_e5m2, int8, e2m1 (FP4 element), e8m0 (MXFP4 block
  scale), int4} conversions, implemented from the bit-level spec of each
  format (including the RNE e4m3 encoder `f32ToE4M3RNE` used only by the
  NVFP4 block-scale path).
- `internal/stconv/dtype_test.go` - unit tests checking each conversion
  against known reference values (e.g. e4m3 max magnitude 448, e5m2 max
  magnitude 57344, round-trip tolerances).
- `internal/stconv/config.go` - JSON config file parsing for per-layer
  overrides.
- `internal/stconv/protect.go` - the built-in default precision policy:
  name patterns for the tensors kept at their original dtype by default
  (norms, token embeddings, output head, and attention projections for
  fp8 targets), per `quantization-advice.md`.
- `internal/stconv/index.go` - Hugging Face `*.safetensors.index.json`
  parsing (`weight_map`, metadata), shard discovery from a model directory
  (index-driven, with sorted-glob fallback), and the `-in`/`-out` path
  resolution (`ResolveInput`, `ResolveOutput`).
- `internal/stconv/shardout.go` - output sharding plan: splits the global
  tensor plan into per-shard output headers and the output index (shard
  names and `weight_map` assignments mirror the input's; `total_size` is
  recomputed from the plan).
- `internal/stconv/convert.go` - orchestrates reading tensors, deciding a
  target per tensor, converting, and building the output file (single
  merged file, or per-shard files plus index).
- `internal/stconv/quant.go` - the streaming conversion passes for the four
  new targets (int4, int8_convrot, mxfp4, nvfp4), including the 256-element
  Hadamard rotation; each pass is chunked and bounded per the memory rules.
- `testdata/gen/gen.go` - standalone generator for small synthetic
  `.safetensors` fixtures for manual end-to-end tests: `single` writes
  one 3-tensor file, `multi` writes a 2-shard model directory with an
  index (shard 2 carries an I32 buffer to exercise passthrough),
  `single256` writes one BF16 [2,128] tensor with a strong outlier
  (256 elements - a ConvRot rotation group), `multirot` writes a
  2-shard model directory of 256-element tensors (one [128,2] so
  rotation groups straddle rows), `qwenlike` writes one BF16 file whose
  tensor names follow the bundled Qwen models' naming (embedding,
  layernorms, q/k/v/o_proj, q/k_norm, mlp projections, final norm,
  lm_head, plus an I32 buffer) - the fixture for the default-precision-
  policy e2e test, and `empty` writes one BF16 file whose header order
  is a zero-element [0,8] tensor, a 256-element tensor (a ConvRot
  rotation group), a zero-element [0] tensor, and a 16-element tensor
  (not a 256-multiple) - the fixture for the zero-element convrot
  regression test. The generated fixtures are committed under
  `testdata/single`, `testdata/multi`, `testdata/single256`,
  `testdata/multirot`, `testdata/qwenlike`, and `testdata/empty`.

## Known limitations / next steps

- Unify fp8 rounding on RNE with a golden refresh - deliberately not done
  in the review-fix round. The `fp8_e4m3`/`fp8_e5m2` casts round the
  mantissa half-away-from-zero; switching them to round-to-nearest-even
  (matching torch's cast bytes at exact mantissa midpoints) is a behavior
  change that requires regenerating the fp8 golden tests, so it is
  deferred as an explicit decision (see How conversion works).
- int8 scale is a single scalar per tensor (no per-channel/group-wise
  quantization yet) - per-channel scales would meaningfully improve
  accuracy for weights with outlier channels and would be the natural
  next step. Note this would also mean per-channel max-abs scanning
  during the streaming pass, which is a straightforward extension of the
  existing chunked scan.
- No CUDA/SIMD - this processes weights on CPU, tensor by tensor, single
  threaded. Fine for offline conversion; not built for serving-time
  speed. Tensor-level conversion is embarrassingly parallel if throughput
  ever matters more than simplicity (each tensor's plan is independent).
- int8_convrot uses plain `max/127` row scales - the guide's `--mseclip`
  clip-boundary optimization is not implemented (a natural next step).
  The rotation group size is fixed at 256 (no flag), and groups are flat
  row-major - identical to per-row last-dimension grouping whenever the
  row width is a multiple of 256, which covers typical LLM linear layers.
  As a value-changing conversion it also needs a serving side that applies
  the matching activation rotation (see How conversion works).
- mxfp4, nvfp4, and int4 outputs need a loader with the matching
  dequantization path - the same "defined-but-not-universal convention"
  caveat as the int8 `.scale` note.
- The `.scale` sibling-tensor convention for int8 is this tool's own
  choice, not a safetensors standard - if you need compatibility with a
  specific downstream loader (e.g. a particular inference engine's
  expected int8 format), check what it expects and adjust
  `convert.go`'s `TargetInt8` branch accordingly.
