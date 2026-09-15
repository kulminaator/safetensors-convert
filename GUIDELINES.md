# Development Guidelines

These are the rules we build this project by. They apply to every phase and
every step. When a change conflicts with a rule, fix the change, not the rule.

## 1. Code: minimalistic, clean Go

- **Standard library only.** No third-party dependencies, ever. If a need
  seems to require one, implement the small piece we actually need by hand.
- **Traditional layout, minimal packages.** `cmd/stconv/` holds the thin
  CLI (`main.go`: flags + report); `internal/stconv/` holds all logic,
  split by concern (`header.go`, `dtype.go`, `config.go`, `convert.go`).
  No further package splits, no interfaces, plugins, or indirection "for
  the future". Add structure only when a real second use case exists.
- **Small functions, flat control flow.** A function does one thing and fits
  on one screen. Prefer early returns over nested `if/else` pyramids.
- **Errors are explicit.** No panics in library paths, no swallowed errors.
  Wrap with context (`fmt.Errorf("...: %w", err)`) where it helps.
- **Comments explain *why*, not *what*.** File-level doc comments state the
  file's job; tricky bit-level math (dtype conversions, header layout) gets
  the spec reasoning written down.
- **Naming and formatting are non-negotiable.** `gofmt` clean, no `go vet`
  or `staticcheck`-level warnings. No dead code, no commented-out blocks.
- **Keep the surface small.** New flags, config fields, and exported
  concepts must earn their place. The CLI and `config.json` formats are
  intentionally simple; extend them only when a concrete need is documented.

## 2. Process: phases and steps, tests gate everything

Development proceeds in **phases**, and each phase is broken into small
**steps**. The rules:

1. **One step = one bounded change.** A step is something you can describe
   in one sentence ("stream int8 max-abs scan in chunks", "add `min-elems`
   flag"). If it needs two sentences, split it.
2. **Every step ships with its tests.** The tests are written for the code
   the step generates, and they must cover that code - new functions get
   unit tests, new behavior gets at least one test that would fail without
   the change. A step without tests is not a finished step.
3. **Tests must pass after every build.** Before a step is done:
   ```
   gofmt -l . && go vet ./... && go build ./... && timeout 600 go test -timeout 300s ./...
   ```
   All four must be clean. The suite takes ~2 minutes (the exhaustive
   fp8/int8 oracle tests), so the timeouts below are sized from that
   expected runtime (rule 7), not from the old sub-minute suite. No "will fix the test later", no skipping tests,
   no weakening assertions to make them pass.
4. **No step merges on red.** A failing or newly flaky test blocks the next
   step. Fix it first.
5. **Update the record when a step lands.** The README (behavior, usage,
   memory notes) and this file (if a rule evolved) are updated in the same
   change as the code.
6. **Commit to git often.** Commit at every green checkpoint - at minimum,
   when a step passes the gate in rule 3. If a step turns out to be bigger
   than expected, commit the working intermediate state before going
   deeper so a rollback is always one `git reset` away. One step, one (or
   a few small) commit(s); no giant end-of-phase dumps, no committing red
   (failing) trees. Never commit model weights or large generated files -
   only source, tests, `testdata/` fixtures, and docs.
7. **Every test and evaluation run has a timeout.** Never launch a test
   suite or an evaluation/CLI run (e.g. converting a large model file,
   benchmarking, memory checks) unattended and unbounded. Wrap them:
   `timeout <N> go test ...` / `timeout <N> ./stconv ...` (and keep
   `go test -timeout` set for the in-process tests). A run that hits its
   timeout is a failure, not a "wait longer" situation - investigate the
   hang (infinite loop, blocking I/O, deadlocked conversion) before
   re-running, and re-run with a timeout. Pick N from the expected runtime
   (generous, e.g. 5–10x), not from hope.

Test style: plain `testing`, table-driven where it helps, no external test
frameworks. Use `testdata/` for generated fixture files instead of committing
large binaries.

## 3. Memory: conservative by construction

Model files are tens to hundreds of GB. This tool must treat a 500GB model
the same way it treats a 5MB one.

- **Never hold more than a bounded chunk.** Peak memory is `O(chunk)`,
  where chunk is `-chunk-elems` (default `2^20` elements ≈ 2MB). Anything
  that scales with tensor size or file size is a bug.
- **Never load a whole tensor - let alone the model.** All I/O goes through
  streaming: `io.ReaderAt`/`io.Writer` for conversion, `io.Copy` over
  `io.SectionReader` for passthrough. No `os.ReadFile` on data, no
  `[]byte` that grows with the tensor.
- **Open file descriptors are bounded like memory.** A run opens at most
  one fd per input shard (and one per output shard in multi-file output)
  and keeps them open for the run - `O(shard count)`, never per tensor -
  with the bound stated in a comment at the open site.
- **Plan from metadata, not data.** Anything derivable from the header
  (dtypes, shapes, element counts, output offsets) is computed before
  touching tensor bytes. This is what lets the output header be written
  once, up front.
- **Multi-pass is fine; buffering between passes is not.** int8 needs a
  max-abs scan and a quantize pass. Both are chunked; the only thing that
  may survive between passes is a scalar (the scale). If a new feature
  needs more state between passes, it must itself be bounded (a fixed-size
  table) and that bound must be stated in a comment.
- **Chunks are a tunable, not a constant.** When memory pressure is
  reported, the first answer is "lower `-chunk-elems`", and the code must
  actually honor that - every allocation path must go through the chunk
  size, not a hard-coded buffer.
- **Verify, don't assume.** Memory-sensitive steps should be spot-checked
  (e.g. peak RSS on a large synthetic file via `testdata/gen`) so that
  "memory use is decoupled from file size" stays a measured fact, not a
  claim.

## 4. Definition of done

A step is done when all of the following hold:

- [ ] The change is the smallest that delivers the step's sentence.
- [ ] New/changed code has tests written for it; they fail without the change.
- [ ] `gofmt`, `go vet`, `go build`, and `go test` are all clean.
- [ ] No allocation path grows with tensor or file size.
- [ ] README (and these guidelines, if applicable) reflect the change.
- [ ] The change is committed to git (only source/tests/fixtures/docs).
