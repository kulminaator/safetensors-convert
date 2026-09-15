// progress.go holds the live progress reporter: one line on the writer
// (the CLI passes os.Stderr) showing which tensor is being converted,
// how far along it is, and the run's rate and ETA. The line is redrawn
// in place with \r and throttled to at most one live update per 100ms;
// Begin and End always write, so each tensor leaves one final line
// (100%, newline) on top of the live updates it produced while running.
//
// Progress is a concrete type, not an interface: the only seam is the
// unexported now field (default time.Now), so tests can drive time
// deterministically.
package stconv

import (
	"fmt"
	"io"
	"strconv"
	"strings"
	"time"
)

// minUpdateInterval bounds the live (Add) update rate: at most one
// update per interval. Begin and End bypass it.
const minUpdateInterval = 100 * time.Millisecond

// nameWidth is the tensor-name field width, matching the per-tensor
// report's 50-char name column.
const nameWidth = 50

// Progress renders a live "[i/N] name  pct%  rate  eta" line on w.
type Progress struct {
	w   io.Writer
	now func() time.Time // test seam; time.Now by default

	total  int // tensors in the run (SetTotal; 0 = not set)
	cur    int // 1-based index of the tensor being converted
	active bool

	name      string
	fromDType DType // carried from Begin for the caller's context
	toDType   DType
	totalWork int64 // the current tensor's total bytes of work
	done      int64 // bytes of work done on the current tensor

	runDone      int64 // bytes of work done in the run
	runRemaining int64 // work not yet done: each Begin adds its tensor's totalWork, each Add removes it
	start        time.Time
	lastWrite    time.Time
}

// NewProgress returns a reporter that writes to w (the CLI passes
// os.Stderr).
func NewProgress(w io.Writer) *Progress {
	return &Progress{w: w, now: time.Now}
}

// SetTotal sets the total tensor count in the run, used to render the
// [i/N] index. ConvertModel calls it once with len(plans) before the
// first Begin.
func (p *Progress) SetTotal(n int) {
	p.total = n
}

// Begin starts one tensor's line and writes it immediately (the initial
// 0% line). totalWork is the tensor's total bytes of work (input bytes
// times the target's pass count; the accounting is defined where Begin
// is called). A Begin on a tensor that was never End-ed finalizes that
// tensor first, so the run counters stay consistent.
func (p *Progress) Begin(name string, fromDType, toDType DType, totalWork int64) {
	if p.active {
		p.finalize()
	}
	if p.start.IsZero() {
		p.start = p.now()
	}
	p.cur++
	p.name = name
	p.fromDType = fromDType
	p.toDType = toDType
	p.totalWork = totalWork
	p.done = 0
	p.active = true
	p.runRemaining += totalWork
	p.write(false)
}

// Add records n bytes of work done on the current tensor and writes a
// live update if at least minUpdateInterval has passed since the last
// write. Add is called once per chunk from the passes' chunk loop,
// which is single-threaded (the fan-out workers run inside the chunk's
// compute phase and are joined before the loop continues), so no
// locking is needed.
func (p *Progress) Add(n int64) {
	if !p.active {
		return
	}
	p.done += n
	p.runDone += n
	p.runRemaining -= n
	if p.now().Sub(p.lastWrite) < minUpdateInterval {
		return
	}
	p.write(false)
}

// End finalizes the current tensor's line at 100% and writes it with a
// trailing newline, so the writer ends up with one final line per
// tensor on top of the live updates.
func (p *Progress) End() {
	if !p.active {
		return
	}
	p.finalize()
	p.write(true)
}

// finalize reconciles the run counters with the current tensor's
// accounting (in case the caller stopped Adding before totalWork) and
// clears the tensor.
func (p *Progress) finalize() {
	missing := p.totalWork - p.done
	p.runDone += missing
	p.runRemaining -= missing
	p.done = p.totalWork
	p.active = false
}

// render formats the current line; it is the single place that builds
// it, so the tests can pin the format without any I/O:
//
//	\r[  12/488] model.layers.11.mlp.down_proj.weight  47%  812.3MB/s  eta 3.2s
//
// The index and total are right-aligned to the width of N; the name is
// truncated to nameWidth chars (the report's width). An empty tensor
// (totalWork == 0) renders "  done" instead of a percentage and an eta,
// so nothing divides by zero. The rate (run work done / elapsed) and
// the eta (run work remaining / rate) are only shown once the run has a
// nonzero elapsed time and rate.
func (p *Progress) render() string {
	var b strings.Builder
	b.WriteString("\r[")
	if p.total > 0 {
		width := len(strconv.Itoa(p.total))
		fmt.Fprintf(&b, "%*d/%d", width, p.cur, p.total)
	} else {
		fmt.Fprintf(&b, "%d", p.cur)
	}
	b.WriteString("] ")
	b.WriteString(truncateName(p.name))
	if p.totalWork == 0 {
		b.WriteString("  done")
		return b.String()
	}
	pct := p.done * 100 / p.totalWork
	if pct > 100 {
		pct = 100
	}
	fmt.Fprintf(&b, "  %d%%", pct)
	elapsed := p.now().Sub(p.start)
	if elapsed > 0 && p.runDone > 0 {
		rate := float64(p.runDone) / elapsed.Seconds()
		fmt.Fprintf(&b, "  %.1fMB/s", rate/(1024*1024))
		remaining := p.runRemaining
		if remaining < 0 {
			remaining = 0
		}
		fmt.Fprintf(&b, "  eta %.1fs", float64(remaining)/rate)
	}
	return b.String()
}

// write renders the current line and writes it to w, adding a newline
// when final. Write errors are discarded: the progress line is
// display-only, and the run's real output goes through the
// conversion's own writer.
func (p *Progress) write(final bool) {
	line := p.render()
	if final {
		line += "\n"
	}
	io.WriteString(p.w, line)
	p.lastWrite = p.now()
}

func truncateName(name string) string {
	if len(name) > nameWidth {
		return name[:nameWidth]
	}
	return name
}
