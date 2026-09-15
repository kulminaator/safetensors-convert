// Tests for the progress reporter: the exact captured byte stream of a
// scripted Begin/Add/End sequence over three tensors under a manually
// advanced fake clock (pinning the line format and the 100ms throttle),
// the empty-tensor "done" form, name truncation at 50 chars, index
// padding for 1-digit vs 3-digit totals, and Begin without SetTotal.
package stconv

import (
	"bytes"
	"strings"
	"testing"
	"time"
)

// fakeClock is a manually advanced clock for driving Progress.now.
type fakeClock struct {
	t time.Time
}

func (c *fakeClock) Now() time.Time { return c.t }
func (c *fakeClock) add(d time.Duration) {
	c.t = c.t.Add(d)
}

// newTestProgress returns a reporter writing to buf with the fake
// clock installed, starting at an arbitrary nonzero time.
func newTestProgress(buf *bytes.Buffer, clk *fakeClock) *Progress {
	p := NewProgress(buf)
	p.now = clk.Now
	return p
}

const testMiB = int64(1) << 20

// TestProgressScriptedRun asserts the exact byte stream of a full
// run: three tensors, the first with five Adds spread over 40ms (the
// throttle must emit exactly one live update for them) and a sixth Add
// 150ms later (the next update), plus a live update and final line per
// remaining tensor. Every rate/eta value is hand-computed from the
// script so the format, the pct/rate/eta formulas, and the throttle
// are all pinned at once:
//
//	t=0     Begin alpha (6MiB)            -> [1/3] alpha  0%
//	t=100ms Add 1MiB  (100ms since begin) -> 16%  10.0MB/s  eta 0.5s
//	t=110..140ms Add 1MiB x4 (10ms each)  -> no updates
//	t=290ms Add 1MiB  (150ms since last)  -> 100%  20.7MB/s  eta 0.0s
//	t=290ms End                          -> final line + \n
//	t=290ms Begin beta (2MiB)            -> [2/3] beta  0%  20.7MB/s  eta 0.1s
//	t=390ms Add 2MiB                     -> 100%  20.5MB/s  eta 0.0s
//	t=390ms End                          -> final line + \n
//	t=390ms Begin gamma (1MiB)           -> [3/3] gamma  0%  20.5MB/s  eta 0.0s
//	t=490ms Add 1MiB                     -> 100%  18.4MB/s  eta 0.0s
//	t=490ms End                          -> final line + \n
func TestProgressScriptedRun(t *testing.T) {
	clk := &fakeClock{t: time.Unix(1_000_000, 0)}
	var buf bytes.Buffer
	p := newTestProgress(&buf, clk)

	p.SetTotal(3)
	p.Begin("alpha", DTypeF16, DTypeI8, 6*testMiB)
	clk.add(100 * time.Millisecond)
	p.Add(testMiB)
	clk.add(10 * time.Millisecond)
	p.Add(testMiB)
	clk.add(10 * time.Millisecond)
	p.Add(testMiB)
	clk.add(10 * time.Millisecond)
	p.Add(testMiB)
	clk.add(10 * time.Millisecond)
	p.Add(testMiB)
	clk.add(150 * time.Millisecond)
	p.Add(testMiB)
	p.End()

	p.Begin("beta", DTypeF16, DTypeF8E4M3, 2*testMiB)
	clk.add(100 * time.Millisecond)
	p.Add(2 * testMiB)
	p.End()

	p.Begin("gamma", DTypeBF16, DTypeF8E4M3, testMiB)
	clk.add(100 * time.Millisecond)
	p.Add(testMiB)
	p.End()

	want := strings.Join([]string{
		"\r[1/3] alpha  0%",
		"\r[1/3] alpha  16%  10.0MB/s  eta 0.5s",
		"\r[1/3] alpha  100%  20.7MB/s  eta 0.0s",
		"\r[1/3] alpha  100%  20.7MB/s  eta 0.0s\n",
		"\r[2/3] beta  0%  20.7MB/s  eta 0.1s",
		"\r[2/3] beta  100%  20.5MB/s  eta 0.0s",
		"\r[2/3] beta  100%  20.5MB/s  eta 0.0s\n",
		"\r[3/3] gamma  0%  20.5MB/s  eta 0.0s",
		"\r[3/3] gamma  100%  18.4MB/s  eta 0.0s",
		"\r[3/3] gamma  100%  18.4MB/s  eta 0.0s\n",
	}, "")
	if got := buf.String(); got != want {
		t.Errorf("progress stream mismatch:\ngot:  %q\nwant: %q", got, want)
	}
}

// TestProgressEmptyTensor: totalWork == 0 renders the "done" form
// without a percentage or an eta (no divide-by-zero), on both the
// initial line and the final line.
func TestProgressEmptyTensor(t *testing.T) {
	clk := &fakeClock{t: time.Unix(1_000_000, 0)}
	var buf bytes.Buffer
	p := newTestProgress(&buf, clk)

	p.SetTotal(2)
	p.Begin("empty", DTypeBF16, DTypeBF16, 0)
	clk.add(50 * time.Millisecond)
	p.End()

	want := "\r[1/2] empty  done" + "\r[1/2] empty  done\n"
	if got := buf.String(); got != want {
		t.Errorf("progress stream mismatch:\ngot:  %q\nwant: %q", got, want)
	}
	if strings.Contains(buf.String(), "%") || strings.Contains(buf.String(), "eta") {
		t.Errorf("empty-tensor line must not show a percentage or eta, got %q", buf.String())
	}
}

// TestProgressNameTruncation: names longer than 50 chars are truncated
// to the report's width; a name of exactly 50 chars is left alone.
func TestProgressNameTruncation(t *testing.T) {
	clk := &fakeClock{t: time.Unix(1_000_000, 0)}
	var buf bytes.Buffer
	p := newTestProgress(&buf, clk)

	long := strings.Repeat("x", 60)
	p.Begin(long, DTypeF16, DTypeI8, 100)
	want1 := "\r[1] " + strings.Repeat("x", 50) + "  0%"
	if got := buf.String(); got != want1 {
		t.Errorf("truncated name mismatch:\ngot:  %q\nwant: %q", got, want1)
	}

	buf.Reset()
	exact := strings.Repeat("y", 50)
	p.Begin(exact, DTypeF16, DTypeI8, 100)
	want2 := "\r[2] " + exact + "  0%"
	if got := buf.String(); got != want2 {
		t.Errorf("50-char name mismatch:\ngot:  %q\nwant: %q", got, want2)
	}
}

// TestProgressIndexPadding: the index and total are padded to the
// width of N (1/3 vs 1/488).
func TestProgressIndexPadding(t *testing.T) {
	clk := &fakeClock{t: time.Unix(1_000_000, 0)}
	var buf bytes.Buffer
	p := newTestProgress(&buf, clk)

	p.SetTotal(3)
	p.Begin("a", DTypeF16, DTypeI8, 100)
	if got := buf.String(); got != "\r[1/3] a  0%" {
		t.Errorf("N=3 padding mismatch: %q", got)
	}

	buf.Reset()
	p.SetTotal(488)
	p.Begin("a", DTypeF16, DTypeI8, 100)
	if got := buf.String(); got != "\r[  2/488] a  0%" {
		t.Errorf("N=488 padding mismatch: %q", got)
	}
}

// TestProgressBeginWithoutSetTotal: a Begin before SetTotal (defensive)
// renders the index without the total.
func TestProgressBeginWithoutSetTotal(t *testing.T) {
	clk := &fakeClock{t: time.Unix(1_000_000, 0)}
	var buf bytes.Buffer
	p := newTestProgress(&buf, clk)

	p.Begin("a", DTypeF16, DTypeI8, 100)
	if got := buf.String(); got != "\r[1] a  0%" {
		t.Errorf("no-total render mismatch: %q", got)
	}
}
