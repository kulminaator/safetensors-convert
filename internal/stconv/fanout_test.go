// Tests for the fan-out primitives: every unit of [0, n) is handled
// exactly once, the contiguous partition's slots and ranges follow the
// documented formula, the strided pattern assigns each stride class
// i%k to one worker, and the n <= 1 case runs inline without
// goroutines. Run under -race: the per-index byte bitmaps below are
// written by whichever worker owns the index, so a partition bug that
// overlaps ranges shows up as a data race.
package stconv

import (
	"fmt"
	"runtime"
	"slices"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
)

// testNs are the unit counts every fan-out test runs: the degenerate
// cases, a small unevenly divisible count, and a count large enough to
// exercise the multi-worker path on any machine.
var testNs = []int{0, 1, 2, 7, 1000}

// goroutineID returns the current goroutine's numeric id, parsed from
// the first line of runtime.Stack ("goroutine N [state]:"). The
// standard library has no exported goroutine id; this is test-only.
func goroutineID() int {
	var buf [128]byte
	n := runtime.Stack(buf[:], false)
	s := string(buf[:n])
	const prefix = "goroutine "
	if len(s) < len(prefix) || s[:len(prefix)] != prefix {
		panic("fanout_test: unexpected runtime.Stack output: " + s)
	}
	j := len(prefix)
	for j < len(s) && s[j] >= '0' && s[j] <= '9' {
		j++
	}
	id, err := strconv.Atoi(s[len(prefix):j])
	if err != nil {
		panic("fanout_test: cannot parse goroutine id: " + s)
	}
	return id
}

// TestParallelism pins parallelism's contract: 0 for n = 0, 1 for
// n = 1, and min(n, NumCPU) above that.
func TestParallelism(t *testing.T) {
	cpus := runtime.NumCPU()
	for _, n := range []int{0, 1, 2, 3, 7, cpus - 1, cpus, cpus + 1, 1000} {
		want := n
		if n > 1 && cpus < want {
			want = cpus
		}
		if got := parallelism(n); got != want {
			t.Errorf("parallelism(%d) = %d, want %d", n, got, want)
		}
	}
}

// TestMapContiguousEveryIndexOnce checks that mapContiguous's workers
// together visit every index in [0, n) exactly once: the atomic counter
// totals the visits, the per-index byte bitmap catches a double visit,
// and a short total catches a miss.
func TestMapContiguousEveryIndexOnce(t *testing.T) {
	for _, n := range testNs {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			var visits, dups atomic.Int64
			seen := make([]byte, n)
			mapContiguous(n, func(_, lo, hi int) {
				for i := lo; i < hi; i++ {
					if seen[i] != 0 {
						dups.Add(1)
					}
					seen[i] = 1
					visits.Add(1)
				}
			})
			if got := visits.Load(); got != int64(n) {
				t.Errorf("visited %d indices, want %d", got, n)
			}
			if d := dups.Load(); d != 0 {
				t.Errorf("%d index(es) visited more than once", d)
			}
		})
	}
}

// TestMapStridedEveryIndexOnce is the same exactly-once check for
// mapStrided.
func TestMapStridedEveryIndexOnce(t *testing.T) {
	for _, n := range testNs {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			var visits, dups atomic.Int64
			seen := make([]byte, n)
			mapStrided(n, func(i int) {
				if seen[i] != 0 {
					dups.Add(1)
				}
				seen[i] = 1
				visits.Add(1)
			})
			if got := visits.Load(); got != int64(n) {
				t.Errorf("visited %d indices, want %d", got, n)
			}
			if d := dups.Load(); d != 0 {
				t.Errorf("%d index(es) visited more than once", d)
			}
		})
	}
}

// TestMapContiguousPartition checks the partition contract: one
// invocation per worker with distinct slots within [0, k), worker slot
// owning exactly [slot*n/k, (slot+1)*n/k), and the ranges disjoint and
// covering [0, n).
func TestMapContiguousPartition(t *testing.T) {
	for _, n := range testNs {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			k := parallelism(n)
			var mu sync.Mutex
			var calls [][3]int // (slot, lo, hi) per invocation
			mapContiguous(n, func(slot, lo, hi int) {
				mu.Lock()
				calls = append(calls, [3]int{slot, lo, hi})
				mu.Unlock()
			})
			wantCalls := k
			if wantCalls <= 1 {
				wantCalls = 1
			}
			if len(calls) != wantCalls {
				t.Fatalf("got %d worker invocations, want %d", len(calls), wantCalls)
			}
			seenSlot := make([]bool, wantCalls)
			var ranges [][2]int
			for _, c := range calls {
				slot, lo, hi := c[0], c[1], c[2]
				if slot < 0 || slot >= wantCalls {
					t.Errorf("slot %d outside [0, %d)", slot, wantCalls)
					continue
				}
				if seenSlot[slot] {
					t.Errorf("slot %d used by two workers", slot)
				}
				seenSlot[slot] = true
				if k > 1 {
					if lo != slot*n/k || hi != (slot+1)*n/k {
						t.Errorf("slot %d owns [%d, %d), want [%d, %d)", slot, lo, hi, slot*n/k, (slot+1)*n/k)
					}
				} else if slot != 0 || lo != 0 || hi != n {
					t.Errorf("inline invocation (%d, %d, %d), want (0, 0, %d)", slot, lo, hi, n)
				}
				ranges = append(ranges, [2]int{lo, hi})
			}
			// Disjoint and covering: the sorted ranges tile [0, n)
			// exactly.
			slices.SortFunc(ranges, func(a, b [2]int) int { return a[0] - b[0] })
			edge := 0
			for _, r := range ranges {
				if r[0] != edge {
					t.Errorf("ranges overlap or leave a gap at %d (next range [%d, %d))", edge, r[0], r[1])
				}
				edge = r[1]
			}
			if edge != n {
				t.Errorf("ranges end at %d, want %d", edge, n)
			}
		})
	}
}

// TestMapInlinePath checks the k <= 1 degradation: for n <= 1 both
// helpers run fn in the caller's goroutine (no goroutines spawned) -
// mapContiguous with exactly one (0, 0, n) invocation, mapStrided with
// one invocation per index (none for n = 0).
func TestMapInlinePath(t *testing.T) {
	self := goroutineID()
	for _, n := range []int{0, 1} {
		var contCalls, strCalls, contID, strID int
		mapContiguous(n, func(_, _, _ int) {
			contCalls++
			contID = goroutineID()
		})
		mapStrided(n, func(int) {
			strCalls++
			strID = goroutineID()
		})
		if contID != self {
			t.Errorf("n=%d: mapContiguous ran fn in goroutine %d, want caller %d (inline, no goroutines)", n, contID, self)
		}
		if strCalls > 0 && strID != self {
			t.Errorf("n=%d: mapStrided ran fn in goroutine %d, want caller %d (inline, no goroutines)", n, strID, self)
		}
		if contCalls != 1 {
			t.Errorf("n=%d: mapContiguous invoked fn %d times, want 1", n, contCalls)
		}
		if strCalls != n {
			t.Errorf("n=%d: mapStrided invoked fn %d times, want %d", n, strCalls, n)
		}
	}
}

// TestMapStridedPattern pins the strided assignment on the
// multi-worker path: every index i is handled by the worker owning
// stride class i%k - all indices of one class by one worker, and
// different classes by different workers. Worker identity is the
// goroutine id; the barrier makes all k workers arrive (and hence be
// alive, with distinct ids) before any of them can finish.
func TestMapStridedPattern(t *testing.T) {
	const n = 1000
	k := parallelism(n)
	if k <= 1 {
		t.Skipf("parallelism(%d) = %d, no multi-worker path on this machine", n, k)
	}
	var (
		mu       sync.Mutex
		cond     = sync.NewCond(&mu)
		seen     = make(map[int]bool)
		arrivals = 0
	)
	owner := make([]int, n)
	mapStrided(n, func(i int) {
		id := goroutineID()
		owner[i] = id
		mu.Lock()
		if !seen[id] {
			seen[id] = true
			arrivals++
			cond.Broadcast()
		}
		for arrivals < k {
			cond.Wait()
		}
		mu.Unlock()
	})
	if len(seen) != k {
		t.Fatalf("%d distinct workers arrived, want %d", len(seen), k)
	}
	for i := 0; i < n; i++ {
		for j := i + k; j < n; j += k {
			if owner[i] != owner[j] {
				t.Fatalf("indices %d and %d share stride class %d but ran in different workers", i, j, i%k)
			}
		}
		for j := i + 1; j < n; j++ {
			if j%k != i%k && owner[i] == owner[j] {
				t.Fatalf("indices %d and %d are different stride classes but ran in the same worker", i, j)
			}
		}
	}
}
