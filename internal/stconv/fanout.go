// fanout.go holds the two small parallel fan-out primitives the
// streaming conversion passes use to parallelize work within a single
// chunk.
//
// The passes parallelize within a chunk rather than across tensors:
// the output must be written in file order without buffering
// per-tensor state (memory rule: peak memory stays O(chunk), and a
// whole-tensor output buffer is forbidden), so the chunk already in
// memory is the only safe unit of parallelism.
package stconv

import (
	"runtime"
	"sync"
)

// parallelism returns the worker count for a unit count n:
// min(n, runtime.NumCPU()), with n <= 1 passing through as 0 or 1. The
// bound is the CPU count, never n: more workers than cores buy nothing,
// and a worker count that scaled with n would scale per-worker scratch
// (e.g. a worker-local max) with tensor size, which the memory rules
// forbid.
func parallelism(n int) int {
	if n <= 1 {
		return n
	}
	return min(n, runtime.NumCPU())
}

// mapContiguous splits [0, n) into k := parallelism(n) contiguous,
// near-equal ranges - worker slot owns exactly [slot*n/k, (slot+1)*n/k)
// - and calls fn once per worker, concurrently for k > 1. For k <= 1
// (n <= 1) it runs fn(0, 0, n) inline in the caller's goroutine and
// spawns no goroutines; note that for n = 0 the inline call still
// happens, with the empty range, so per-worker scratch indexed by slot
// needs length at least 1 (or the caller skips the call for n = 0).
//
// k is bounded by runtime.NumCPU() (see parallelism): the helper
// allocates nothing that scales with n, and callers must preserve that
// - fn may only touch the disjoint indices of its own [lo, hi) range,
// never a shared element. The slot argument lets a caller keep
// per-worker scratch (e.g. a worker-local max) in a slice of length
// parallelism(n) and reduce it after the join.
func mapContiguous(n int, fn func(slot int, lo, hi int)) {
	k := parallelism(n)
	if k <= 1 {
		fn(0, 0, n)
		return
	}
	var wg sync.WaitGroup
	wg.Add(k)
	for slot := 0; slot < k; slot++ {
		lo, hi := slot*n/k, (slot+1)*n/k
		go func() {
			defer wg.Done()
			fn(slot, lo, hi)
		}()
	}
	wg.Wait()
}

// mapStrided distributes the n independent units of [0, n) across
// k := parallelism(n) workers by stride - worker w handles
// i = w, w+k, w+2k, ... - and calls fn once per unit, concurrently for
// k > 1. For k <= 1 it runs fn(0), fn(1), ..., fn(n-1) inline and
// spawns no goroutines.
//
// The same NumCPU bound and disjoint-index rule as mapContiguous apply
// (see parallelism): the helper allocates nothing that scales with n,
// and fn must only touch unit i's own state. Striding (rather than
// contiguous ranges) is for independent fixed-size units - rotation
// groups, scale blocks - where it keeps each worker's units evenly
// spread across the whole range.
func mapStrided(n int, fn func(i int)) {
	k := parallelism(n)
	if k <= 1 {
		for i := 0; i < n; i++ {
			fn(i)
		}
		return
	}
	var wg sync.WaitGroup
	wg.Add(k)
	for w := 0; w < k; w++ {
		go func() {
			defer wg.Done()
			for i := w; i < n; i += k {
				fn(i)
			}
		}()
	}
	wg.Wait()
}
