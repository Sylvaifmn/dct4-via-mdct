//! Buffer pooling with `K` slots (§6).
//!
//! A block holds **two** slots for its whole traversal: the folded `2N` word while the backend
//! reads it, and the `N`-word output while POST consumes it. The accounting is deliberately
//! conservative — both slots are reserved from the start of `FOLD` to the end of `POST` — so
//! the bound `capacity = K / 2` is a hard one, not an average.
//!
//! The pool tracks *logical* time: [`BufferPool::earliest_start`] answers "when could a block
//! begin if it must not start before `t`?", and [`BufferPool::occupy`] books the slot
//! lifetime. That is exactly the information both the event-driven simulation and the
//! closed-form projection need, and it keeps the pool free of threads, `unsafe`, or GPU APIs.

/// Cumulative statistics of a [`BufferPool`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PoolStats {
    /// Number of successful reservations.
    pub reservations: u64,
    /// Number of reservations that had to wait for slots.
    pub wait_events: u64,
    /// Longest single wait, in model time units.
    pub max_wait: f64,
    /// Maximum number of blocks simultaneously in flight.
    pub peak_in_flight: usize,
}

/// A pool of `capacity` slots, `per_block` of which are needed by one in-flight block.
#[derive(Debug, Clone)]
pub struct BufferPool {
    capacity: usize,
    per_block: usize,
    /// Time at which each slot becomes free.
    free_at: Vec<f64>,
    stats: PoolStats,
}

impl BufferPool {
    /// Builds a pool.
    ///
    /// # Panics
    ///
    /// Panics if `per_block == 0` or `capacity < per_block`: with `K < 2` slots no block can
    /// ever be in flight, which is a configuration error, not a runtime condition.
    pub fn new(capacity: usize, per_block: usize) -> Self {
        assert!(per_block >= 1, "per_block must be >= 1");
        assert!(
            capacity >= per_block,
            "pool capacity {capacity} cannot hold one block needing {per_block} slots"
        );
        Self {
            capacity,
            per_block,
            free_at: vec![0.0; capacity],
            stats: PoolStats::default(),
        }
    }

    /// Total number of slots.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Slots consumed by one in-flight block.
    pub fn per_block(&self) -> usize {
        self.per_block
    }

    /// Maximum number of blocks that can be in flight: `floor(K / per_block)`.
    pub fn max_in_flight(&self) -> usize {
        self.capacity / self.per_block
    }

    /// Number of slots free at time `now`.
    pub fn free_slots(&self, now: f64) -> usize {
        self.free_at.iter().filter(|&&t| t <= now + 1e-12).count()
    }

    /// Earliest time at which a new block may start, given `not_before`.
    ///
    /// `None` only if the pool is degenerate (`K < per_block`), which the constructor forbids.
    pub fn earliest_start(&self, not_before: f64) -> Option<f64> {
        if self.max_in_flight() == 0 {
            return None;
        }
        let mut v = self.free_at.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Some(v[self.per_block - 1].max(not_before))
    }

    /// Books `per_block` slots for a block that starts at `start` and releases at `release_at`.
    ///
    /// Returns `false` (and changes nothing) if `start` is earlier than the earliest feasible
    /// start, which can only happen if the caller ignores [`BufferPool::earliest_start`].
    pub fn occupy(&mut self, not_before: f64, start: f64, release_at: f64) -> bool {
        let Some(earliest) = self.earliest_start(not_before) else {
            return false;
        };
        if start + 1e-12 < earliest || release_at + 1e-12 < start {
            return false;
        }

        // Track occupancy before mutating: this is the number of blocks already in flight.
        let free_before = self.free_slots(start);
        let in_flight_after = (self.capacity - free_before) / self.per_block + 1;
        self.stats.peak_in_flight = self.stats.peak_in_flight.max(in_flight_after);

        let mut idx: Vec<usize> = (0..self.capacity).collect();
        idx.sort_by(|&a, &b| {
            self.free_at[a]
                .partial_cmp(&self.free_at[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for &i in &idx[..self.per_block] {
            self.free_at[i] = release_at;
        }

        self.stats.reservations += 1;
        let wait = start - not_before;
        if wait > 1e-12 {
            self.stats.wait_events += 1;
            self.stats.max_wait = self.stats.max_wait.max(wait);
        }
        true
    }

    /// Cumulative statistics.
    pub fn stats(&self) -> PoolStats {
        self.stats
    }

    /// Per-slot release times, for inspection.
    pub fn slot_free_times(&self) -> &[f64] {
        &self.free_at
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_bound() {
        let p = BufferPool::new(6, 2);
        assert_eq!(p.max_in_flight(), 3);
        let p = BufferPool::new(7, 2);
        assert_eq!(
            p.max_in_flight(),
            3,
            "7 slots still host only 3 two-slot blocks"
        );
        assert_eq!(p.capacity(), 7);
    }

    #[test]
    fn first_block_starts_immediately() {
        let p = BufferPool::new(4, 2);
        assert_eq!(p.earliest_start(0.0), Some(0.0));
        assert_eq!(p.earliest_start(5.0), Some(5.0));
    }

    #[test]
    fn third_block_waits_for_the_first_release() {
        let mut p = BufferPool::new(4, 2);
        assert_eq!(p.max_in_flight(), 2);
        assert!(p.occupy(0.0, 0.0, 10.0));
        assert!(p.occupy(0.0, 0.0, 12.0));
        // Both in-flight pairs are busy until t=10, so a third block cannot start earlier.
        assert_eq!(p.earliest_start(0.0), Some(10.0));
        assert!(
            !p.occupy(0.0, 9.0, 20.0),
            "a start earlier than the earliest feasible one must be rejected"
        );
        let s = p.stats();
        assert_eq!(s.reservations, 2);
        assert_eq!(s.wait_events, 0);
        assert_eq!(s.peak_in_flight, 2);
    }

    #[test]
    fn waiting_is_recorded() {
        let mut p = BufferPool::new(2, 2);
        assert!(p.occupy(0.0, 0.0, 10.0));
        let earliest = p.earliest_start(5.0).unwrap();
        assert_eq!(earliest, 10.0);
        assert!(p.occupy(5.0, earliest, 20.0));
        let s = p.stats();
        assert_eq!(s.wait_events, 1);
        assert!((s.max_wait - 5.0).abs() < 1e-12);
    }

    #[test]
    fn free_slots_tracking() {
        let mut p = BufferPool::new(4, 2);
        assert_eq!(p.free_slots(0.0), 4);
        assert!(p.occupy(0.0, 0.0, 3.0));
        assert_eq!(p.free_slots(0.0), 2);
        assert_eq!(p.free_slots(3.0), 4);
        assert_eq!(p.slot_free_times()[0], 3.0);
    }
}
