//! Ordered parallel map over independent work units (groups, frames).
//!
//! The worker budget is always the caller's (`EncodeOptions::threads`,
//! or the framework's `ExecutionContext` through the registry): this
//! module never asks the host how many cores it has. Results come back
//! in input order whatever the budget, so everything built from them
//! is byte-identical at every worker count.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// `f(index, item)` for every item, on up to `threads` scoped workers;
/// the results in input order. One worker (or one item) runs inline.
pub(crate) fn par_map<T, R, F>(threads: usize, items: Vec<T>, f: F) -> Vec<R>
where
    T: Send,
    R: Send,
    F: Fn(usize, T) -> R + Sync,
{
    let n = items.len();
    let workers = threads.min(n).max(1);
    if workers == 1 {
        return items
            .into_iter()
            .enumerate()
            .map(|(i, t)| f(i, t))
            .collect();
    }
    let inputs: Vec<Mutex<Option<T>>> = items.into_iter().map(|t| Mutex::new(Some(t))).collect();
    let outputs: Vec<Mutex<Option<R>>> = (0..n).map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    // Each slot is taken exactly once; a poisoned lock
                    // only means another worker panicked, which the
                    // scope re-raises when it joins.
                    let item = inputs[i].lock().ok().and_then(|mut s| s.take());
                    if let Some(item) = item {
                        let r = f(i, item);
                        if let Ok(mut slot) = outputs[i].lock() {
                            *slot = Some(r);
                        }
                    }
                }
            });
        }
    });
    outputs
        .into_iter()
        .map(|m| {
            m.into_inner()
                .ok()
                .flatten()
                .expect("every work unit produced a result")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn results_keep_input_order_at_every_budget() {
        let items: Vec<u32> = (0..1000).collect();
        let want: Vec<u64> = items.iter().map(|&v| u64::from(v) * 3 + 1).collect();
        for threads in [0usize, 1, 2, 3, 8, 64, 5000] {
            let got = par_map(threads, items.clone(), |i, v| {
                assert_eq!(i as u32, v);
                u64::from(v) * 3 + 1
            });
            assert_eq!(got, want, "{threads} threads");
        }
        assert!(par_map(8, Vec::<u8>::new(), |_, v| v).is_empty());
    }
}
