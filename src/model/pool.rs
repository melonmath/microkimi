// Persistent thread pool (std only): avoids the cost of
// std::thread::scope per matvec (~1090 matvecs/token → spawns dominated).
// Jobs carry raw pointers; their validity is guaranteed by
// waiting for completion (barrier) before `run` returns.

use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Condvar, Mutex};

pub type Job = Box<dyn FnOnce() + Send>;

enum Msg {
    Run(Job),
}

/// Cache-line-padded wrapper (128 B): keeps a hot shared counter on its own
/// cache line. The completion counter below is locked and decremented by
/// every worker at the end of every job while the runner sleeps on the
/// condvar; packed side by side they share a line that ping-pongs between
/// cores on each of the ~1090 matvecs/token (false sharing).
#[repr(align(128))]
struct Padded<T>(T);

/// Completion barrier: `n` jobs still running, `cv` signaled at zero. The
/// padded fields keep the counter and the condvar on separate cache lines.
struct Pending {
    n: Padded<Mutex<usize>>,
    cv: Padded<Condvar>,
}

pub struct Pool {
    tx: Sender<Msg>,
    pending: Arc<Pending>,
    pub workers: usize,
}

static POOL: std::sync::OnceLock<Pool> = std::sync::OnceLock::new();

pub fn pool() -> &'static Pool {
    POOL.get_or_init(|| Pool::new(crate::model::n_threads()))
}

impl Pool {
    fn new(n: usize) -> Pool {
        let (tx, rx) = channel::<Msg>();
        let rx = Arc::new(Mutex::new(rx));
        let pending = Arc::new(Pending { n: Padded(Mutex::new(0usize)), cv: Padded(Condvar::new()) });
        for _ in 0..n {
            let rx = rx.clone();
            let pending = pending.clone();
            std::thread::spawn(move || loop {
                let msg = {
                    let lock = rx.lock().unwrap();
                    lock.recv()
                };
                match msg {
                    Ok(Msg::Run(job)) => {
                        let p2 = pending.clone();
                        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
                        let mut g = p2.n.0.lock().unwrap();
                        *g -= 1;
                        if *g == 0 {
                            p2.cv.0.notify_all();
                        }
                    }
                    Err(_) => break, // channel closed → process shutdown
                }
            });
        }
        Pool { tx, pending, workers: n }
    }

    /// Runs `jobs` in parallel and waits for their completion (barrier).
    pub fn run(&self, jobs: Vec<Job>) {
        if jobs.len() == 1 {
            let job = jobs.into_iter().next().unwrap();
            job();
            return;
        }
        let n = jobs.len();
        {
            *self.pending.n.0.lock().unwrap() += n;
        }
        for j in jobs {
            let _ = self.tx.send(Msg::Run(j));
        }
        let mut g = self.pending.n.0.lock().unwrap();
        while *g != 0 {
            g = self.pending.cv.0.wait(g).unwrap();
        }
    }
}

/// Pointer shareable between threads (validity guaranteed by run's barrier).
#[derive(Clone, Copy)]
pub struct SPtr(pub *const f32);
unsafe impl Send for SPtr {}
#[derive(Clone, Copy)]
pub struct MPtr(pub *mut f32);
unsafe impl Send for MPtr {}
#[derive(Clone, Copy)]
pub struct MPtrU8(pub *mut u8);
unsafe impl Send for MPtrU8 {}
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct SPtrU8(pub *const u8);
unsafe impl Send for SPtrU8 {}
