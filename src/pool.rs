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

pub struct Pool {
    tx: Sender<Msg>,
    pending: Arc<(Mutex<usize>, Condvar)>,
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
        let pending = Arc::new((Mutex::new(0usize), Condvar::new()));
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
                        let (lock, cv) = &*p2;
                        let mut g = lock.lock().unwrap();
                        *g -= 1;
                        if *g == 0 {
                            cv.notify_all();
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
            let (lock, _) = &*self.pending;
            *lock.lock().unwrap() += n;
        }
        for j in jobs {
            let _ = self.tx.send(Msg::Run(j));
        }
        let (lock, cv) = &*self.pending;
        let mut g = lock.lock().unwrap();
        while *g != 0 {
            g = cv.wait(g).unwrap();
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
#[allow(dead_code)]
pub struct SPtrU8(pub *const u8);
unsafe impl Send for SPtrU8 {}
