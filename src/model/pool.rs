// Persistent thread pool (std only): avoids the cost of
// std::thread::scope per matvec (~1090 matvecs/token → spawns dominated).
// Jobs carry raw pointers; their validity is guaranteed by
// waiting for completion (barrier) before `run` returns.
//
// Dispatch is a SPINNING job board, not a channel: the decode issues a
// batch every ~100 microseconds, and the old mpsc + condvar handshake
// cost tens of microseconds of wakeup latency per batch (a flat ~3
// ms/token of sync at ~100 barriers/token). The whole board is ONE
// AtomicU64 word - (epoch 32 | len 16 | next 16) - so a claim is a
// CAS tagged with the batch epoch: a straggler from a finished batch
// can neither consume a ticket of the next batch nor dereference the
// job vector after its batch completed (indices at or past `len`
// break without touching `jobs`, and `jobs` is only rewritten while
// pending == 0, when no in-range claim can exist). Workers spin
// briefly between batches and park after ~a millisecond without work;
// every publish unparks them, so during generation a barrier costs a
// handful of atomics.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

pub type Job = Box<dyn FnOnce() + Send>;

std::thread_local! {
    static IN_POOL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// True on a pool worker thread. Kernels that may be called both from
/// the main thread and from inside pool jobs (MoE experts) use this to
/// pick the persistent pool when it is safe and scoped threads when it
/// is not (the pool's completion barrier does not nest).
pub fn in_pool_worker() -> bool {
    IN_POOL.with(|c| c.get())
}

/// Idle spins before a worker parks (MICROKIMI_SPIN, default 6000
/// ~= 0.5-1 ms). Low values trade barrier wakeup latency for freeing
/// the cores during non-pool compute sections.
fn spin_limit() -> u32 {
    static N: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("MICROKIMI_SPIN").ok().and_then(|v| v.parse().ok()).unwrap_or(6_000)
    })
}

/// Cache-line-padded wrapper (128 B): keeps each hot shared counter on
/// its own cache line - packed together they would ping-pong one line
/// between every core on every claim and every completion.
#[repr(align(128))]
struct Padded<T>(T);

const TAG_SHIFT: u32 = 32;
const LEN_SHIFT: u32 = 16;
const LEN_MASK: u64 = 0xFFFF;
const NEXT_MASK: u64 = 0xFFFF;

struct Board {
    /// (epoch 32 | len 16 | next 16): the entire dispatch state.
    word: Padded<AtomicU64>,
    /// Jobs of the current batch not yet finished.
    pending: Padded<AtomicUsize>,
    /// Number of parked workers (fast should-unpark check).
    parked: Padded<AtomicUsize>,
    /// The published batch. Written ONLY by the single runner thread and
    /// ONLY while pending == 0 (no in-range claim exists then, so no
    /// worker dereferences it); the Release store of `word` publishes it.
    jobs: std::cell::UnsafeCell<Vec<Job>>,
    /// Worker thread handles for unparking.
    handles: Mutex<Vec<std::thread::Thread>>,
    /// Serializes concurrent runners (the engine has one compute thread,
    /// but the test harness runs models on several threads at once).
    runner: Mutex<()>,
}

// SAFETY: see the field comments - `jobs` mutation is confined to the
// single runner thread during pending == 0 windows, reads are gated by
// epoch-tagged CAS claims ordered through `word`.
unsafe impl Send for Board {}
unsafe impl Sync for Board {}

pub struct Pool {
    board: Arc<Board>,
    pub workers: usize,
}

static POOL: std::sync::OnceLock<Pool> = std::sync::OnceLock::new();

pub fn pool() -> &'static Pool {
    POOL.get_or_init(|| Pool::new(crate::model::n_threads()))
}

/// One logical CPU per physical core, in core order, restricted to the
/// CPUs this process may run on. Some(list) only on Linux hosts where
/// SMT is present (some core lists two hardware threads); None
/// elsewhere, so callers keep their previous behavior. Read from sysfs
/// (`thread_siblings_list`) and sched_getaffinity - no dependencies.
pub fn physical_cpus() -> Option<Vec<usize>> {
    #[cfg(target_os = "linux")]
    {
        static CORES: std::sync::OnceLock<Option<Vec<usize>>> = std::sync::OnceLock::new();
        return CORES
            .get_or_init(|| {
                let allowed = affinity::current_set()?;
                let mut seen = std::collections::HashSet::new();
                let mut cores = Vec::new();
                let mut smt = false;
                for &cpu in &allowed {
                    let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list");
                    let list = std::fs::read_to_string(&path).ok()?;
                    let sibs: Vec<usize> = list
                        .trim()
                        .split(',')
                        .flat_map(|part| match part.split_once('-') {
                            Some((a, b)) => {
                                let (a, b) = (a.parse::<usize>().ok(), b.parse::<usize>().ok());
                                match (a, b) {
                                    (Some(a), Some(b)) => (a..=b).collect::<Vec<_>>(),
                                    _ => Vec::new(),
                                }
                            }
                            None => part.parse::<usize>().ok().into_iter().collect(),
                        })
                        .collect();
                    if sibs.len() > 1 {
                        smt = true;
                    }
                    let key = *sibs.iter().min()?;
                    if seen.insert(key) {
                        cores.push(cpu);
                    }
                }
                if smt && !cores.is_empty() {
                    Some(cores)
                } else {
                    None
                }
            })
            .clone();
    }
    #[cfg(not(target_os = "linux"))]
    None
}

/// Linux CPU affinity through raw libc calls (sched_getaffinity /
/// sched_setaffinity on a cpu_set_t of 1024 bits, the glibc default).
#[cfg(target_os = "linux")]
mod affinity {
    const SET_WORDS: usize = 1024 / 64;
    unsafe extern "C" {
        fn sched_getaffinity(pid: i32, cpusetsize: usize, mask: *mut u64) -> i32;
        fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u64) -> i32;
    }
    /// CPUs the calling thread may run on.
    pub fn current_set() -> Option<Vec<usize>> {
        let mut mask = [0u64; SET_WORDS];
        // SAFETY: mask is SET_WORDS * 8 bytes, as declared.
        let rc = unsafe { sched_getaffinity(0, SET_WORDS * 8, mask.as_mut_ptr()) };
        if rc != 0 {
            return None;
        }
        let mut cpus = Vec::new();
        for (w, word) in mask.iter().enumerate() {
            for b in 0..64 {
                if word & (1u64 << b) != 0 {
                    cpus.push(w * 64 + b);
                }
            }
        }
        Some(cpus)
    }
    /// Pins the calling thread to one CPU (best effort).
    pub fn pin_current(cpu: usize) {
        if cpu >= 1024 {
            return;
        }
        let mut mask = [0u64; SET_WORDS];
        mask[cpu / 64] |= 1u64 << (cpu % 64);
        // SAFETY: mask is SET_WORDS * 8 bytes, as declared.
        unsafe {
            sched_setaffinity(0, SET_WORDS * 8, mask.as_ptr());
        }
    }
}

/// The pinning plan for `n` participants (the calling thread runs jobs
/// too): with SMT and n <= physical cores, the caller sits on core 0
/// and worker i on core i + 1 - every participant its own core, no
/// sibling sharing; otherwise no pinning. MICROKIMI_NO_PIN=1 disables it.
fn pin_plan(n: usize) -> Option<Vec<usize>> {
    if std::env::var("MICROKIMI_NO_PIN").map(|v| v == "1").unwrap_or(false) {
        return None;
    }
    let cores = physical_cpus()?;
    if n <= cores.len() {
        Some(cores)
    } else {
        None
    }
}

impl Pool {
    fn new(n: usize) -> Pool {
        let plan = pin_plan(n);
        // pinned: the caller is one of the n participants, so n - 1
        // workers; unpinned: n workers as before (the caller's share of
        // the tickets is whatever it claims)
        let spawn = if plan.is_some() { n.saturating_sub(1) } else { n };
        #[cfg(target_os = "linux")]
        if let Some(cores) = &plan {
            affinity::pin_current(cores[0]);
        }
        let board = Arc::new(Board {
            word: Padded(AtomicU64::new(0)),
            pending: Padded(AtomicUsize::new(0)),
            parked: Padded(AtomicUsize::new(0)),
            jobs: std::cell::UnsafeCell::new(Vec::new()),
            handles: Mutex::new(Vec::new()),
            runner: Mutex::new(()),
        });
        for i in 0..spawn {
            let b = board.clone();
            let pin = plan.as_ref().map(|cores| cores[(i + 1) % cores.len()]);
            let h = std::thread::spawn(move || {
                IN_POOL.with(|c| c.set(true));
                #[cfg(target_os = "linux")]
                if let Some(cpu) = pin {
                    affinity::pin_current(cpu);
                }
                #[cfg(not(target_os = "linux"))]
                let _ = pin;
                let mut seen: u64 = 0; // tag of the last batch we worked or skipped
                let mut idle: u32 = 0;
                loop {
                    let mut cur = b.word.0.load(Ordering::Acquire);
                    if (cur >> TAG_SHIFT) != seen {
                        // new batch: claim tickets while they exist
                        idle = 0;
                        let tag = cur >> TAG_SHIFT;
                        loop {
                            if (cur >> TAG_SHIFT) != tag {
                                break; // an even newer batch; outer loop re-enters
                            }
                            let len = (cur >> LEN_SHIFT) & LEN_MASK;
                            let idx = cur & NEXT_MASK;
                            if idx >= len {
                                break;
                            }
                            match b.word.0.compare_exchange_weak(
                                cur,
                                cur + 1,
                                Ordering::AcqRel,
                                Ordering::Acquire,
                            ) {
                                Ok(_) => {
                                    // SAFETY: the claim is tagged with the live
                                    // epoch and in range, so `jobs` is the batch
                                    // this ticket belongs to and index `idx` is
                                    // claimed exactly once; FnOnce moves out via
                                    // ptr::read and the shell is never dropped
                                    // (the runner set_len(0)s before reuse).
                                    let job: Job = unsafe {
                                        let jobs = &*b.jobs.get();
                                        std::ptr::read(&jobs[idx as usize] as *const Job)
                                    };
                                    let _ = std::panic::catch_unwind(
                                        std::panic::AssertUnwindSafe(job),
                                    );
                                    b.pending.0.fetch_sub(1, Ordering::Release);
                                    cur = b.word.0.load(Ordering::Acquire);
                                }
                                Err(now) => cur = now,
                            }
                        }
                        seen = tag;
                    } else {
                        idle += 1;
                        if idle < spin_limit() {
                            if idle & 0x3F == 0 {
                                std::thread::yield_now();
                            } else {
                                std::hint::spin_loop();
                            }
                        } else {
                            b.parked.0.fetch_add(1, Ordering::SeqCst);
                            // re-check after registering: a publish between
                            // the load above and here must not be slept
                            // through (unpark permits are sticky anyway)
                            if (b.word.0.load(Ordering::Acquire) >> TAG_SHIFT) == seen {
                                std::thread::park();
                            }
                            b.parked.0.fetch_sub(1, Ordering::SeqCst);
                            idle = 0;
                        }
                    }
                }
            });
            board.handles.lock().unwrap().push(h.thread().clone());
        }
        Pool { board, workers: n }
    }

    /// Runs `jobs` in parallel and waits for their completion (barrier).
    /// Must be called from one thread at a time (the engine's single
    /// compute thread) and never from inside a pool job.
    pub fn run(&self, jobs: Vec<Job>) {
        if jobs.len() == 1 {
            let job = jobs.into_iter().next().unwrap();
            job();
            return;
        }
        let n = jobs.len();
        assert!(n <= LEN_MASK as usize, "job batch too large for the board");
        debug_assert!(!in_pool_worker(), "pool.run() must not nest inside a pool job");
        let b = &self.board;
        let _serial = b.runner.lock().unwrap();
        // SAFETY: the previous barrier left pending == 0, so no worker
        // holds an in-range claim and none dereferences `jobs`. The old
        // shells were moved out by ptr::read - set_len(0) forgets them
        // without dropping. The Release store of `word` below publishes
        // the new vector to claimers.
        unsafe {
            let slot = &mut *b.jobs.get();
            slot.set_len(0);
            *slot = jobs;
        }
        b.pending.0.store(n, Ordering::Relaxed);
        let old = b.word.0.load(Ordering::Relaxed);
        let tag = (old >> TAG_SHIFT).wrapping_add(1) & 0xFFFF_FFFF;
        b.word.0.store((tag << TAG_SHIFT) | ((n as u64) << LEN_SHIFT), Ordering::Release);
        if b.parked.0.load(Ordering::SeqCst) > 0 {
            for t in b.handles.lock().unwrap().iter() {
                t.unpark();
            }
        }
        // the runner claims tickets too: with W workers and W+? jobs it
        // would otherwise idle-spin while contributing nothing
        let mut cur = b.word.0.load(Ordering::Acquire);
        while (cur >> TAG_SHIFT) == tag {
            let len = (cur >> LEN_SHIFT) & LEN_MASK;
            let idx = cur & NEXT_MASK;
            if idx >= len {
                break;
            }
            match b.word.0.compare_exchange_weak(cur, cur + 1, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => {
                    // SAFETY: same claim contract as the workers.
                    let job: Job = unsafe {
                        let jobs = &*b.jobs.get();
                        std::ptr::read(&jobs[idx as usize] as *const Job)
                    };
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
                    b.pending.0.fetch_sub(1, Ordering::Release);
                    cur = b.word.0.load(Ordering::Acquire);
                }
                Err(now) => cur = now,
            }
        }
        let mut spins = 0u32;
        while b.pending.0.load(Ordering::Acquire) != 0 {
            spins += 1;
            if spins & 0xFF == 0 {
                std::thread::yield_now();
            } else {
                std::hint::spin_loop();
            }
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
