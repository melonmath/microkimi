// VQ1: shared-codebook vector quantization ("microquant") for cold MoE
// experts. Weights are grouped in vectors of VQ_DIM consecutive row-major
// values; each vector maps to the nearest of VQ_K codebook entries (squared
// L2, raw unnormalized vectors). Storage = 1 byte index per vector = 0.5
// bit/weight, plus ONE global 256 x 16 f32 codebook (16 KB) per model file
// (tensor "vq_codebook"), shared by every VQ1 tensor. A per-layer or
// per-tensor codebook was rejected: at nanokimi expert dims (128x64 = 512
// index bytes per tensor) even a per-layer codebook would cost more than the
// indices it serves, and a single global codebook measured good enough
// (see the microquant report).
//
// The codebook is trained by `microkimi slice --cold-vq` (slice.rs) with the
// Lloyd k-means below, over a seeded reservoir sample of the dequantized
// cold-expert values. Everything is deterministic: splitmix64 RNG, fixed
// iteration count, ties broken by lowest index.

pub const VQ_DIM: usize = 16;
pub const VQ_K: usize = 256;
/// Lloyd iterations: enough to converge on 500k samples (inertia plateaus
/// well before), cheap enough that slicing stays I/O bound.
pub const VQ_ITERS: usize = 30;

/// splitmix64: tiny deterministic RNG (zero dependencies).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform index in 0..n.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Squared L2 distance between two VQ_DIM vectors.
#[inline]
fn dist2(a: &[f32], b: &[f32]) -> f32 {
    let mut d = 0f32;
    for j in 0..VQ_DIM {
        let e = a[j] - b[j];
        d += e * e;
    }
    d
}

/// Nearest codebook entry (squared L2, ties -> lowest index).
#[inline]
pub fn nearest(v: &[f32], codebook: &[f32]) -> u8 {
    debug_assert_eq!(codebook.len(), VQ_K * VQ_DIM);
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for k in 0..VQ_K {
        let d = dist2(v, &codebook[k * VQ_DIM..(k + 1) * VQ_DIM]);
        if d < best_d {
            best_d = d;
            best = k;
        }
    }
    best as u8
}

/// Lloyd k-means over 16-vectors. `samples` = n * VQ_DIM raw values
/// (row-major vectors). Init: VQ_K distinct seeded random sample vectors.
/// Empty clusters are re-seeded on the sample vector with the largest current
/// quantization error (deterministic, farthest-point style). Returns the
/// codebook (VQ_K * VQ_DIM f32).
pub fn train_codebook(samples: &[f32], seed: u64) -> Vec<f32> {
    assert!(samples.len() % VQ_DIM == 0 && !samples.is_empty());
    let n = samples.len() / VQ_DIM;
    let vec = |i: usize| &samples[i * VQ_DIM..(i + 1) * VQ_DIM];
    let mut rng = Rng::new(seed);
    // distinct random init
    let mut chosen: Vec<usize> = Vec::with_capacity(VQ_K);
    while chosen.len() < VQ_K.min(n) {
        let i = rng.below(n);
        if !chosen.contains(&i) {
            chosen.push(i);
        }
    }
    let mut cb = vec![0f32; VQ_K * VQ_DIM];
    for (k, &i) in chosen.iter().enumerate() {
        cb[k * VQ_DIM..(k + 1) * VQ_DIM].copy_from_slice(vec(i));
    }
    let mut assign = vec![0u8; n];
    for iter in 0..VQ_ITERS {
        // assignment
        let mut err = vec![0f32; n];
        for i in 0..n {
            let k = nearest(vec(i), &cb);
            assign[i] = k;
            err[i] = dist2(vec(i), &cb[k as usize * VQ_DIM..(k as usize + 1) * VQ_DIM]);
        }
        // update (f64 sums for stable means)
        let mut sums = vec![0f64; VQ_K * VQ_DIM];
        let mut counts = vec![0u64; VQ_K];
        for i in 0..n {
            let k = assign[i] as usize;
            counts[k] += 1;
            for j in 0..VQ_DIM {
                sums[k * VQ_DIM + j] += vec(i)[j] as f64;
            }
        }
        let mut moved = 0u64;
        for k in 0..VQ_K {
            if counts[k] > 0 {
                for j in 0..VQ_DIM {
                    let mean = (sums[k * VQ_DIM + j] / counts[k] as f64) as f32;
                    if mean != cb[k * VQ_DIM + j] {
                        moved += 1;
                    }
                    cb[k * VQ_DIM + j] = mean;
                }
            } else {
                // empty cluster: re-seed on the worst-quantized sample
                let worst = err
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                cb[k * VQ_DIM..(k + 1) * VQ_DIM].copy_from_slice(vec(worst));
                err[worst] = 0.0; // do not re-seed every empty cluster on the same vector
            }
        }
        if moved == 0 && iter > 0 {
            break; // converged
        }
    }
    cb
}

/// Quantizes a [rows, cols] matrix (cols % VQ_DIM == 0) into VQ1 index bytes:
/// rows * cols / VQ_DIM u8, one index per vector of VQ_DIM consecutive
/// row-major values.
pub fn quantize(w: &[f32], codebook: &[f32]) -> Vec<u8> {
    assert!(w.len() % VQ_DIM == 0);
    w.chunks_exact(VQ_DIM).map(|v| nearest(v, codebook)).collect()
}

/// Relative Frobenius error ||w - wq|| / ||w|| of a VQ1 quantization
/// (diagnostic printed by the slicer).
pub fn rel_error(w: &[f32], indices: &[u8], codebook: &[f32]) -> f64 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (vi, v) in w.chunks_exact(VQ_DIM).enumerate() {
        let cb = &codebook[indices[vi] as usize * VQ_DIM..(indices[vi] as usize + 1) * VQ_DIM];
        for j in 0..VQ_DIM {
            let e = v[j] as f64 - cb[j] as f64;
            num += e * e;
            den += v[j] as f64 * v[j] as f64;
        }
    }
    if den == 0.0 {
        0.0
    } else {
        (num / den).sqrt()
    }
}

/// VQ1 matvec: out[r] = sum_c W[r,c] * x[c] with W gathered from the
/// codebook on the fly. The 16 KB codebook stays hot in L1; the index row
/// is streamed once. Per row the accumulation is vector-sequential (dot of
/// each 16-vector in order), deterministic.
pub fn matvec_vq(codebook: &[f32], indices: &[u8], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(cols % VQ_DIM, 0);
    debug_assert_eq!(out.len(), rows);
    let vpr = cols / VQ_DIM; // vectors per row
    for (r, o) in out.iter_mut().enumerate() {
        let irow = &indices[r * vpr..(r + 1) * vpr];
        let mut sum = 0f32;
        for (v, &idx) in irow.iter().enumerate() {
            let cb = &codebook[idx as usize * VQ_DIM..(idx as usize + 1) * VQ_DIM];
            let xs = &x[v * VQ_DIM..(v + 1) * VQ_DIM];
            let mut s = 0f32;
            for j in 0..VQ_DIM {
                s += cb[j] * xs[j];
            }
            sum += s;
        }
        *o = sum;
    }
}

/// Batched VQ1 matvec for prefill, same contract as model::matvec_packed_nt:
/// `xt` = m inputs transposed [cols * m] (m a multiple of 8, zero-padded
/// lanes ignored by the caller), `out` position-major [m * rows]. The index
/// row and the codebook entries are read once per element for the whole
/// block of positions.
pub fn matvec_vq_nt(codebook: &[f32], indices: &[u8], rows: usize, cols: usize, xt: &[f32], m: usize, out: &mut [f32]) {
    debug_assert_eq!(cols % VQ_DIM, 0);
    debug_assert_eq!(m % 8, 0);
    let vpr = cols / VQ_DIM;
    for t0 in (0..m).step_by(8) {
        for r in 0..rows {
            let irow = &indices[r * vpr..(r + 1) * vpr];
            let mut sum = [0f32; 8];
            for (v, &idx) in irow.iter().enumerate() {
                let cb = &codebook[idx as usize * VQ_DIM..(idx as usize + 1) * VQ_DIM];
                for j in 0..VQ_DIM {
                    let wv = cb[j];
                    let xc = &xt[(v * VQ_DIM + j) * m + t0..(v * VQ_DIM + j) * m + t0 + 8];
                    for p in 0..8 {
                        sum[p] += wv * xc[p];
                    }
                }
            }
            for p in 0..8 {
                out[(t0 + p) * rows + r] = sum[p];
            }
        }
    }
}
