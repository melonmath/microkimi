// Expert MERGING for `microkimi slice --merge-experts N` (moved from slice.rs):
// instead of DELETING the tail experts of every MoE layer, the N output
// experts are clusters of the old ones: experts close in weight space are
// averaged into one, so their knowledge survives in the merged weights
// instead of being dropped. Fully training-free.
//
// Per kept MoE layer:
//   1. every old expert is the concatenation of its w1/w2/w3 (mxfp4
//      dequantized). The k-means distance is computed on a seeded coordinate
//      subset (sketch) of that vector - the full vector when it is small
//      enough - so a whole layer of dequantized experts never sits in RAM.
//      The subset is the same for every expert of the layer and distances on
//      it are a uniformly scaled estimate of the full ones; the scaling is a
//      per-layer constant that leaves every k-means decision (nearest
//      centroid, seeding probabilities) unchanged.
//   2. deterministic weighted k-means (k-means++ seeding, xorshift64* with a
//      fixed seed, MERGE_ITERS Lloyd iterations). The usage score of an
//      expert (the persistent slice score cache, score.rs) weights the
//      seeding probabilities, the centroid updates (weighted mean) and the
//      empty-cluster reseeding; without a cache every expert weighs the
//      same. (Scaling one expert's distance by its own weight cannot change
//      its argmin assignment, so the weight acts where it matters: hot
//      experts pull a centroid onto themselves and stay near-singletons,
//      cold experts merge into each other.)
//   3. merged expert = weighted average of the member experts (weights =
//      usage scores, uniform without the cache), requantized to mxfp4 with
//      the searched scale (mxfp4::quantize).
//   4. router rows conserve routing mass: new_logit = log(sum_i
//      exp(old_logit_i)), a numerically stable logsumexp over the cluster
//      members, so the merged expert inherits the total routing probability
//      of its members. The e_score_correction_bias merges the same way.
//      Shared experts are separate tensors and are never touched.

use super::ckpt::SliceCkpt;
use super::score::ScoreCache;
use super::source::Source;
use crate::quant::weights::{DTYPE_MXFP4, DTYPE_MXFP4SQ};
use std::collections::HashMap;

/// Lloyd iteration cap for the expert k-means (assignments almost always
/// settle earlier; the cap only bounds the worst case).
const MERGE_ITERS: usize = 25;

/// Clustering sketch width: squared distances are computed on a seeded
/// subset of at most this many coordinates of the concatenated w1|w2|w3
/// vector. At 384 experts x ~44 M weights an exact 25-iteration k-means
/// would cost ~1e14 flops per layer; the sketch keeps it at ~1e10. Vectors
/// at or below SKETCH_DIM are used in full (exact k-means).
const SKETCH_DIM: usize = 16384;

/// Fixed k-means seed (xorshift64* state), mixed with the layer index.
const MERGE_SEED: u64 = 0x6A09_E667_F3BC_C909;

/// xorshift64*: tiny deterministic RNG (zero dependencies), fixed-seeded so
/// a merge rerun on the same model produces the same clusters.
struct XorShift(u64);

impl XorShift {
    fn new(seed: u64) -> XorShift {
        XorShift(seed | 1) // the zero state is absorbing
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform f64 in [0, 1).
    fn f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Seeded coordinate subset of a dim-wide vector: the identity when dim fits
/// the sketch, else SKETCH_DIM distinct positions, sorted for sequential
/// blob access. Deterministic in (dim, seed).
fn sketch_positions(dim: usize, seed: u64) -> Vec<usize> {
    if dim <= SKETCH_DIM {
        return (0..dim).collect();
    }
    let mut rng = XorShift::new(seed ^ 0xA5A5_A5A5_5A5A_5A5A);
    let mut set = std::collections::HashSet::with_capacity(SKETCH_DIM);
    while set.len() < SKETCH_DIM {
        set.insert((rng.next() % dim as u64) as usize);
    }
    let mut pos: Vec<usize> = set.into_iter().collect();
    pos.sort_unstable();
    pos
}

/// Values of an mxfp4 blob (either flavor) at flat row-major positions
/// (same layout as mxfp4::dequant, but gathering only the needed elements).
fn blob_values_at(dtype: u8, blob: &[u8], rows: usize, cols: usize, pos: &[usize], out: &mut [f32]) {
    assert_eq!(pos.len(), out.len());
    let np = rows * cols / 2;
    let packed = &blob[..np];
    let scales = &blob[np..np + rows * cols / 32];
    let smax = if dtype == DTYPE_MXFP4SQ {
        f32::from_le_bytes(blob[np + rows * cols / 32..np + rows * cols / 32 + 4].try_into().unwrap())
    } else {
        0.0
    };
    for (&q, o) in pos.iter().zip(out) {
        let (r, c) = (q / cols, q % cols);
        let byte = packed[r * cols / 2 + c / 2];
        let nib = if c % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        let sb = scales[r * cols / 32 + c / 32];
        *o = match dtype {
            DTYPE_MXFP4 => crate::quant::mxfp4::E2M1[nib as usize] * crate::quant::mxfp4::exp2_i(sb as i32 - 127),
            DTYPE_MXFP4SQ => crate::quant::mxfp4::E2M1[nib as usize] * crate::quant::mxfp4::scale_sq(sb, smax),
            _ => panic!("dtype {} is not an mxfp4 flavor", dtype),
        };
    }
}

/// Weighted draw over `mass` (proportional probabilities). Degenerate
/// all-zero mass falls back to the first not-yet-chosen index, so k-means++
/// always picks a fresh centroid. Deterministic given the rng stream.
fn draw(rng: &mut XorShift, mass: &[f64], chosen: &[bool]) -> usize {
    let total: f64 = mass.iter().sum();
    if !(total > 0.0) || !total.is_finite() {
        return chosen.iter().position(|&c| !c).unwrap_or(0);
    }
    let mut t = rng.f64() * total;
    for (i, &m) in mass.iter().enumerate() {
        if m > 0.0 {
            t -= m;
            if t < 0.0 {
                return i;
            }
        }
    }
    mass.iter().rposition(|&m| m > 0.0).unwrap_or(0)
}

/// Deterministic weighted k-means over n row-major vectors of width `dim`:
/// k-means++ seeding (probabilities proportional to weight x current
/// squared distance), then Lloyd iterations with weight-scaled centroid
/// means. Ties break to the lowest index everywhere. Returns the cluster id
/// (0..k) of every vector; every cluster is non-empty.
pub(super) fn kmeans_weighted(vecs: &[f32], n: usize, dim: usize, weights: &[f64], k: usize, iters: usize, seed: u64) -> Vec<usize> {
    assert_eq!(vecs.len(), n * dim);
    assert_eq!(weights.len(), n);
    assert!(k >= 1 && k <= n);
    let vec = |i: usize| &vecs[i * dim..(i + 1) * dim];
    let d2c = |i: usize, c: &[f32]| -> f64 {
        vec(i)
            .iter()
            .zip(c)
            .map(|(&a, &b)| {
                let d = a as f64 - b as f64;
                d * d
            })
            .sum()
    };
    let mut rng = XorShift::new(seed);
    // k-means++ seeding
    let mut cent = vec![0f32; k * dim];
    let mut chosen = vec![false; n];
    let mut d2 = vec![f64::INFINITY; n]; // squared distance to the nearest chosen centroid
    for c in 0..k {
        let mass: Vec<f64> = if c == 0 {
            weights.to_vec()
        } else {
            (0..n).map(|i| if chosen[i] { 0.0 } else { weights[i] * d2[i] }).collect()
        };
        let i = draw(&mut rng, &mass, &chosen);
        chosen[i] = true;
        cent[c * dim..(c + 1) * dim].copy_from_slice(vec(i));
        for j in 0..n {
            let d = d2c(j, vec(i));
            if d < d2[j] {
                d2[j] = d;
            }
        }
    }
    // Lloyd iterations
    let mut assign = vec![usize::MAX; n];
    let mut err = vec![0f64; n]; // squared distance to the assigned centroid
    for it in 0..iters {
        let mut changed = false;
        for i in 0..n {
            let mut best = 0;
            let mut bd = f64::INFINITY;
            for c in 0..k {
                let d = d2c(i, &cent[c * dim..(c + 1) * dim]);
                if d < bd {
                    bd = d;
                    best = c;
                }
            }
            err[i] = bd;
            if assign[i] != best {
                assign[i] = best;
                changed = true;
            }
        }
        if !changed && it > 0 {
            break;
        }
        let mut sums = vec![0f64; k * dim];
        let mut wsum = vec![0f64; k];
        for i in 0..n {
            let c = assign[i];
            wsum[c] += weights[i];
            for j in 0..dim {
                sums[c * dim + j] += weights[i] * vec(i)[j] as f64;
            }
        }
        for c in 0..k {
            if wsum[c] > 0.0 {
                for j in 0..dim {
                    cent[c * dim + j] = (sums[c * dim + j] / wsum[c]) as f32;
                }
            } else {
                // empty cluster: re-seed on the expert with the largest
                // weighted error (deterministic, farthest-point style)
                let mut worst = 0;
                let mut wm = f64::NEG_INFINITY;
                for i in 0..n {
                    let m = weights[i] * err[i];
                    if m > wm {
                        wm = m;
                        worst = i;
                    }
                }
                cent[c * dim..(c + 1) * dim].copy_from_slice(vec(worst));
                err[worst] = 0.0; // never re-seed two empty clusters on the same expert
            }
        }
    }
    // a final assignment can still strand a cluster (its centroid was
    // re-seeded onto an expert that then left): hand each empty cluster the
    // highest weighted-error expert whose own cluster keeps > 1 member
    loop {
        let mut counts = vec![0usize; k];
        for &a in &assign {
            counts[a] += 1;
        }
        let Some(c) = counts.iter().position(|&n| n == 0) else { break };
        let mut best = None;
        let mut bm = f64::NEG_INFINITY;
        for i in 0..n {
            if counts[assign[i]] > 1 && weights[i] * err[i] > bm {
                bm = weights[i] * err[i];
                best = Some(i);
            }
        }
        match best {
            Some(i) => assign[i] = c,
            None => break, // k == n and every other cluster is a singleton: impossible, but never loop
        }
    }
    assign
}

/// Cluster id per vector -> member lists (old ids ascending inside each).
pub(super) fn clusters_of(assign: &[usize], k: usize) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new(); k];
    for (i, &a) in assign.iter().enumerate() {
        out[a].push(i);
    }
    out
}

/// Per-expert merge weights: the score-cache scores normalized to sum 1
/// (Frobenius magnitudes double as the usage proxy), uniform without a
/// usable cache entry.
fn layer_weights(sc: &ScoreCache, layer: usize, n: usize) -> Vec<f64> {
    let uniform = || vec![1.0 / n as f64; n];
    match sc.get(layer) {
        Some(s) if s.len() == n => {
            let sum: f64 = s.iter().map(|&x| x.max(0.0)).sum();
            if sum > 0.0 {
                s.iter().map(|&x| x.max(0.0) / sum).collect()
            } else {
                uniform()
            }
        }
        _ => uniform(),
    }
}

/// The merge result of one MoE layer.
pub(super) struct LayerMerge {
    pub(super) clusters: Vec<Vec<usize>>, // new expert id -> old expert ids (ascending)
    pub(super) weights: Vec<f64>,         // merge weight per OLD expert (sum 1)
}

/// Clusters the experts of one MoE layer: sketch-gather the seeded
/// coordinates of every expert's concatenated w1|w2|w3, then the weighted
/// k-means. Returns the cluster id of every old expert.
fn cluster_layer(src: &Source, layer: usize, n: usize, weights: &[f64]) -> Vec<usize> {
    let cfg = src.config();
    let n_exp = cfg.n_experts;
    let pfx = format!("layers.{}.block_sparse_moe.experts.", layer);
    let wnames = ["w1", "w2", "w3"];
    let mut dims3 = [(0usize, 0usize); 3];
    let mut bounds = [0usize; 4]; // tensor boundaries of the concatenated layout
    for (i, wn) in wnames.iter().enumerate() {
        let e = src.entry(&format!("{}0.{}", pfx, wn));
        dims3[i] = (e.dims[0] as usize, e.dims[1] as usize);
        bounds[i + 1] = bounds[i] + dims3[i].0 * dims3[i].1;
    }
    let dim = bounds[3];
    let seed = MERGE_SEED ^ (layer as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let pos = sketch_positions(dim, seed);
    let d = pos.len();
    let mut vecs = vec![0f32; n_exp * d];
    for e in 0..n_exp {
        let row = &mut vecs[e * d..(e + 1) * d];
        for (i, wn) in wnames.iter().enumerate() {
            // the sketch positions landing in this tensor, in local coords
            let lo = pos.partition_point(|&q| q < bounds[i]);
            let hi = pos.partition_point(|&q| q < bounds[i + 1]);
            if lo == hi {
                continue;
            }
            let local: Vec<usize> = pos[lo..hi].iter().map(|&q| q - bounds[i]).collect();
            let entry = src.entry(&format!("{}{}.{}", pfx, e, wn));
            assert!(matches!(entry.dtype, DTYPE_MXFP4 | DTYPE_MXFP4SQ), "{} is not an mxfp4 flavor", entry.name);
            assert_eq!((entry.dims[0] as usize, entry.dims[1] as usize), dims3[i], "{}: expert dims differ within a layer", entry.name);
            let blob = src.raw_blob(entry);
            blob_values_at(entry.dtype, &blob, dims3[i].0, dims3[i].1, &local, &mut row[lo..hi]);
        }
    }
    kmeans_weighted(&vecs, n_exp, d, weights, n, MERGE_ITERS, seed)
}

/// Clusters every kept MoE layer's experts into n clusters (one thread per
/// layer, like the Frobenius scoring). The per-old-expert cluster ids are
/// checkpointed (crash-safe resume); the score cache is READ ONLY here, so a
/// merge rerun on the same model + cache state is bit-identical.
pub(super) fn cluster_layers(src: &Source, kept_layers: &[usize], n: usize, ckpt: &SliceCkpt, sc: &ScoreCache) -> HashMap<usize, LayerMerge> {
    let cfg = src.config();
    let moe_layers: Vec<usize> = kept_layers.iter().copied().filter(|&l| cfg.is_moe(l)).collect();
    let total = moe_layers.len();
    let weights: HashMap<usize, Vec<f64>> = moe_layers.iter().map(|&l| (l, layer_weights(sc, l, cfg.n_experts))).collect();
    let mut restored: Vec<(usize, Vec<Vec<usize>>)> = Vec::new();
    let mut todo: Vec<usize> = Vec::new();
    for &l in &moe_layers {
        let valid = ckpt
            .merges
            .get(&l)
            .filter(|a| a.len() == cfg.n_experts && a.iter().all(|&c| c < n))
            .map(|a| clusters_of(a, n))
            .filter(|cl| cl.iter().all(|c| !c.is_empty()));
        match valid {
            Some(cl) => restored.push((l, cl)),
            None => todo.push(l),
        }
    }
    for (l, _) in &restored {
        println!("merge: layer {} clusters restored from checkpoint", l);
    }
    let done = std::sync::atomic::AtomicUsize::new(restored.len());
    let results: Vec<(usize, Vec<Vec<usize>>)> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for &l in &todo {
            let weights = &weights;
            let done = &done;
            handles.push(scope.spawn(move || {
                let t = std::time::Instant::now();
                let assign = cluster_layer(src, l, n, &weights[&l]);
                ckpt.record_merge(l, &assign);
                let cl = clusters_of(&assign, n);
                let (lo, hi) = (
                    cl.iter().map(|c| c.len()).min().unwrap_or(0),
                    cl.iter().map(|c| c.len()).max().unwrap_or(0),
                );
                let nd = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                println!(
                    "merge: clustered layer {}/{} (layer {}, {} -> {} experts, {}..{} members per cluster, {:.1?})",
                    nd, total, l, cfg.n_experts, n, lo, hi, t.elapsed()
                );
                (l, cl)
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    restored
        .into_iter()
        .chain(results)
        .map(|(l, clusters)| {
            let w = weights[&l].clone();
            (l, LayerMerge { clusters, weights: w })
        })
        .collect()
}

/// Weighted average of same-length f32 vectors (f64 accumulation, weights
/// renormalized to sum 1). A single member round-trips exactly.
pub(super) fn weighted_avg(members: &[&[f32]], weights: &[f64]) -> Vec<f32> {
    assert!(!members.is_empty());
    let wsum: f64 = weights.iter().sum();
    assert!(wsum > 0.0);
    let mut acc = vec![0f64; members[0].len()];
    for (m, &w) in members.iter().zip(weights) {
        assert_eq!(m.len(), members[0].len());
        let a = w / wsum;
        for (s, &v) in acc.iter_mut().zip(m.iter()) {
            *s += a * v as f64;
        }
    }
    acc.iter().map(|&x| x as f32).collect()
}

/// The merged expert matrix of one cluster: dequantize every member
/// (w1/w2/w3 tensor `wname` of the SOURCE layer), weighted-average with the
/// layer's merge weights, requantize to mxfp4 (searched scale). Returns the
/// blob in the packed ++ scales layout.
pub(super) fn merge_expert_blob(src: &Source, layer: usize, members: &[usize], weights: &[f64], wname: &str, rows: usize, cols: usize) -> Vec<u8> {
    let mut tensors: Vec<Vec<f32>> = Vec::with_capacity(members.len());
    let mut mws = Vec::with_capacity(members.len());
    for &e in members {
        let name = format!("layers.{}.block_sparse_moe.experts.{}.{}", layer, e, wname);
        let entry = src.entry(&name);
        assert!(matches!(entry.dtype, DTYPE_MXFP4 | DTYPE_MXFP4SQ), "{} is not an mxfp4 flavor", name);
        assert_eq!((entry.dims[0] as usize, entry.dims[1] as usize), (rows, cols), "{}: member dims mismatch", name);
        let blob = src.raw_blob(entry);
        tensors.push(crate::quant::mxfp4::dequant_any(entry.dtype, &blob, rows, cols));
        mws.push(weights[e]);
    }
    // a cluster of all-zero-score members averages uniformly (never NaN)
    if mws.iter().sum::<f64>() <= 0.0 {
        mws = vec![1.0; members.len()];
    }
    let refs: Vec<&[f32]> = tensors.iter().map(|t| t.as_slice()).collect();
    let avg = weighted_avg(&refs, &mws);
    let (mut packed, scales) = crate::quant::mxfp4::quantize(&avg, rows, cols);
    packed.extend_from_slice(&scales);
    packed
}

/// Stable logsumexp: m + ln(sum(exp(v - m))); all -inf stays -inf.
fn logsumexp(vals: impl Iterator<Item = f32> + Clone) -> f32 {
    let m = vals.clone().fold(f32::NEG_INFINITY, f32::max);
    if !m.is_finite() {
        return m;
    }
    (m as f64 + vals.map(|v| ((v - m) as f64).exp()).sum::<f64>().ln()) as f32
}

/// Merged router weight rows [clusters, ch.len()]: per output column j,
/// new_logit = log(sum_i exp(old_logit_i)) over the cluster members, so the
/// total routing mass exp(logit) of each column is conserved exactly.
pub(super) fn router_merge_w(w: &[f32], rows: usize, cols: usize, clusters: &[Vec<usize>], ch: &[usize]) -> Vec<f32> {
    assert_eq!(w.len(), rows * cols);
    let mut out = Vec::with_capacity(clusters.len() * ch.len());
    for cl in clusters {
        assert!(!cl.is_empty());
        for &j in ch {
            out.push(logsumexp(cl.iter().map(|&e| w[e * cols + j])));
        }
    }
    out
}

/// Merged e_score_correction_bias [clusters]: same logsumexp mass
/// conservation, one scalar per cluster.
pub(super) fn router_merge_b(b: &[f32], clusters: &[Vec<usize>]) -> Vec<f32> {
    clusters.iter().map(|cl| logsumexp(cl.iter().map(|&e| b[e]))).collect()
}

/// --merge-experts replaces the whole expert axis of a layer, so it cannot
/// combine with the other expert-axis options.
pub(super) fn check_compatible(experts: Option<usize>, merge: Option<usize>, cold_vq: Option<usize>, expert_order: Option<&String>) -> Result<(), String> {
    if merge.is_none() {
        return Ok(());
    }
    if experts.is_some() {
        return Err("--merge-experts is mutually exclusive with --experts (merging replaces deletion)".to_string());
    }
    if cold_vq.is_some() {
        return Err("--merge-experts is mutually exclusive with --cold-vq (the hot/cold split has no meaning on merged experts)".to_string());
    }
    if expert_order.is_some() {
        return Err("--merge-experts is mutually exclusive with --expert-order (merged experts are new ids, there is no old order to keep)".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic synthetic value (hash noise in [-1, 1)).
    fn noise(i: u64) -> f32 {
        let h = i.wrapping_mul(2654435761).wrapping_add(0x9E3779B9);
        ((h >> 13) % 2000) as f32 / 1000.0 - 1.0
    }

    #[test]
    fn kmeans_deterministic_and_recovers_groups() {
        // 3 tight groups of 4 experts (groups 10 apart, noise 0.01)
        let (n, dim, k) = (12usize, 32usize, 3usize);
        let mut v = vec![0f32; n * dim];
        for e in 0..n {
            for j in 0..dim {
                v[e * dim + j] = (e / 4 * 10) as f32 + noise((e * dim + j) as u64) * 0.01;
            }
        }
        let w = vec![1.0 / n as f64; n];
        let a1 = kmeans_weighted(&v, n, dim, &w, k, MERGE_ITERS, 42);
        let a2 = kmeans_weighted(&v, n, dim, &w, k, MERGE_ITERS, 42);
        assert_eq!(a1, a2, "same input, same seed -> same clusters");
        let mut got = clusters_of(&a1, k);
        let mut want: Vec<Vec<usize>> = (0..k).map(|g| (g * 4..g * 4 + 4).collect()).collect();
        got.sort();
        want.sort();
        assert_eq!(got, want, "clusters recover the planted groups");
        // weighted runs are deterministic too, and every cluster non-empty
        let w2: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let a3 = kmeans_weighted(&v, n, dim, &w2, k, MERGE_ITERS, 42);
        assert_eq!(a3, kmeans_weighted(&v, n, dim, &w2, k, MERGE_ITERS, 42));
        assert!(clusters_of(&a3, k).iter().all(|c| !c.is_empty()));
        // k = n - 1 still leaves no empty cluster
        let a4 = kmeans_weighted(&v, n, dim, &w, n - 1, MERGE_ITERS, 7);
        assert!(clusters_of(&a4, n - 1).iter().all(|c| !c.is_empty()));
    }

    #[test]
    fn hot_expert_stays_singleton() {
        // 5 cold experts near the origin, 1 far-away hot expert (weight 1000)
        let (n, dim, k) = (6usize, 16usize, 2usize);
        let mut v = vec![0f32; n * dim];
        for e in 0..n {
            for j in 0..dim {
                v[e * dim + j] = if e == 5 { 100.0 + noise((e * dim + j) as u64) } else { noise((e * dim + j) as u64) * 0.01 };
            }
        }
        let mut w = vec![1.0; n];
        w[5] = 1000.0;
        let a = kmeans_weighted(&v, n, dim, &w, k, MERGE_ITERS, 1);
        let cl = clusters_of(&a, k);
        assert!(cl.iter().any(|c| c.len() == 1 && c[0] == 5), "hot expert 5 must be a singleton: {:?}", cl);
    }

    #[test]
    fn logsumexp_conserves_routing_mass() {
        // bias rows: total exp(logit) over the layer is conserved per cluster merge
        let logits: Vec<f32> = (0..40).map(|i| ((i * 37 % 100) as f32 - 50.0) / 10.0).collect();
        let clusters = vec![vec![0, 1, 2, 3], vec![4, 5], (6..40).collect::<Vec<_>>()];
        let merged = router_merge_b(&logits, &clusters);
        let sum_old: f64 = logits.iter().map(|&x| (x as f64).exp()).sum();
        let sum_new: f64 = merged.iter().map(|&x| (x as f64).exp()).sum();
        assert!((sum_new - sum_old).abs() / sum_old < 1e-5, "{} vs {}", sum_new, sum_old);
        // weight rows: mass is conserved per COLUMN
        let (rows, cols) = (8usize, 16usize);
        let w: Vec<f32> = (0..rows * cols).map(|i| ((i * 53 % 97) as f32 - 48.0) / 8.0).collect();
        let cls = vec![vec![0, 1, 2], vec![3], vec![4, 5, 6, 7]];
        let ch: Vec<usize> = (0..cols).collect();
        let mw = router_merge_w(&w, rows, cols, &cls, &ch);
        assert_eq!(mw.len(), cls.len() * cols);
        for j in 0..cols {
            let so: f64 = (0..rows).map(|e| (w[e * cols + j] as f64).exp()).sum();
            let sn: f64 = (0..cls.len()).map(|k| (mw[k * cols + j] as f64).exp()).sum();
            assert!((sn - so).abs() / so < 1e-5, "column {} mass not conserved: {} vs {}", j, sn, so);
        }
        // channel gather applies on top of the merge
        let ch2: Vec<usize> = (0..cols).step_by(3).collect();
        let mw2 = router_merge_w(&w, rows, cols, &cls, &ch2);
        assert_eq!(mw2.len(), cls.len() * ch2.len());
        for (j, &c) in ch2.iter().enumerate() {
            assert_eq!(mw2[1 * ch2.len() + j], mw[1 * cols + c]);
        }
        // numerical stability: huge and all -inf inputs stay finite
        assert!(router_merge_b(&[1000.0, 1000.0], &[vec![0, 1]])[0].is_finite());
        assert_eq!(router_merge_b(&[f32::NEG_INFINITY, f32::NEG_INFINITY], &[vec![0, 1]])[0], f32::NEG_INFINITY);
    }

    #[test]
    fn merged_average_weighted() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![4.0f32, 3.0, 2.0, 1.0];
        let avg = weighted_avg(&[&a, &b], &[1.0, 3.0]);
        for (&x, &y) in avg.iter().zip([3.25f32, 2.75, 2.25, 1.75].iter()) {
            assert!((x - y).abs() < 1e-6, "{} vs {}", x, y);
        }
        // uniform weights == plain mean
        let avg = weighted_avg(&[&a, &b], &[1.0, 1.0]);
        for (&x, &y) in avg.iter().zip([2.5f32, 2.5, 2.5, 2.5].iter()) {
            assert!((x - y).abs() < 1e-6);
        }
        // a singleton merge is the exact identity, whatever its weight
        assert_eq!(weighted_avg(&[&a], &[7.0]), a);
    }

    #[test]
    fn mxfp4_roundtrip_through_merge() {
        let (r, c) = (4usize, 64usize);
        let w1: Vec<f32> = (0..r * c).map(|i| noise(i as u64)).collect();
        let w2: Vec<f32> = (0..r * c).map(|i| noise(i as u64 + 12345)).collect();
        // quantize each exactly like the builder stores them
        let store = |w: &[f32]| {
            let (mut p, s) = crate::quant::mxfp4::quantize(w, r, c);
            p.extend_from_slice(&s);
            p
        };
        let (b1, b2) = (store(&w1), store(&w2));
        let d1 = crate::quant::mxfp4::dequant_any(DTYPE_MXFP4, &b1, r, c);
        let d2 = crate::quant::mxfp4::dequant_any(DTYPE_MXFP4, &b2, r, c);
        // the merge path: dequant members, weighted average, requantize
        let avg = weighted_avg(&[&d1, &d2], &[1.0, 2.0]);
        let back = crate::quant::mxfp4::dequant_any(DTYPE_MXFP4, &store(&avg), r, c);
        let rel = |a: &[f32], b: &[f32]| {
            let (mut num, mut den) = (0f64, 0f64);
            for (&x, &y) in a.iter().zip(b) {
                num += (x as f64 - y as f64) * (x as f64 - y as f64);
                den += x as f64 * x as f64;
            }
            (num / den).sqrt()
        };
        let err_merge = rel(&avg, &back);
        // sanity bound: requantizing the merged average must not be much
        // worse than a plain quantize roundtrip of a member (same format)
        let err_plain = rel(&w1, &d1);
        assert!(err_merge <= err_plain * 1.5 + 1e-6, "merged requant rel RMS {} vs plain {}", err_merge, err_plain);
        // the sketch gather reproduces a full dequant at the same positions
        let pos: Vec<usize> = (0..r * c).step_by(7).collect();
        let mut out = vec![0f32; pos.len()];
        blob_values_at(DTYPE_MXFP4, &b1, r, c, &pos, &mut out);
        for (&q, &o) in pos.iter().zip(&out) {
            assert_eq!(o, d1[q]);
        }
    }

    #[test]
    fn sketch_positions_deterministic() {
        let a = sketch_positions(100_000, 7);
        assert_eq!(a, sketch_positions(100_000, 7));
        assert_eq!(a.len(), SKETCH_DIM);
        assert!(a.windows(2).all(|w| w[0] < w[1]), "sorted distinct positions");
        assert_ne!(a, sketch_positions(100_000, 8), "another seed, another subset");
        // small dims: the identity (exact k-means)
        assert_eq!(sketch_positions(100, 7), (0..100).collect::<Vec<_>>());
        assert_eq!(sketch_positions(SKETCH_DIM, 7).len(), SKETCH_DIM);
    }

    #[test]
    fn arg_validation() {
        assert!(check_compatible(None, Some(8), None, None).is_ok());
        assert!(check_compatible(Some(8), Some(8), None, None).is_err(), "merge + delete");
        assert!(check_compatible(None, Some(8), Some(4), None).is_err(), "merge + cold-vq");
        assert!(check_compatible(None, Some(8), None, Some(&"frequency".to_string())).is_err(), "merge + expert-order");
        assert!(check_compatible(Some(8), None, Some(4), None).is_ok());
        assert!(check_compatible(None, None, None, None).is_ok());
    }
}
