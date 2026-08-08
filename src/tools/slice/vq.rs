// Cold-expert VQ1 requantization: reservoir sampling and per-tensor quantization (moved from slice.rs).

use super::source::{DirEntry, Source};
use crate::quant::weights::{DTYPE_MXFP4, DTYPE_MXFP4SQ};

/// Reservoir sample (algorithm R, seeded splitmix64) of the raw 16-vectors
/// of every cold expert tensor, for the global VQ codebook training. Each
/// tensor is dequantized from mxfp4 (either flavor) one at a time (never the
/// whole model in RAM). `cold` = per kept MoE layer the cold expert indices
/// (ascending). With an imatrix, per-value activation weights are sampled
/// alongside the vectors (same layout) for the weighted codebook training;
/// without one the rng/sample sequence is exactly the historical one
/// (bit-identical codebook).
pub(super) fn vq_reservoir(
    src: &Source,
    kept_layers: &[usize],
    cold: &std::collections::HashMap<usize, Vec<usize>>,
    cap: usize,
    seed: u64,
    im: Option<&crate::quant::imatrix::Imatrix>,
) -> (Vec<f32>, Option<Vec<f32>>) {
    use crate::quant::quant::{Rng, VQ_DIM};
    let cfg = src.config();
    let moe_layers: Vec<usize> = kept_layers.iter().copied().filter(|&l| cfg.is_moe(l)).collect();
    let mut rng = Rng::new(seed);
    let mut res: Vec<f32> = Vec::with_capacity(cap * VQ_DIM);
    let mut wres: Option<Vec<f32>> = im.map(|_| Vec::with_capacity(cap * VQ_DIM));
    let mut seen = 0u64; // vectors offered so far
    // A tensor without usable imatrix stats falls back to flat weights (1.0),
    // so `wres` always stays aligned with `res`. `cw` holds one weight per
    // matrix COLUMN: vector `vi` of a row of `vpr` vectors covers columns
    // (vi % vpr) * VQ_DIM .. + VQ_DIM.
    let feed = |w: &[f32], cw: Option<&[f32]>, vpr: usize, res: &mut Vec<f32>, wres: &mut Option<Vec<f32>>, rng: &mut Rng, seen: &mut u64| {
        for (vi, v) in w.chunks_exact(VQ_DIM).enumerate() {
            let c0 = (vi % vpr) * VQ_DIM;
            let t = *seen;
            *seen += 1;
            if (t as usize) < cap {
                res.extend_from_slice(v);
                if let Some(wr) = wres.as_mut() {
                    match cw {
                        Some(cw) => wr.extend_from_slice(&cw[c0..c0 + VQ_DIM]),
                        None => wr.extend_from_slice(&[1.0; VQ_DIM]),
                    }
                }
            } else {
                let j = rng.below(t as usize + 1);
                if j < cap {
                    res[j * VQ_DIM..(j + 1) * VQ_DIM].copy_from_slice(v);
                    if let Some(wr) = wres.as_mut() {
                        match cw {
                            Some(cw) => wr[j * VQ_DIM..(j + 1) * VQ_DIM].copy_from_slice(&cw[c0..c0 + VQ_DIM]),
                            None => wr[j * VQ_DIM..(j + 1) * VQ_DIM].copy_from_slice(&[1.0; VQ_DIM]),
                        }
                    }
                }
            }
        }
    };
    for &l in &moe_layers {
        let pfx = format!("layers.{}.block_sparse_moe.experts.", l);
        for &e in &cold[&l] {
            for wn in ["w1", "w2", "w3"] {
                let name = format!("{}{}.{}", pfx, e, wn);
                let entry = src.entry(&name);
                assert!(matches!(entry.dtype, DTYPE_MXFP4 | DTYPE_MXFP4SQ), "{} is not an mxfp4 flavor", name);
                let blob = src.raw_blob(entry);
                let (r, c) = (entry.dims[0] as usize, entry.dims[1] as usize);
                let w = crate::quant::mxfp4::dequant_any(entry.dtype, &blob, r, c);
                let cw = im.and_then(|im| im.col_weights(l, wn));
                feed(&w, cw.as_deref(), c / VQ_DIM, &mut res, &mut wres, &mut rng, &mut seen);
            }
        }
    }
    println!("vq: reservoir sampled {}/{} cold-expert vectors (cap {})", res.len() / VQ_DIM, seen, cap);
    (res, wres)
}

/// Dequantizes one source expert tensor (either mxfp4 flavor) and VQ-quantizes
/// it with the shared codebook. With `quant_col_w` the nearest-centroid
/// assignment is the activation-weighted one (slice --imatrix); both weight
/// arguments hold ONE WEIGHT PER MATRIX COLUMN (expanded per element here).
/// Returns (index bytes, relative Frobenius error, activation-weighted
/// relative error when `score_col_w` is given).
pub(super) fn vq_quantize_tensor(
    src: &Source,
    e: &DirEntry,
    codebook: &[f32],
    quant_col_w: Option<&[f32]>,
    score_col_w: Option<&[f32]>,
) -> (Vec<u8>, f64, Option<f64>) {
    assert!(matches!(e.dtype, DTYPE_MXFP4 | DTYPE_MXFP4SQ), "{} is not an mxfp4 flavor", e.name);
    let (r, c) = (e.dims[0] as usize, e.dims[1] as usize);
    assert_eq!(c % crate::quant::quant::VQ_DIM, 0, "{}: cols {} not a multiple of {}", e.name, c, crate::quant::quant::VQ_DIM);
    let blob = src.raw_blob(e);
    let w = crate::quant::mxfp4::dequant_any(e.dtype, &blob, r, c);
    // per-column importance -> per-element weights (same row of c weights
    // repeated r times; expert matrices are small enough to materialize it)
    let expand = |cw: &[f32]| -> Vec<f32> {
        assert_eq!(cw.len(), c, "{}: imatrix column count {} != tensor cols {}", e.name, cw.len(), c);
        let mut v = Vec::with_capacity(r * c);
        for _ in 0..r {
            v.extend_from_slice(cw);
        }
        v
    };
    let quant_w = quant_col_w.map(expand);
    let score_w = score_col_w.map(expand);
    let idx = match &quant_w {
        Some(wv) => crate::quant::quant::quantize_weighted(&w, wv, codebook),
        None => crate::quant::quant::quantize(&w, codebook),
    };
    let err = crate::quant::quant::rel_error(&w, &idx, codebook);
    let werr = score_w.map(|wv| crate::quant::quant::rel_error_weighted(&w, &wv, &idx, codebook));
    (idx, err, werr)
}
