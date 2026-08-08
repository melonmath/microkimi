// Seam adapter: low-rank (A, B) patch embedded in the .bin, applied on the
// residual stream after layer seam_after (h += B @ (A @ h)). Load is strict:
// unpaired tensors, a missing seam_after key (or the reverse) and bad shapes
// are hard errors. Does not touch any weight outside the adapter.

use super::*;

pub(super) struct SeamW {
    pub(super) a: T,
    pub(super) b: T,
    pub(super) rank: usize,
    pub(super) after: usize,
}

/// Reads the embedded seam adapter, if any. Clear load-time errors on an
/// inconsistent file: only one of seam.A/seam.B, tensors without the
/// seam_after config key (or the reverse), a seam_after that leaves no layer
/// N+1, wrong dtypes or shapes.
pub(super) fn seam_load(cfg: &Config, entries: &std::collections::HashMap<String, Entry>) -> Option<SeamW> {
    let ea = entries.get("seam.A");
    let eb = entries.get("seam.B");
    if ea.is_none() && eb.is_none() {
        assert!(
            cfg.seam_after.is_none(),
            "the config declares seam_after {} but the file has no seam.A/seam.B tensors",
            cfg.seam_after.unwrap()
        );
        return None;
    }
    let (ea, eb) = (
        ea.unwrap_or_else(|| panic!("seam.B present but seam.A is missing")),
        eb.unwrap_or_else(|| panic!("seam.A present but seam.B is missing")),
    );
    let after = cfg
        .seam_after
        .unwrap_or_else(|| panic!("seam.A/seam.B tensors present but the config has no seam_after key"));
    assert!(
        after + 1 < cfg.n_layers,
        "seam_after {} out of range [0, {}] for a {}-layer model: the seam adapter needs a layer N+1",
        after,
        cfg.n_layers.saturating_sub(2),
        cfg.n_layers
    );
    for (name, e) in [("seam.A", ea), ("seam.B", eb)] {
        assert_eq!(e.dtype, crate::weights::DTYPE_F32, "{}: dtype {}, only fp32", name, e.dtype);
        assert_eq!(e.dims.len(), 2, "{}: dims {:?}, expected a 2D matrix", name, e.dims);
    }
    let (rank, d) = (ea.dims[0] as usize, ea.dims[1] as usize);
    assert_eq!(d, cfg.d, "seam.A: dims {:?}, expected [rank, {}] (hidden)", ea.dims, cfg.d);
    assert_eq!(
        eb.dims,
        vec![cfg.d as u32, rank as u32],
        "seam.B: dims {:?}, expected [{}, {}] (hidden, rank)",
        eb.dims,
        cfg.d,
        rank
    );
    Some(SeamW { a: T::from(ea), b: T::from(eb), rank, after })
}

/// h += (h @ A^T) @ B^T on one residual-stream row: the same two matvecs the
/// Python SeamAdapter computes (h + (h @ A.T) @ B.T), through the engine's
/// bit-exact dot(), so the batched prefill (per position) and the decode
/// produce the same values.
pub(super) fn seam_apply(data: &[u8], w: &SeamW, d: usize, h: &mut [f32]) {
    let a = as_f32(&data[w.a.off..w.a.off + w.a.len * 4]);
    let b = as_f32(&data[w.b.off..w.b.off + w.b.len * 4]);
    let mut tmp = vec![0f32; w.rank];
    matvec(a, w.rank, d, h, &mut tmp);
    let mut delta = vec![0f32; d];
    matvec(b, d, w.rank, &tmp, &mut delta);
    for j in 0..d {
        h[j] += delta[j];
    }
}
