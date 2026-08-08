// Output tensor plan and the f32 slicing primitives (moved from slice.rs).

use super::source::{DirEntry, Role};

/// One output tensor: what to write and where the data comes from.
pub(super) struct Plan {
    pub(super) out_name: String,
    pub(super) dtype: u8,
    pub(super) dims: Vec<u32>,
    pub(super) src_name: String,
    pub(super) role: Role,
    pub(super) channels: Vec<usize>,          // hidden keep-set (identity when no --hidden)
    pub(super) experts: Option<Vec<usize>>,  // expert keep-set for RouterW/RouterB rows
    pub(super) vocab: Option<Vec<usize>>,    // vocab keep-set (old row ids, ascending) for embed/lm_head
}

/// Slices an f32 tensor according to its role. Returns (values, new_dims).
pub(super) fn slice_f32(e: &DirEntry, w: &[f32], role: Role, ch: &[usize], experts: Option<&Vec<usize>>) -> (Vec<f32>, Vec<u32>) {
    let (r, c) = (e.dims[0] as usize, *e.dims.get(1).unwrap_or(&1) as usize);
    match role {
        Role::VecD => (ch.iter().map(|&j| w[j]).collect(), vec![ch.len() as u32]),
        Role::ColsD => {
            let mut out = Vec::with_capacity(r * ch.len());
            for row in w.chunks_exact(c) {
                out.extend(ch.iter().map(|&j| row[j]));
            }
            (out, vec![r as u32, ch.len() as u32])
        }
        Role::RowsD => {
            let mut out = Vec::with_capacity(ch.len() * c);
            for &j in ch {
                out.extend_from_slice(&w[j * c..(j + 1) * c]);
            }
            (out, vec![ch.len() as u32, c as u32])
        }
        Role::BothD => {
            let mut out = Vec::with_capacity(ch.len() * ch.len());
            for &i in ch {
                let row = &w[i * c..(i + 1) * c];
                out.extend(ch.iter().map(|&j| row[j]));
            }
            (out, vec![ch.len() as u32, ch.len() as u32])
        }
        Role::RouterW => {
            let rows = experts.cloned().unwrap_or_else(|| (0..r).collect());
            let mut out = Vec::with_capacity(rows.len() * ch.len());
            for &i in &rows {
                let row = &w[i * c..(i + 1) * c];
                out.extend(ch.iter().map(|&j| row[j]));
            }
            (out, vec![rows.len() as u32, ch.len() as u32])
        }
        Role::RouterB => {
            let rows = experts.cloned().unwrap_or_else(|| (0..r).collect());
            (rows.iter().map(|&i| w[i]).collect(), vec![rows.len() as u32])
        }
        _ => unreachable!(),
    }
}

/// Row-chunk slicing for ColsD/RowsD. `vals` holds input rows r0..r1 and the
/// returned output rows are produced in ascending order (ch is sorted), so
/// concatenating over chunks yields the full sliced tensor.
pub(super) fn slice_f32_rows(role: Role, vals: &[f32], r0: usize, r1: usize, cols: usize, ch: &[usize]) -> Vec<f32> {
    match role {
        Role::ColsD => {
            let mut out = Vec::with_capacity((r1 - r0) * ch.len());
            for row in vals.chunks_exact(cols) {
                out.extend(ch.iter().map(|&j| row[j]));
            }
            out
        }
        Role::RowsD => {
            let lo = ch.partition_point(|&j| j < r0);
            let hi = ch.partition_point(|&j| j < r1);
            let mut out = Vec::with_capacity((hi - lo) * cols);
            for &j in &ch[lo..hi] {
                out.extend_from_slice(&vals[(j - r0) * cols..(j - r0 + 1) * cols]);
            }
            out
        }
        _ => unreachable!(),
    }
}

/// Row-chunk slicing for embed/lm_head under --vocab-top: emits only the
/// kept vocab rows of input chunk r0..r1 (keep is ascending), with columns
/// pruned to ch exactly like ColsD.
pub(super) fn slice_vocab_rows(vals: &[f32], r0: usize, r1: usize, cols: usize, ch: &[usize], keep: &[usize]) -> Vec<f32> {
    let lo = keep.partition_point(|&j| j < r0);
    let hi = keep.partition_point(|&j| j < r1);
    let mut out = Vec::with_capacity((hi - lo) * ch.len());
    for &j in &keep[lo..hi] {
        let row = &vals[(j - r0) * cols..(j - r0 + 1) * cols];
        out.extend(ch.iter().map(|&c| row[c]));
    }
    out
}

/// (new expert id, w index) of an expert plan's output name
/// "layers.N.block_sparse_moe.experts.<id>.<w>", for the physical re-sort of
/// a reordered (--expert-order) layer's expert run.
pub(super) fn expert_plan_key(out_name: &str) -> (usize, u8) {
    let tail = out_name.rsplit(".experts.").next().unwrap();
    let dot = tail.find('.').unwrap();
    let id: usize = tail[..dot].parse().unwrap();
    let w = match &tail[dot + 1..] {
        "w1" => 0,
        "w2" => 1,
        "w3" => 2,
        _ => 3,
    };
    (id, w)
}
