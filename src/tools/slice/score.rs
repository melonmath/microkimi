// Expert/channel scoring, ranking and the persistent score cache (moved from slice.rs).

use super::ckpt::{fnv1a, SliceCkpt};
use super::source::{n_rows, row_chunks, row_width, role_of, split_layer, Role, Source};
use crate::quant::weights::DTYPE_MXFP4;

/// Top-n indices by descending score, ties broken by lower index, returned
/// in ascending order (deterministic keep-set).
pub(super) fn top_n(scores: &[f64], n: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap().then(a.cmp(&b)));
    idx.truncate(n);
    idx.sort_unstable();
    idx
}

/// Channel scores: for every tensor touching d, sum |w| per channel.
/// Tensors are processed in row chunks (embeddings never sit in RAM whole;
/// remote sources stream the same chunks).
pub(super) fn channel_scores(src: &Source, kept_layers: &[usize], d: usize) -> Vec<f64> {
    let cfg = src.config();
    let arch = src.arch();
    let mut s = vec![0f64; d];
    for e in src.entries() {
        let in_layers = match split_layer(&e.name) {
            None => true, // global tensor
            Some((l, _)) => kept_layers.contains(&l),
        };
        if !in_layers {
            continue;
        }
        let role = role_of(&e.name, cfg, arch);
        if matches!(role, Role::Copy | Role::RouterB | Role::Expert) {
            continue;
        }
        let (rows, cols) = (n_rows(e), row_width(e));
        match role {
            Role::VecD => {
                let w = src.f32_rows(e, 0, 1);
                assert_eq!(w.len(), d, "{}: expected [{}]", e.name, d);
                for (j, &x) in w.iter().enumerate() {
                    s[j] += x.abs() as f64;
                }
            }
            Role::ColsD | Role::RouterW => {
                assert_eq!(cols, d, "{}: expected [_, {}]", e.name, d);
                for (r0, r1) in row_chunks(rows, cols) {
                    for row in src.f32_rows(e, r0, r1).chunks_exact(cols) {
                        for (j, &x) in row.iter().enumerate() {
                            s[j] += x.abs() as f64;
                        }
                    }
                }
            }
            Role::RowsD => {
                assert_eq!(rows, d, "{}: expected [{}, _]", e.name, d);
                for (r0, r1) in row_chunks(rows, cols) {
                    for (j, row) in src.f32_rows(e, r0, r1).chunks_exact(cols).enumerate() {
                        s[r0 + j] += row.iter().map(|&x| x.abs() as f64).sum::<f64>();
                    }
                }
            }
            Role::BothD => {
                assert_eq!((rows, cols), (d, d), "{}: expected [{}, {}]", e.name, d, d);
                let w = src.f32_rows(e, 0, rows);
                // the channel appears as BOTH the output (row) and the input
                // (column) axis: row sums + column sums
                for (j, row) in w.chunks_exact(cols).enumerate() {
                    s[j] += row.iter().map(|&x| x.abs() as f64).sum::<f64>();
                }
                for row in w.chunks_exact(cols) {
                    for (j, &x) in row.iter().enumerate() {
                        s[j] += x.abs() as f64;
                    }
                }
            }
            _ => unreachable!(),
        }
    }
    s
}

/// Squared Frobenius norm of an expert's w1+w2+w3 (mxfp4 dequantized).
pub(super) fn expert_norm_sq(blob_w1: &[u8], blob_w2: &[u8], blob_w3: &[u8], rows_cols: [(usize, usize); 3]) -> f64 {
    let mut s = 0f64;
    for (blob, &(r, c)) in [blob_w1, blob_w2, blob_w3].iter().zip(rows_cols.iter()) {
        let np = r * c / 2;
        let w = crate::quant::mxfp4::dequant(&blob[..np], &blob[np..], r, c);
        s += w.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>();
    }
    s
}

/// Scale-energy score: proportional to the squared Frobenius norm under the
/// e2m1 codebook. ||W||^2 = sum over groups of 2^(2*(s-127)) * sum(lut^2);
/// the per-group lut factor has the same expectation for every expert, so
/// ranking on the scale energy alone ranks the same way while reading 1/17
/// of the bytes (the weight_scale tensors only). Used for safetensors
/// sources; the .bin path keeps the exact dequantized Frobenius.
pub(super) fn mxfp4_scale_energy(scale_bytes: &[u8]) -> f64 {
    scale_bytes
        .iter()
        .map(|&s| {
            let e = s as i32 - 127;
            (2f64).powi(2 * e)
        })
        .sum()
}

/// Per kept MoE layer: the n_experts_keep expert indices to keep (ascending).
/// .bin path: exact dequantized Frobenius, one thread per layer (disk reads).
/// Safetensors path: scale-energy from the weight_scale tensors only; layers
/// are processed sequentially but each layer's experts are split across 8
/// threads (remote scoring is curl-latency bound).
/// Every finished layer is logged and checkpointed immediately; layers found
/// in the checkpoint are restored instead of re-scored (both branches). Full
/// scores are persisted in the persistent ScoreCache (config-independent:
/// the keep-set is recomputed from the cached scores with the current N).
pub(super) fn expert_keep_sets(src: &Source, kept_layers: &[usize], n_keep: usize, ckpt: &SliceCkpt, sc: &ScoreCache) -> std::collections::HashMap<usize, Vec<usize>> {
    let cfg = src.config();
    let moe_layers: Vec<usize> = kept_layers.iter().copied().filter(|&l| cfg.is_moe(l)).collect();
    let total = moe_layers.len();
    let mut restored: Vec<(usize, Vec<usize>)> = Vec::new();
    let mut todo: Vec<usize> = Vec::new();
    for &l in &moe_layers {
        match ckpt.experts.get(&l) {
            Some(set) => restored.push((l, set.clone())),
            None => todo.push(l),
        }
    }
    for (l, _) in &restored {
        println!("experts: layer {} restored from checkpoint", l);
    }
    let done = std::sync::atomic::AtomicUsize::new(restored.len());
    // shared by both branches: keep-set of one layer, from the score cache
    // when possible (scoring skipped entirely), freshly computed otherwise.
    // Returns (keep-set, Some(top score range) when freshly scored).
    let keep_of = |l: usize, compute: &dyn Fn() -> Vec<f64>| -> (Vec<usize>, Option<(f64, f64)>) {
        if let Some(cached) = sc.get(l) {
            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            println!("experts: layer {} scores restored from score cache ({}/{})", l, n, total);
            return (top_n(cached, n_keep.min(cfg.n_experts)), None);
        }
        let scores = compute();
        sc.record(l, &scores);
        let keep = top_n(&scores, n_keep.min(cfg.n_experts));
        let lo = keep.iter().map(|&e| scores[e]).fold(f64::INFINITY, f64::min);
        let hi = keep.iter().map(|&e| scores[e]).fold(f64::NEG_INFINITY, f64::max);
        let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        println!("experts: scored layer {}/{} (layer {}, kept {}/{}, top score range {:.3} .. {:.3})", n, total, l, keep.len(), cfg.n_experts, lo, hi);
        (keep, Some((lo, hi)))
    };
    if matches!(src, Source::St(_)) {
        let mut results = restored;
        for &l in &todo {
            let (keep, _) = keep_of(l, &|| {
                let pfx = format!("layers.{}.block_sparse_moe.experts.", l);
                let mut scores = vec![0f64; cfg.n_experts];
                let nt = 8.min(cfg.n_experts);
                let chunk = cfg.n_experts.div_ceil(nt);
                std::thread::scope(|scope| {
                    for (ti, sc) in scores.chunks_mut(chunk).enumerate() {
                        let pfx = &pfx;
                        scope.spawn(move || {
                            for (i, slot) in sc.iter_mut().enumerate() {
                                let e = ti * chunk + i;
                                // scales only: the packed nibbles are never fetched
                                *slot = ["w1", "w2", "w3"]
                                    .iter()
                                    .map(|wn| mxfp4_scale_energy(&src.expert_scales(src.entry(&format!("{}{}.{}", pfx, e, wn)))))
                                    .sum();
                            }
                        });
                    }
                });
                scores
            });
            ckpt.record_experts(l, &keep);
            results.push((l, keep));
        }
        return results.into_iter().collect();
    }
    let results: Vec<(usize, Vec<usize>)> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for &l in &todo {
            let keep_of = &keep_of;
            handles.push(scope.spawn(move || {
                let (keep, _) = keep_of(l, &|| {
                    let pfx = format!("layers.{}.block_sparse_moe.experts.", l);
                    let mut scores = vec![0f64; cfg.n_experts];
                    for e in 0..cfg.n_experts {
                        let mut blobs = [Vec::new(), Vec::new(), Vec::new()];
                        let mut rc = [(0usize, 0usize); 3];
                        for (i, wn) in ["w1", "w2", "w3"].iter().enumerate() {
                            let name = format!("{}{}.{}", pfx, e, wn);
                            let entry = src.entry(&name);
                            assert_eq!(entry.dtype, DTYPE_MXFP4, "{} is not mxfp4", name);
                            blobs[i] = src.raw_blob(entry);
                            rc[i] = (entry.dims[0] as usize, entry.dims[1] as usize);
                        }
                        scores[e] = expert_norm_sq(&blobs[0], &blobs[1], &blobs[2], rc);
                    }
                    scores
                });
                ckpt.record_experts(l, &keep);
                (l, keep)
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    results.into_iter().chain(restored).collect()
}

/// Per kept MoE layer: the full per-expert Frobenius scores (.bin sources
/// only; the exact dequantized w1+w2+w3 norm, one thread per layer). Used by
/// --cold-vq, which needs the complete ranking (hot top-N vs cold tail), not
/// just a keep-set.
pub(super) fn expert_score_map(src: &Source, kept_layers: &[usize]) -> std::collections::HashMap<usize, Vec<f64>> {
    let cfg = src.config();
    let moe_layers: Vec<usize> = kept_layers.iter().copied().filter(|&l| cfg.is_moe(l)).collect();
    let results: Vec<(usize, Vec<f64>)> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for &l in &moe_layers {
            handles.push(scope.spawn(move || {
                let pfx = format!("layers.{}.block_sparse_moe.experts.", l);
                let mut scores = vec![0f64; cfg.n_experts];
                for e in 0..cfg.n_experts {
                    let mut blobs = [Vec::new(), Vec::new(), Vec::new()];
                    let mut rc = [(0usize, 0usize); 3];
                    for (i, wn) in ["w1", "w2", "w3"].iter().enumerate() {
                        let name = format!("{}{}.{}", pfx, e, wn);
                        let entry = src.entry(&name);
                        assert_eq!(entry.dtype, DTYPE_MXFP4, "{} is not mxfp4", name);
                        blobs[i] = src.raw_blob(entry);
                        rc[i] = (entry.dims[0] as usize, entry.dims[1] as usize);
                    }
                    scores[e] = expert_norm_sq(&blobs[0], &blobs[1], &blobs[2], rc);
                }
                (l, scores)
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    results.into_iter().collect()
}

// ── persistent expert-score cache (<out>.slicecache/expert_scores.<fnv1a64>.bin) ──
//
// expert_keep_sets scores every kept MoE layer of the SOURCE model
// (scale-energy for remote safetensors, dequantized Frobenius for local
// .bin). That scoring depends ONLY on the source model and the layer - not
// on --layers/--hidden/--experts/--cold-vq - yet the full scores used to be
// discarded once the top-N keep-set was computed (~45k HTTP range requests
// / 2h+ lost on every remote rerun). This cache persists the full scores
// across runs AND across pruning configs, one level below the per-run
// .sliceckpt (which keeps the config-dependent keep-sets of the current
// run; its config hash does NOT cover this cache, by design). It is a pure
// optimization: deleting the file changes nothing but time.
//
// Format (all little-endian):
//   magic "MKSCORE1" (8 bytes) | u32 version (=1) | u32 n_layers | u32 n_experts
//   then one record per scored MoE layer, appended as soon as it finishes:
//     u32 layer | n_experts * f64 score
// Appends are one write + fsync (kill -9 safe); a torn trailing record is
// ignored on read. magic/version/n_experts mismatch -> the file is ignored
// and replaced (never a panic). Duplicate layer records: the last wins.

const SCORE_MAGIC: &[u8; 8] = b"MKSCORE1";
const SCORE_VERSION: u32 = 1;
const SCORE_HEADER: usize = 8 + 4 + 4 + 4;

pub(super) struct ScoreCache {
    file: std::sync::Mutex<std::fs::File>,
    scores: std::collections::HashMap<usize, Vec<f64>>, // layer -> full per-expert scores
}

impl ScoreCache {
    /// Opens (or creates) the score cache for `model` under
    /// <out>.slicecache/. A valid existing file restores its layer scores,
    /// anything else (missing, bad magic/version/expert count) starts empty.
    pub(super) fn open(out: &str, model: &str, n_layers: usize, n_experts: usize) -> ScoreCache {
        use std::io::Write;
        let dir = std::path::PathBuf::from(format!("{}.slicecache", out));
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join(format!("expert_scores.{:016x}.bin", fnv1a(model)));
        let mut scores = std::collections::HashMap::new();
        let mut valid = !path.is_file();
        let mut valid_len = 0usize; // end of the last complete record
        if let Ok(bytes) = std::fs::read(&path) {
            valid = false;
            if bytes.len() >= SCORE_HEADER
                && &bytes[..8] == SCORE_MAGIC
                && u32::from_le_bytes(bytes[8..12].try_into().unwrap()) == SCORE_VERSION
                && u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize == n_experts
            {
                let rec = 4 + 8 * n_experts;
                let mut pos = SCORE_HEADER;
                while pos + rec <= bytes.len() {
                    let layer = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
                    let v: Vec<f64> = (0..n_experts)
                        .map(|i| f64::from_le_bytes(bytes[pos + 4 + 8 * i..pos + 12 + 8 * i].try_into().unwrap()))
                        .collect();
                    scores.insert(layer, v); // a duplicate layer record: last wins
                    pos += rec;
                }
                // a torn trailing record (kill between write and fsync):
                // dropped below, otherwise the records appended after it
                // would be orphaned behind unparsable bytes
                valid = true;
                valid_len = pos;
                if !scores.is_empty() {
                    println!("experts: score cache {} ({} layers restored)", path.display(), scores.len());
                }
            }
        }
        if !valid {
            println!("experts: ignoring score cache {} (bad magic/version/expert count), rescoring", path.display());
            std::fs::remove_file(&path).ok();
            scores = std::collections::HashMap::new();
        }
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&path).unwrap();
        let len = f.metadata().unwrap().len() as usize;
        if len == 0 {
            let mut hdr = Vec::with_capacity(SCORE_HEADER);
            hdr.extend_from_slice(SCORE_MAGIC);
            hdr.extend_from_slice(&SCORE_VERSION.to_le_bytes());
            hdr.extend_from_slice(&(n_layers as u32).to_le_bytes());
            hdr.extend_from_slice(&(n_experts as u32).to_le_bytes());
            f.write_all(&hdr).unwrap();
            f.sync_data().unwrap();
        } else if valid && valid_len < len {
            f.set_len(valid_len as u64).unwrap();
            println!("experts: score cache torn tail dropped ({} bytes)", len - valid_len);
        }
        ScoreCache { file: std::sync::Mutex::new(f), scores }
    }

    pub(super) fn get(&self, layer: usize) -> Option<&Vec<f64>> {
        self.scores.get(&layer)
    }

    /// Appends one layer record in ONE write + fsync (kill -9 safe).
    pub(super) fn record(&self, layer: usize, scores: &[f64]) {
        use std::io::Write;
        let mut buf = Vec::with_capacity(4 + 8 * scores.len());
        buf.extend_from_slice(&(layer as u32).to_le_bytes());
        for s in scores {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        let mut f = self.file.lock().unwrap();
        f.write_all(&buf).unwrap();
        f.sync_data().unwrap();
    }
}
