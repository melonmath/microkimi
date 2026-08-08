// VQ1 expert shadows (--stream-fallback): a 0.5-bit "shadow" of EVERY routed
// MoE expert, fully resident in RAM, served by the stream engine on an expert
// cache miss instead of blocking the decode on the disk tier (stream.rs).
//
//   microkimi shadow --model X.bin [--out X.shadows]
//
// The sidecar (default <model>.shadows) holds, for every MoE layer and every
// expert, the VQ1 index bytes of w1 ++ w2 ++ w3 (1 byte per 16 weights, same
// layout as the sliced cold-VQ1 tensors of slice --cold-vq) plus ONE global
// 256 x 16 f32 codebook. The main .bin format is untouched.
//
// Size: n_moe_layers * n_experts * 3 * (routed_hidden * moe_inter / 16) bytes
// + 16 KB codebook (vs 4.25 bit/weight for the mxfp4 blobs: ~8.5x smaller).
//
// The codebook is trained by Lloyd k-means (quant::train_codebook) on a seeded
// reservoir sample of the dequantized weights of the first SHADOW_TRAIN_LAYERS
// MoE layers (all their experts). Sampling a subset of layers instead of the
// whole model is an approximation: the microquant report showed a single
// global codebook generalizes across layers for the cold tail, and the shadow
// is a latency fallback, not a quality target - the full-precision expert
// replaces it on the next request (see stream.rs). Everything is
// deterministic: fixed seed, fixed iteration order, ties to the lowest index.
//
// A shadowed expert is served with the shadow codebook through the same
// quant::matvec_vq path as sliced cold experts. This is a DEGRADED mode: a
// token computed with a shadow expert is not bit-identical to the
// full-precision one. OFF by default; opt in with --stream-fallback or
// MICROKIMI_STREAM_FALLBACK=1.

use crate::quant::{VQ_DIM, VQ_K};
use crate::weights::{BinFile, DTYPE_MXFP4, DTYPE_MXFP4SQ, DTYPE_VQ1};

/// Sidecar magic.
pub const MAGIC: &[u8; 8] = b"MKSH0001";

/// MoE layers whose experts feed the codebook training sample (a reservoir
/// cap applies on top). Enough to cover the weight distribution, cheap
/// enough that building the shadows of a large model stays I/O bound.
const SHADOW_TRAIN_LAYERS: usize = 4;
/// Reservoir cap in 16-vectors (same scale as the --cold-vq training set).
const SHADOW_SAMPLE_CAP: usize = 300_000;
/// Fixed training seed (deterministic builds).
const SHADOW_SEED: u64 = 0x5EED_5A10_0B15_E5u64;

/// The decoded sidecar, fully resident in RAM.
pub struct Shadows {
    /// MoE layer ids, ascending.
    pub layers: Vec<u32>,
    pub n_experts: usize,
    /// VQ1 index bytes per expert matrix (routed_hidden * moe_inter / VQ_DIM).
    pub vq_blob: usize,
    /// Global codebook, VQ_K * VQ_DIM f32.
    pub cb: Vec<f32>,
    /// Index bytes: per layer (in `layers` order), per expert (0..n_experts):
    /// w1 ++ w2 ++ w3, vq_blob bytes each.
    pub data: Vec<u8>,
}

impl Shadows {
    /// Byte offset of expert `expert` of `layer` in `data` (w1 ++ w2 ++ w3,
    /// 3 * vq_blob bytes), None when the layer is not shadowed.
    pub fn offset(&self, layer: u32, expert: u32) -> Option<usize> {
        let li = self.layers.binary_search(&layer).ok()?;
        if expert as usize >= self.n_experts {
            return None;
        }
        Some((li * self.n_experts + expert as usize) * 3 * self.vq_blob)
    }

    /// Loads and validates a sidecar against the model shape: the MoE layer
    /// set, the expert count and the per-matrix VQ1 blob size must match
    /// exactly (a sidecar built for another model is a hard error, never a
    /// silent degradation).
    pub fn load(path: &str, moe_layers: &[usize], n_experts: usize, vq_blob: usize) -> Shadows {
        let raw = std::fs::read(path).unwrap_or_else(|e| {
            panic!(
                "{} unreadable: {} (build it with `microkimi shadow --model <model.bin>`, or drop --stream-fallback)",
                path, e
            )
        });
        let mut p = 0usize;
        let take = |p: &mut usize, n: usize| -> &[u8] {
            assert!(*p + n <= raw.len(), "{}: truncated sidecar", path);
            let s = &raw[*p..*p + n];
            *p += n;
            s
        };
        assert_eq!(take(&mut p, 8), MAGIC, "{}: bad magic (expected MKSH0001)", path);
        let u32at = |p: &mut usize| -> u32 { u32::from_le_bytes(take(p, 4).try_into().unwrap()) };
        let n_layers = u32at(&mut p) as usize;
        let layers: Vec<u32> = (0..n_layers).map(|_| u32at(&mut p)).collect();
        let f_experts = u32at(&mut p) as usize;
        let f_blob = u32at(&mut p) as usize;
        let mut cb = vec![0f32; VQ_K * VQ_DIM];
        for (i, b) in take(&mut p, VQ_K * VQ_DIM * 4).chunks_exact(4).enumerate() {
            cb[i] = f32::from_le_bytes(b.try_into().unwrap());
        }
        let want_layers: Vec<u32> = moe_layers.iter().map(|&l| l as u32).collect();
        assert_eq!(layers, want_layers, "{}: MoE layer set does not match the model", path);
        assert_eq!(f_experts, n_experts, "{}: expert count {} != model's {}", path, f_experts, n_experts);
        assert_eq!(f_blob, vq_blob, "{}: expert shadow blob {} != model's {}", path, f_blob, vq_blob);
        let rest = raw.len() - p;
        let data = take(&mut p, rest).to_vec();
        assert_eq!(
            data.len(),
            n_layers * n_experts * 3 * vq_blob,
            "{}: shadow data size mismatch (built for another model?)",
            path
        );
        Shadows { layers, n_experts, vq_blob, cb, data }
    }
}

/// Default sidecar path for a model file.
pub fn sidecar_path(model: &str) -> String {
    format!("{}.shadows", model)
}

/// Dequantizes one expert tensor to f32, whatever its storage flavor:
/// mxfp4/mxfp4sq via mxfp4::dequant_any, VQ1 (a --cold-vq sliced model) by
/// gathering the model's own codebook. The shadow re-quantizes from f32 with
/// the shadow codebook either way.
fn dequant_tensor(bin: &BinFile, name: &str, vq1_cb: &Option<Vec<f32>>) -> (Vec<f32>, usize, usize) {
    let e = bin.entries.get(name).unwrap_or_else(|| panic!("missing tensor: {}", name));
    let (r, c) = (e.dims[0] as usize, e.dims[1] as usize);
    let blob = &bin.data[e.offset as usize..(e.offset + e.size) as usize];
    match e.dtype {
        DTYPE_MXFP4 | DTYPE_MXFP4SQ => (crate::mxfp4::dequant_any(e.dtype, blob, r, c), r, c),
        DTYPE_VQ1 => {
            let cb = vq1_cb.as_ref().expect("VQ1 expert tensor without a vq_codebook tensor");
            let mut w = vec![0f32; r * c];
            for (vi, &idx) in blob.iter().enumerate() {
                let src = &cb[idx as usize * VQ_DIM..(idx as usize + 1) * VQ_DIM];
                w[vi * VQ_DIM..(vi + 1) * VQ_DIM].copy_from_slice(src);
            }
            (w, r, c)
        }
        dt => panic!("{}: unexpected expert dtype {} (not mxfp4/vq1)", name, dt),
    }
}

/// `microkimi shadow --model X.bin [--out X.shadows]`: build the VQ1 shadow
/// sidecar of every routed expert (see the module header).
pub fn cmd(args: &[String]) {
    let model = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .unwrap_or_else(|| {
            eprintln!("error: usage: microkimi shadow --model X.bin [--out X.shadows]");
            std::process::exit(1);
        })
        .clone();
    let out = args
        .iter()
        .position(|a| a == "--out")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| sidecar_path(&model));
    build(&model, &out);
}

fn build(model: &str, out: &str) {
    let t0 = std::time::Instant::now();
    let bin = BinFile::open(model);
    let cfg = &bin.config;
    let moe_layers: Vec<usize> = (0..cfg.n_layers).filter(|&l| cfg.is_moe(l)).collect();
    assert!(!moe_layers.is_empty(), "{}: no MoE layers, nothing to shadow", model);
    let n_mat = cfg.routed_hidden * cfg.moe_inter;
    assert_eq!(n_mat % VQ_DIM, 0, "expert matrix {} values not a multiple of VQ_DIM {}", n_mat, VQ_DIM);
    let vq_blob = n_mat / VQ_DIM;
    let n_experts = cfg.n_experts;
    // codebook of a VQ1-sliced source (dequant source for cold experts)
    let vq1_cb: Option<Vec<f32>> = if bin.entries.values().any(|e| e.dtype == DTYPE_VQ1) {
        Some(bin.f32_vec("vq_codebook"))
    } else {
        None
    };

    // ── 1. codebook: seeded reservoir over the first SHADOW_TRAIN_LAYERS MoE
    // layers (all their experts), then Lloyd k-means ──
    let mut rng = crate::quant::Rng::new(SHADOW_SEED);
    let mut res: Vec<f32> = Vec::new();
    let mut seen = 0u64;
    for &l in moe_layers.iter().take(SHADOW_TRAIN_LAYERS) {
        for e in 0..n_experts {
            for wn in ["w1", "w2", "w3"] {
                let (w, _, _) = dequant_tensor(&bin, &format!("layers.{}.block_sparse_moe.experts.{}.{}", l, e, wn), &vq1_cb);
                for v in w.chunks_exact(VQ_DIM) {
                    let t = seen;
                    seen += 1;
                    if (t as usize) < SHADOW_SAMPLE_CAP {
                        res.extend_from_slice(v);
                    } else {
                        let j = rng.below(t as usize + 1);
                        if j < SHADOW_SAMPLE_CAP {
                            res[j * VQ_DIM..(j + 1) * VQ_DIM].copy_from_slice(v);
                        }
                    }
                }
            }
        }
        println!("shadow: sampled layer {} ({} vectors offered so far)", l, seen);
    }
    let t = std::time::Instant::now();
    let cb = crate::quant::train_codebook(&res, SHADOW_SEED);
    println!(
        "shadow: global codebook ({}x{}) trained on {}/{} sampled vectors ({} MoE layers) in {:.1?}",
        VQ_K,
        VQ_DIM,
        res.len() / VQ_DIM,
        seen,
        SHADOW_TRAIN_LAYERS.min(moe_layers.len()),
        t.elapsed()
    );

    // ── 2. quantize every expert (parallel over (layer, expert) pairs) ──
    let mut data = vec![0u8; moe_layers.len() * n_experts * 3 * vq_blob];
    let pairs = moe_layers.len() * n_experts;
    let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(pairs.max(1));
    let chunk = pairs.div_ceil(n_threads);
    let mut data_chunks: Vec<&mut [u8]> = data.chunks_mut(chunk * 3 * vq_blob).collect();
    let binp = &bin;
    let cbp = &cb;
    let vq1p = &vq1_cb;
    let mlp = &moe_layers;
    let err_sums: Vec<f64> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for (ci, dchunk) in data_chunks.drain(..).enumerate() {
            handles.push(s.spawn(move || {
                let mut err_sum = 0f64;
                let first = ci * chunk;
                let last = ((ci + 1) * chunk).min(pairs);
                for p in first..last {
                    let (li, e) = (p / n_experts, p % n_experts);
                    let l = mlp[li];
                    let base = (p - first) * 3 * vq_blob;
                    for (wi, wn) in ["w1", "w2", "w3"].iter().enumerate() {
                        let name = format!("layers.{}.block_sparse_moe.experts.{}.{}", l, e, wn);
                        let (w, _, _) = dequant_tensor(binp, &name, vq1p);
                        let idx = crate::quant::quantize(&w, cbp);
                        err_sum += crate::quant::rel_error(&w, &idx, cbp);
                        dchunk[base + wi * vq_blob..base + (wi + 1) * vq_blob].copy_from_slice(&idx);
                    }
                }
                err_sum
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let n_tensors = (pairs * 3) as f64;
    let mean_err = err_sums.iter().sum::<f64>() / n_tensors;

    // ── 3. write the sidecar ──
    let mut outb: Vec<u8> = Vec::with_capacity(8 + 4 * (3 + moe_layers.len()) + VQ_K * VQ_DIM * 4 + data.len());
    outb.extend_from_slice(MAGIC);
    outb.extend_from_slice(&(moe_layers.len() as u32).to_le_bytes());
    for &l in &moe_layers {
        outb.extend_from_slice(&(l as u32).to_le_bytes());
    }
    outb.extend_from_slice(&(n_experts as u32).to_le_bytes());
    outb.extend_from_slice(&(vq_blob as u32).to_le_bytes());
    for &c in &cb {
        outb.extend_from_slice(&c.to_le_bytes());
    }
    outb.extend_from_slice(&data);
    std::fs::write(out, &outb).unwrap_or_else(|e| panic!("{} unwritable: {}", out, e));
    let mxfp4_bytes: u64 = bin.entries.iter().filter(|(n, _)| crate::weights::is_expert_tensor(n)).map(|(_, e)| e.size).sum();
    println!(
        "shadow: {} MoE layers x {} experts, {} B/expert ({} B/matrix) + 16 KB codebook",
        moe_layers.len(),
        n_experts,
        3 * vq_blob,
        vq_blob
    );
    println!("shadow: mean rel Frobenius error {:.3} over {} quantized matrices", mean_err, n_tensors as u64);
    println!(
        "shadow: wrote {} ({:.1} MB resident in RAM under --stream-fallback, vs {:.1} MB of stored expert bytes: {:.1}x smaller) in {:.1?}",
        out,
        outb.len() as f64 / (1024.0 * 1024.0),
        mxfp4_bytes as f64 / (1024.0 * 1024.0),
        mxfp4_bytes as f64 / outb.len() as f64,
        t0.elapsed()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes a synthetic sidecar for (layers, n_experts, vq_blob) and loads
    /// it back: roundtrip + offset math.
    #[test]
    fn sidecar_roundtrip() {
        let layers = [1usize, 3, 5];
        let (n_experts, vq_blob) = (4, 8);
        let cb = vec![0.5f32; VQ_K * VQ_DIM];
        let data: Vec<u8> = (0..layers.len() * n_experts * 3 * vq_blob).map(|i| (i % 251) as u8).collect();
        let mut raw: Vec<u8> = Vec::new();
        raw.extend_from_slice(MAGIC);
        raw.extend_from_slice(&(layers.len() as u32).to_le_bytes());
        for &l in &layers {
            raw.extend_from_slice(&(l as u32).to_le_bytes());
        }
        raw.extend_from_slice(&(n_experts as u32).to_le_bytes());
        raw.extend_from_slice(&(vq_blob as u32).to_le_bytes());
        for &c in &cb {
            raw.extend_from_slice(&c.to_le_bytes());
        }
        raw.extend_from_slice(&data);
        let path = std::env::temp_dir().join(format!("microkimi-shadow-test-{}", std::process::id()));
        let path = path.to_string_lossy().into_owned();
        std::fs::write(&path, &raw).unwrap();
        let sh = Shadows::load(&path, &layers, n_experts, vq_blob);
        assert_eq!(sh.layers, layers.iter().map(|&l| l as u32).collect::<Vec<_>>());
        assert_eq!(sh.cb, cb);
        assert_eq!(sh.data, data);
        // offset of (layer, expert): (li * n_experts + e) * 3 * vq_blob
        assert_eq!(sh.offset(1, 0), Some(0));
        assert_eq!(sh.offset(1, 3), Some(3 * 3 * vq_blob));
        assert_eq!(sh.offset(3, 0), Some(n_experts * 3 * vq_blob));
        assert_eq!(sh.offset(2, 0), None); // not a MoE layer
        assert_eq!(sh.offset(1, 4), None); // out of expert range
        std::fs::remove_file(&path).ok();
    }

    #[test]
    #[should_panic]
    fn sidecar_rejects_bad_magic() {
        let path = std::env::temp_dir().join(format!("microkimi-shadow-test-bad-{}", std::process::id()));
        let path = path.to_string_lossy().into_owned();
        std::fs::write(&path, b"NOPE0000xxxx").unwrap();
        Shadows::load(&path, &[1], 1, 8);
    }
}
