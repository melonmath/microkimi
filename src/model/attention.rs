// MLA attention (NoPE, latent KV): reference kernel, tiled flash kernel,
// MQA head-sharing variant, optional q8 KV cache with Hadamard rotation.
// mla_prefill fills the MlaCache for a chunk, mla_forward decodes one token.
// Kernel choice is env-gated; every variant must return identical logits.

use super::*;

/// True when MICROKIMI_NO_FLASH=1 (A/B toggle for the flash attention kernel).
fn no_flash() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("MICROKIMI_NO_FLASH").map(|v| v == "1").unwrap_or(false))
}

/// KV tile size of the flash kernel: a tile's scores live in a 64-float
/// stack buffer and its K rows span 64 x 192 f32 = 48 KB (micro dims), L1/L2
/// friendly; 64 also keeps the running-rescale corrections rare enough that
/// the numerics stay within ~1e-6 of the materialized path.
const FLASH_KV: usize = 64;

/// Attention output for one (query, head) over cache positions 0..=pos:
/// materialized-score reference (the historical kernel, kept for the
/// MICROKIMI_NO_FLASH toggle and the selftest A/B). `oh` must be zeroed.
pub(crate) fn mla_attn_ref(cfg: &Config, k: &[f32], v: &[f32], qh: &[f32], h: usize, pos: usize, scale: f32, oh: &mut [f32]) {
    let mut scores = vec![0f32; pos + 1];
    for j in 0..=pos {
        let kj = &k[(j * cfg.mla_heads + h) * cfg.mla_qh()..(j * cfg.mla_heads + h + 1) * cfg.mla_qh()];
        scores[j] = dot(qh, kj) * scale;
    }
    let m = scores.iter().fold(f32::NEG_INFINITY, |a, &x| a.max(x));
    let mut z = 0f32;
    for s in scores.iter_mut() {
        *s = (*s - m).exp();
        z += *s;
    }
    for s in scores.iter_mut() {
        *s /= z;
    }
    for j in 0..=pos {
        let vj = &v[(j * cfg.mla_heads + h) * cfg.mla_v..(j * cfg.mla_heads + h + 1) * cfg.mla_v];
        let p = scores[j];
        for d in 0..cfg.mla_v {
            oh[d] += p * vj[d];
        }
    }
}

/// Flash attention for one (query, head) over cache positions 0..=pos:
/// online softmax in KV tiles of FLASH_KV positions, no score row
/// materialized (a 64-float stack buffer is the only working memory, vs
/// pos+1 floats allocated per (query, head) in the reference). `oh` must be
/// zeroed. Numerics: identical math to mla_attn_ref, different f32
/// association (the accumulator and the normalizer are rescaled whenever the
/// running max grows instead of normalizing once at the end); the deviation
/// is bounded by the selftest A/B (tol 1e-5, measured ~1e-6).
pub(crate) fn mla_attn_flash(cfg: &Config, k: &[f32], v: &[f32], qh: &[f32], h: usize, pos: usize, scale: f32, oh: &mut [f32]) {
    let (hd, vd, nh) = (cfg.mla_qh(), cfg.mla_v, cfg.mla_heads);
    let mut m = f32::NEG_INFINITY; // running max
    let mut l = 0f32; // running normalizer (sum of exp(s - m) so far)
    let mut scores = [0f32; FLASH_KV];
    let mut t = 0usize;
    while t <= pos {
        let end = (t + FLASH_KV - 1).min(pos);
        // tile scores (causal mask = the loop bound; NoPE: nothing positional)
        let mut tm = f32::NEG_INFINITY;
        for (i, j) in (t..=end).enumerate() {
            let kj = &k[(j * nh + h) * hd..(j * nh + h + 1) * hd];
            let s = dot(qh, kj) * scale;
            scores[i] = s;
            tm = tm.max(s);
        }
        let m_new = m.max(tm);
        let corr = (m - m_new).exp(); // 1 when the max did not move, 0 on the first tile
        let mut tile_l = 0f32;
        for s in scores.iter_mut().take(end - t + 1) {
            *s = (*s - m_new).exp();
            tile_l += *s;
        }
        for d in 0..vd {
            oh[d] *= corr;
        }
        l = l * corr + tile_l;
        for (i, j) in (t..=end).enumerate() {
            let vj = &v[(j * nh + h) * vd..(j * nh + h + 1) * vd];
            let p = scores[i];
            for d in 0..vd {
                oh[d] += p * vj[d];
            }
        }
        m = m_new;
        t = end + 1;
    }
    for d in 0..vd {
        oh[d] /= l;
    }
}

/// True when MICROKIMI_NO_MQA=1 (A/B toggle: per-head flash loops instead of
/// the all-heads MQA-style kernels, f32 and q8 alike).
fn no_mqa() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("MICROKIMI_NO_MQA").map(|v| v == "1").unwrap_or(false))
}

/// MQA-style flash attention for ALL heads at once over cache positions
/// 0..=pos. The per-head loop of mla_forward streams the whole KV cache
/// once PER HEAD (each pass strided over the full cache: ~H cold
/// re-traversals of L2/TLB per token). MLA decodes like MQA: the KV row of
/// one position (all heads' slices, contiguous) is consumed by every head
/// together, so the cache is streamed exactly ONCE per token, tile by tile.
///
/// Bit-identical to the per-head mla_attn_flash loop BY CONSTRUCTION: each
/// head keeps its own online-softmax state (m, l, accumulator) and sees the
/// exact same sequence of tiles and operations; only the interleaving
/// across heads changes, and the heads are independent. `q` is [H * qh],
/// `attn` (zeroed) is [H * vd].
pub(crate) fn mla_attn_flash_mqa(cfg: &Config, k: &[f32], v: &[f32], q: &[f32], pos: usize, scale: f32, attn: &mut [f32]) {
    let (hd, vd, nh) = (cfg.mla_qh(), cfg.mla_v, cfg.mla_heads);
    let mut m = vec![f32::NEG_INFINITY; nh]; // running max per head
    let mut l = vec![0f32; nh]; // running normalizer per head
    let mut scores = vec![0f32; nh * FLASH_KV]; // head-major tile scores
    let mut t = 0usize;
    while t <= pos {
        let end = (t + FLASH_KV - 1).min(pos);
        let tn = end - t + 1;
        // scores of every head over the tile: each KV row is read once
        for (i, j) in (t..=end).enumerate() {
            let kj = &k[j * nh * hd..(j + 1) * nh * hd];
            for h in 0..nh {
                scores[h * FLASH_KV + i] = dot(&q[h * hd..(h + 1) * hd], &kj[h * hd..(h + 1) * hd]) * scale;
            }
        }
        // per-head online-softmax update: the exact tile body of
        // mla_attn_flash, same order, same values
        for h in 0..nh {
            let sh = &mut scores[h * FLASH_KV..h * FLASH_KV + tn];
            let tm = sh.iter().fold(f32::NEG_INFINITY, |a, &x| a.max(x));
            let m_new = m[h].max(tm);
            let corr = (m[h] - m_new).exp();
            let mut tile_l = 0f32;
            for s in sh.iter_mut() {
                *s = (*s - m_new).exp();
                tile_l += *s;
            }
            let oh = &mut attn[h * vd..(h + 1) * vd];
            for d in 0..vd {
                oh[d] *= corr;
            }
            l[h] = l[h] * corr + tile_l;
            m[h] = m_new;
        }
        // V accumulation: each V row is read once for all heads
        for (i, j) in (t..=end).enumerate() {
            let vj = &v[j * nh * vd..(j + 1) * nh * vd];
            for h in 0..nh {
                let p = scores[h * FLASH_KV + i];
                let oh = &mut attn[h * vd..(h + 1) * vd];
                let vh = &vj[h * vd..(h + 1) * vd];
                for d in 0..vd {
                    oh[d] += p * vh[d];
                }
            }
        }
        t = end + 1;
    }
    for h in 0..nh {
        for d in 0..vd {
            attn[h * vd + d] /= l[h];
        }
    }
}

// ── MLA KV cache: f32 rows or q8_0 (latent quantized, rope f32) ──
//
// The MLA cache is the only engine state that grows with the context (KDA
// states are fixed-size). By default it is stored q8_0: the latent (nope)
// part of K and all of V are quantized per block of 32 (i8 + one f32
// scale), the rope part of K stays f32 (position-sensitive; a farm
// reference confirms rope must not be quantized) and is stored ONCE per
// position instead of duplicated per head. Bytes per position per layer:
//   f32: H*(nope+v)*4 + (H-1)*rope*4 (rope duplicated)   [micro: 5248]
//   q8:  H*(nope+v) + H*(nope+v)/32*4 + rope*4           [micro: 1936, ÷2.7]
// The Q.K dot of the latent part runs in INTEGER (q8 query x q8 cache row,
// q8::block_dot_i8); the rope dot, the softmax and the V accumulation stay
// f32 (V rows are dequantized tile by tile, never in full). Quantization
// happens at append (MlaCache::push). MICROKIMI_NO_KVQ8=1 restores the f32
// cache. MICROKIMI_KV_HADAMARD=1 rotates the latent K and V rows with an
// unnormalized 64-point Walsh-Hadamard transform before quantization
// (smearing outliers over the block makes q8 near-lossless; measured in
// tools::selftest::run_kvq8) and inverts the rotation at read: H = H^T with
// H.H = 64 I, so one butterfly routine serves both directions.

/// True when the MLA KV cache quantizes to q8_0 (default).
/// MICROKIMI_NO_KVQ8=1 keeps the historical f32 cache.
fn kvq8_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MICROKIMI_NO_KVQ8").map(|v| v != "1").unwrap_or(true))
}

/// True when MICROKIMI_KV_HADAMARD=1 (Hadamard rotation before KV
/// quantization; default off - the measured gain on nanokimi is marginal,
/// see tools::selftest::run_kvq8).
fn kv_hadamard() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MICROKIMI_KV_HADAMARD").map(|v| v == "1").unwrap_or(false))
}

/// Unnormalized 64-point Walsh-Hadamard butterfly, in place (H = H^T and
/// H.H = 64 I: applying it twice with a 1/64 scale is the identity, so this
/// one routine rotates before quantization and de-rotates after).
pub(crate) fn hadamard64(x: &mut [f32]) {
    debug_assert_eq!(x.len(), 64);
    let mut h = 1usize;
    while h < 64 {
        for i in (0..64).step_by(2 * h) {
            for j in i..i + h {
                let a = x[j];
                let b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
}

impl MlaCache {
    pub(super) fn new() -> MlaCache {
        MlaCache {
            k: Vec::new(),
            v: Vec::new(),
            kq: Vec::new(),
            ks: Vec::new(),
            kr: Vec::new(),
            vq: Vec::new(),
            vs: Vec::new(),
            q8: kvq8_on(),
            had: kv_hadamard(),
        }
    }

    /// Positions held by the cache.
    pub(super) fn positions(&self, cfg: &Config) -> usize {
        if self.q8 {
            self.kr.len() / cfg.mla_rope
        } else {
            self.k.len() / (cfg.mla_heads * cfg.mla_qh())
        }
    }

    /// Appends one position's rows (k_row: H x (nope+rope) with the latent
    /// part first per head, v_row: H x v) - the f32 layout the attention
    /// builders produce. Quantizes on the fly in q8 mode.
    pub(crate) fn push(&mut self, cfg: &Config, k_row: &[f32], v_row: &[f32]) {
        if !self.q8 {
            self.k.extend_from_slice(k_row);
            self.v.extend_from_slice(v_row);
            return;
        }
        let (nh, nope, rope, vd) = (cfg.mla_heads, cfg.mla_nope, cfg.mla_rope, cfg.mla_v);
        let had = self.had && nope % 64 == 0 && vd % 64 == 0;
        let qh = nope + rope;
        // rope is shared across heads: stored once, from head 0
        self.kr.extend_from_slice(&k_row[nope..qh]);
        let mut scratch = vec![0f32; nope.max(vd)];
        let mut qv = crate::quant::q8::Q8Vec::new();
        for h in 0..nh {
            let row = &k_row[h * qh..h * qh + nope];
            scratch[..nope].copy_from_slice(row);
            if had {
                for b in scratch[..nope].chunks_mut(64) {
                    hadamard64(b);
                }
            }
            crate::quant::q8::quantize_q8_into(&scratch[..nope], &mut qv);
            self.kq.extend_from_slice(&qv.q);
            self.ks.extend_from_slice(&qv.scales);
            let rowv = &v_row[h * vd..(h + 1) * vd];
            scratch[..vd].copy_from_slice(rowv);
            if had {
                for b in scratch[..vd].chunks_mut(64) {
                    hadamard64(b);
                }
            }
            crate::quant::q8::quantize_q8_into(&scratch[..vd], &mut qv);
            self.vq.extend_from_slice(&qv.q);
            self.vs.extend_from_slice(&qv.scales);
        }
    }

    /// Dequantizes back to the f32 row layout (mkmem snapshot; the .mkmem
    /// format stays f32 whatever the runtime cache mode is). In f32 mode
    /// this is a plain clone.
    pub(crate) fn to_f32(&self, cfg: &Config) -> (Vec<f32>, Vec<f32>) {
        if !self.q8 {
            return (self.k.clone(), self.v.clone());
        }
        let (nh, nope, rope, vd) = (cfg.mla_heads, cfg.mla_nope, cfg.mla_rope, cfg.mla_v);
        let had = self.had && nope % 64 == 0 && vd % 64 == 0;
        let qh = nope + rope;
        let n = self.positions(cfg);
        let nb_n = nope / 32;
        let nb_v = vd / 32;
        let mut k = vec![0f32; n * nh * qh];
        let mut v = vec![0f32; n * nh * vd];
        for j in 0..n {
            for h in 0..nh {
                // latent K: dequant (+ de-rotate), then rope from the shared row
                let base_q = (j * nh + h) * nope;
                let base_s = (j * nh + h) * nb_n;
                let out = &mut k[(j * nh + h) * qh..(j * nh + h) * qh + nope];
                for g in 0..nb_n {
                    let s = self.ks[base_s + g];
                    for i in 0..32 {
                        out[g * 32 + i] = self.kq[base_q + g * 32 + i] as f32 * s;
                    }
                }
                if had {
                    for b in out.chunks_mut(64) {
                        hadamard64(b);
                        for x in b.iter_mut() {
                            *x /= 64.0;
                        }
                    }
                }
                k[(j * nh + h) * qh + nope..(j * nh + h) * qh + qh].copy_from_slice(&self.kr[j * rope..(j + 1) * rope]);
                let base_q = (j * nh + h) * vd;
                let base_s = (j * nh + h) * nb_v;
                let out = &mut v[(j * nh + h) * vd..(j * nh + h) * vd + vd];
                for g in 0..nb_v {
                    let s = self.vs[base_s + g];
                    for i in 0..32 {
                        out[g * 32 + i] = self.vq[base_q + g * 32 + i] as f32 * s;
                    }
                }
                if had {
                    for b in out.chunks_mut(64) {
                        hadamard64(b);
                        for x in b.iter_mut() {
                            *x /= 64.0;
                        }
                    }
                }
            }
        }
        (k, v)
    }

    /// Replaces the cache contents by f32 rows (mkmem restore; requantizes
    /// when the runtime cache is in q8 mode).
    pub(crate) fn assign_f32(&mut self, cfg: &Config, k: Vec<f32>, v: Vec<f32>) {
        *self = MlaCache::new();
        if !self.q8 {
            self.k = k;
            self.v = v;
            return;
        }
        let (nh, qh, vd) = (cfg.mla_heads, cfg.mla_qh(), cfg.mla_v);
        let n = k.len() / (nh * qh);
        for j in 0..n {
            self.push(cfg, &k[j * nh * qh..(j + 1) * nh * qh], &v[j * nh * vd..(j + 1) * nh * vd]);
        }
    }
}

/// Flash attention over the q8_0 KV cache for one (query, head), positions
/// 0..=pos. Same online-softmax tile structure as mla_attn_flash; only the
/// dot inputs differ: the latent part of the score runs in INTEGER (q8
/// query x q8 K row via q8::block_dot_i8, per-block scale product), the rope
/// part is an f32 dot on the shared rope rows. V rows are dequantized per
/// tile (never in full). With the Hadamard option the query is rotated once
/// (dot(Hq, Hk) = 64 * dot(q, k), folded back as 1/64) and the V accumulator
/// is de-rotated once at the end (linearity: sum of p.V in the Hadamard
/// domain, then H./64). `oh` must be zeroed. NOT bit-identical to the f32
/// path: the q8 rounding of K, V and the query is the deal (measured in
/// tools::selftest::run_kvq8, max rel << 1e-3).
pub(crate) fn mla_attn_flash_q8(cfg: &Config, c: &MlaCache, qh: &[f32], h: usize, pos: usize, scale: f32, oh: &mut [f32]) {
    let (nh, nope, rope, vd) = (cfg.mla_heads, cfg.mla_nope, cfg.mla_rope, cfg.mla_v);
    let had = c.had && nope % 64 == 0 && vd % 64 == 0;
    let nb = nope / 32;
    // query prep: latent part (Hadamard-rotated when enabled) quantized to q8
    let mut qnope = qh[..nope].to_vec();
    if had {
        for b in qnope.chunks_mut(64) {
            hadamard64(b);
        }
    }
    let qq = crate::quant::q8::quantize_q8(&qnope);
    let q_rope = &qh[nope..];
    let had_k = if had { 1.0 / 64.0 } else { 1.0 };
    let mut m = f32::NEG_INFINITY;
    let mut l = 0f32;
    let mut scores = [0f32; FLASH_KV];
    let mut vtile = vec![0f32; FLASH_KV * vd];
    let mut t = 0usize;
    while t <= pos {
        let end = (t + FLASH_KV - 1).min(pos);
        let tn = end - t + 1;
        let mut tm = f32::NEG_INFINITY;
        for (i, j) in (t..=end).enumerate() {
            // rope dot (f32) + latent dot (integer, per-block scale product)
            let mut s = dot(q_rope, &c.kr[j * rope..(j + 1) * rope]);
            let mut acc = 0f32;
            let bq = (j * nh + h) * nope;
            let bs = (j * nh + h) * nb;
            for g in 0..nb {
                let d = crate::quant::q8::block_dot_i8(&c.kq[bq + g * 32..bq + g * 32 + 32], &qq.q[g * 32..g * 32 + 32]);
                acc += qq.scales[g] * c.ks[bs + g] * d as f32;
            }
            s += had_k * acc;
            let s = s * scale;
            scores[i] = s;
            tm = tm.max(s);
        }
        let m_new = m.max(tm);
        let corr = (m - m_new).exp();
        let mut tile_l = 0f32;
        for s in scores.iter_mut().take(tn) {
            *s = (*s - m_new).exp();
            tile_l += *s;
        }
        for d in 0..vd {
            oh[d] *= corr;
        }
        l = l * corr + tile_l;
        // dequant the tile's V rows (Hadamard domain when enabled: the
        // accumulator is de-rotated once at the end)
        for (i, j) in (t..=end).enumerate() {
            let bq = (j * nh + h) * vd;
            let bs = (j * nh + h) * (vd / 32);
            let out = &mut vtile[i * vd..(i + 1) * vd];
            for g in 0..vd / 32 {
                let sc = c.vs[bs + g];
                for d2 in 0..32 {
                    out[g * 32 + d2] = c.vq[bq + g * 32 + d2] as f32 * sc;
                }
            }
            let p = scores[i];
            for d in 0..vd {
                oh[d] += p * out[d];
            }
        }
        m = m_new;
        t = end + 1;
    }
    if had {
        for b in oh[..vd].chunks_mut(64) {
            hadamard64(b);
            for x in b.iter_mut() {
                *x /= 64.0;
            }
        }
    }
    for d in 0..vd {
        oh[d] /= l;
    }
}

/// MQA-style flash attention over the q8_0 KV cache for ALL heads at once:
/// the tile-outer / head-inner restructure of mla_attn_flash_q8, exactly
/// like mla_attn_flash_mqa is for mla_attn_flash. The per-head q8 loop
/// re-streams the whole quantized cache once PER HEAD; here one position's
/// row (all heads' latent slices contiguous in kq/ks, the shared rope row,
/// then all heads' V slices in vq/vs) is consumed while it is hot, so the
/// cache is streamed exactly ONCE per token. The integer latent dot is
/// unchanged (q8 query x q8 cache row via q8::block_dot_i8, per-block scale
/// product): for a fixed position the head-inner loop keeps every head's q8
/// query (H * nope bytes, L1-resident) against the streamed row.
///
/// Bit-identical to the per-head mla_attn_flash_q8 loop BY CONSTRUCTION:
/// each head keeps its own online-softmax state and sees the exact same
/// tile sequence, the same integer dots (exact int32) and the same f32
/// operations in the same order; only the interleaving across independent
/// heads changes. `q` is [H * (nope+rope)], `attn` (zeroed) is [H * vd].
pub(crate) fn mla_attn_flash_q8_mqa(cfg: &Config, c: &MlaCache, q: &[f32], pos: usize, scale: f32, attn: &mut [f32]) {
    let (nh, nope, rope, vd) = (cfg.mla_heads, cfg.mla_nope, cfg.mla_rope, cfg.mla_v);
    let had = c.had && nope % 64 == 0 && vd % 64 == 0;
    let hd = nope + rope;
    let nb = nope / 32;
    let nbv = vd / 32;
    // query prep, per head: latent part (Hadamard-rotated when enabled)
    // quantized to q8 - the exact prep of mla_attn_flash_q8
    let mut qqs = Vec::with_capacity(nh);
    for h in 0..nh {
        let mut qnope = q[h * hd..h * hd + nope].to_vec();
        if had {
            for b in qnope.chunks_mut(64) {
                hadamard64(b);
            }
        }
        qqs.push(crate::quant::q8::quantize_q8(&qnope));
    }
    let had_k = if had { 1.0 / 64.0 } else { 1.0 };
    let mut m = vec![f32::NEG_INFINITY; nh]; // running max per head
    let mut l = vec![0f32; nh]; // running normalizer per head
    let mut scores = vec![0f32; nh * FLASH_KV]; // head-major tile scores
    let mut vtile = vec![0f32; vd]; // one head's dequantized V row
    let mut t = 0usize;
    while t <= pos {
        let end = (t + FLASH_KV - 1).min(pos);
        let tn = end - t + 1;
        // scores of every head over the tile: the position's K row (rope
        // shared + all latent slices, contiguous) is read once
        for (i, j) in (t..=end).enumerate() {
            let kr = &c.kr[j * rope..(j + 1) * rope];
            let kq = &c.kq[j * nh * nope..(j + 1) * nh * nope];
            let ks = &c.ks[j * nh * nb..(j + 1) * nh * nb];
            for h in 0..nh {
                // rope dot (f32) + latent dot (integer, per-block scale
                // product): the same operations, in the same order, as
                // mla_attn_flash_q8
                let mut s = dot(&q[h * hd + nope..(h + 1) * hd], kr);
                let qq = &qqs[h];
                let mut acc = 0f32;
                for g in 0..nb {
                    let d = crate::quant::q8::block_dot_i8(&kq[h * nope + g * 32..h * nope + g * 32 + 32], &qq.q[g * 32..g * 32 + 32]);
                    acc += qq.scales[g] * ks[h * nb + g] * d as f32;
                }
                s += had_k * acc;
                scores[h * FLASH_KV + i] = s * scale;
            }
        }
        // per-head online-softmax update: the exact tile body of
        // mla_attn_flash_q8, same order, same values
        for h in 0..nh {
            let sh = &mut scores[h * FLASH_KV..h * FLASH_KV + tn];
            let tm = sh.iter().fold(f32::NEG_INFINITY, |a, &x| a.max(x));
            let m_new = m[h].max(tm);
            let corr = (m[h] - m_new).exp();
            let mut tile_l = 0f32;
            for s in sh.iter_mut() {
                *s = (*s - m_new).exp();
                tile_l += *s;
            }
            let oh = &mut attn[h * vd..(h + 1) * vd];
            for d in 0..vd {
                oh[d] *= corr;
            }
            l[h] = l[h] * corr + tile_l;
            m[h] = m_new;
        }
        // V accumulation: the position's V row (all heads' slices,
        // contiguous) is read once, dequantized head by head (Hadamard
        // domain when enabled: the accumulators are de-rotated at the end)
        for (i, j) in (t..=end).enumerate() {
            let vq = &c.vq[j * nh * vd..(j + 1) * nh * vd];
            let vs = &c.vs[j * nh * nbv..(j + 1) * nh * nbv];
            for h in 0..nh {
                for g in 0..nbv {
                    let sc = vs[h * nbv + g];
                    for d2 in 0..32 {
                        vtile[g * 32 + d2] = vq[h * vd + g * 32 + d2] as f32 * sc;
                    }
                }
                let p = scores[h * FLASH_KV + i];
                let oh = &mut attn[h * vd..(h + 1) * vd];
                for d in 0..vd {
                    oh[d] += p * vtile[d];
                }
            }
        }
        t = end + 1;
    }
    if had {
        for h in 0..nh {
            for b in attn[h * vd..(h + 1) * vd].chunks_mut(64) {
                hadamard64(b);
                for x in b.iter_mut() {
                    *x /= 64.0;
                }
            }
        }
    }
    for h in 0..nh {
        for d in 0..vd {
            attn[h * vd + d] /= l[h];
        }
    }
}

/// Materialized-score reference over the q8_0 cache (MICROKIMI_NO_FLASH
/// debug path): dequantizes the cache (to_f32) and runs the historical
/// three-pass structure. Same q8 rounding as mla_attn_flash_q8, same f32
/// reassociation as mla_attn_ref.
pub(crate) fn mla_attn_ref_q8(cfg: &Config, c: &MlaCache, qh: &[f32], h: usize, pos: usize, scale: f32, oh: &mut [f32]) {
    let (nh, hd, vd) = (cfg.mla_heads, cfg.mla_qh(), cfg.mla_v);
    let (k, v) = c.to_f32(cfg);
    let mut scores = vec![0f32; pos + 1];
    for j in 0..=pos {
        scores[j] = dot(qh, &k[(j * nh + h) * hd..(j * nh + h + 1) * hd]) * scale;
    }
    let mx = scores.iter().fold(f32::NEG_INFINITY, |a, &x| a.max(x));
    let mut z = 0f32;
    for s in scores.iter_mut() {
        *s = (*s - mx).exp();
        z += *s;
    }
    for s in scores.iter_mut() {
        *s /= z;
    }
    for j in 0..=pos {
        let vj = &v[(j * nh + h) * vd..(j * nh + h + 1) * vd];
        let p = scores[j];
        for d in 0..vd {
            oh[d] += p * vj[d];
        }
    }
}

pub(super) fn mla_forward(
    cfg: &Config,
    data: &[u8],
    w: &MlaW,
    cache: &mut MlaCache,
    x: &[f32],
    prof: &mut Prof,
) -> Vec<f32> {
    let tm = Instant::now();
    // q = q_b(rmsnorm(q_a(x))) [H*(nope+rope)]
    let mut qa = vec![0f32; cfg.mla_qa];
    matvec(Model::t(data, &w.q_a), cfg.mla_qa, cfg.d, x, &mut qa);
    let mut qa_n = vec![0f32; cfg.mla_qa];
    rmsnorm(cfg, &qa, Model::t(data, &w.q_a_ln), &mut qa_n);
    let mut q = vec![0f32; cfg.mla_qb()];
    matvec(Model::t(data, &w.q_b), cfg.mla_qb(), cfg.mla_qa, &qa_n, &mut q);
    // c = kv_a(x) [kva+rope] ; k_pass [kva] ; k_rot [rope] (shared across heads)
    let mut c = vec![0f32; cfg.mla_c_dim()];
    matvec(Model::t(data, &w.kv_a), cfg.mla_c_dim(), cfg.d, x, &mut c);
    let k_rot: Vec<f32> = c[cfg.mla_kva..cfg.mla_kva + cfg.mla_rope].to_vec();
    let mut kp_n = vec![0f32; cfg.mla_kva];
    rmsnorm(cfg, &c[..cfg.mla_kva], Model::t(data, &w.kv_a_ln), &mut kp_n);
    let mut kb = vec![0f32; cfg.mla_kvb()];
    matvec(Model::t(data, &w.kv_b), cfg.mla_kvb(), cfg.mla_kva, &kp_n, &mut kb);
    // K[h] = kb[h][..nope] ++ k_rot ; V[h] = kb[h][nope..nope+v]
    let mut k_new = vec![0f32; cfg.mla_heads * cfg.mla_qh()];
    let mut v_new = vec![0f32; cfg.mla_hv()];
    for h in 0..cfg.mla_heads {
        k_new[h * cfg.mla_qh()..h * cfg.mla_qh() + cfg.mla_nope]
            .copy_from_slice(&kb[h * (cfg.mla_nope + cfg.mla_v)..h * (cfg.mla_nope + cfg.mla_v) + cfg.mla_nope]);
        k_new[h * cfg.mla_qh() + cfg.mla_nope..(h + 1) * cfg.mla_qh()].copy_from_slice(&k_rot);
        v_new[h * cfg.mla_v..(h + 1) * cfg.mla_v].copy_from_slice(
            &kb[h * (cfg.mla_nope + cfg.mla_v) + cfg.mla_nope..(h + 1) * (cfg.mla_nope + cfg.mla_v)],
        );
    }
    cache.push(cfg, &k_new, &v_new);
    let pos = cache.positions(cfg) - 1;
    // causal attention, scale (nope+rope)^-0.5
    let scale = (cfg.mla_qh() as f32).powf(-0.5);
    let mut attn = vec![0f32; cfg.mla_heads * cfg.mla_v];
    let flash = !no_flash();
    if cache.q8 {
        if flash && !no_mqa() {
            // q8_0 cache, MQA-style: all heads together, the quantized
            // cache is streamed once (integer latent dot, bit-identical
            // to the per-head q8 loop)
            mla_attn_flash_q8_mqa(cfg, cache, &q, pos, scale, &mut attn);
        } else {
            // q8_0 cache: per-head kernels with the integer latent dot
            for h in 0..cfg.mla_heads {
                let qh = &q[h * cfg.mla_qh()..(h + 1) * cfg.mla_qh()];
                let oh = &mut attn[h * cfg.mla_v..(h + 1) * cfg.mla_v];
                if flash {
                    mla_attn_flash_q8(cfg, cache, qh, h, pos, scale, oh);
                } else {
                    mla_attn_ref_q8(cfg, cache, qh, h, pos, scale, oh);
                }
            }
        }
    } else if flash && !no_mqa() {
        // MQA-style: all heads together, the KV cache is streamed once
        mla_attn_flash_mqa(cfg, &cache.k, &cache.v, &q, pos, scale, &mut attn);
    } else {
        for h in 0..cfg.mla_heads {
            let qh = &q[h * cfg.mla_qh()..(h + 1) * cfg.mla_qh()];
            let oh = &mut attn[h * cfg.mla_v..(h + 1) * cfg.mla_v];
            if flash {
                mla_attn_flash(cfg, &cache.k, &cache.v, qh, h, pos, scale, oh);
            } else {
                mla_attn_ref(cfg, &cache.k, &cache.v, qh, h, pos, scale, oh);
            }
        }
    }
    // output gate + o_proj (g_proj is [H*v, d]: H*v == d only in the micro
    // config; real K3 is [12288, 7168])
    let hv = cfg.mla_hv();
    let mut g = vec![0f32; hv];
    matvec(Model::t(data, &w.g_proj), hv, cfg.d, x, &mut g);
    for i in 0..hv {
        attn[i] *= sigmoid(g[i]);
    }
    let mut out = vec![0f32; cfg.d];
    matvec(Model::t(data, &w.o_proj), cfg.d, hv, &attn, &mut out);
    prof.t_mla += tm.elapsed().as_secs_f64();
    out
}

/// Batched MLA for prefill: `x` = n position rows [n * d], returns [n * d].
/// Projections run as gemm_batch; the n new latent K/V rows are appended to
/// the cache in position order (identical layout to n single-token calls),
/// then attention runs per query position over the cache entries 0..=pos
/// (parallel causal attention, NoPE: no positional encoding anywhere).
/// Bit-identical to mla_forward per position.
#[allow(clippy::too_many_arguments)]
pub(super) fn mla_prefill(
    cfg: &Config,
    data: &[u8],
    w: &MlaW,
    cache: &mut MlaCache,
    x: &[f32],
    n: usize,
    prof: &mut Prof,
) -> Vec<f32> {
    let tm = Instant::now();
    let (qa_dim, qb, kvb, c_dim) = (cfg.mla_qa, cfg.mla_qb(), cfg.mla_kvb(), cfg.mla_c_dim());
    // q = q_b(rmsnorm(q_a(x))) for all positions
    let mut qa = vec![0f32; n * qa_dim];
    gemm_batch(Model::t(data, &w.q_a), qa_dim, cfg.d, x, n, &mut qa);
    let mut qa_n = vec![0f32; n * qa_dim];
    for t in 0..n {
        rmsnorm(cfg, &qa[t * qa_dim..(t + 1) * qa_dim], Model::t(data, &w.q_a_ln), &mut qa_n[t * qa_dim..(t + 1) * qa_dim]);
    }
    let mut q = vec![0f32; n * qb];
    gemm_batch(Model::t(data, &w.q_b), qb, qa_dim, &qa_n, n, &mut q);
    // c = kv_a(x) [kva+rope] ; k_rot = c[kva..kva+rope] ; kp_n = rmsnorm(c[..kva])
    let mut c = vec![0f32; n * c_dim];
    gemm_batch(Model::t(data, &w.kv_a), c_dim, cfg.d, x, n, &mut c);
    let mut kp_n = vec![0f32; n * cfg.mla_kva];
    for t in 0..n {
        rmsnorm(cfg, &c[t * c_dim..t * c_dim + cfg.mla_kva], Model::t(data, &w.kv_a_ln), &mut kp_n[t * cfg.mla_kva..(t + 1) * cfg.mla_kva]);
    }
    let mut kb = vec![0f32; n * kvb];
    gemm_batch(Model::t(data, &w.kv_b), kvb, cfg.mla_kva, &kp_n, n, &mut kb);
    // build K[h] = kb[h][..nope] ++ k_rot ; V[h] = kb[h][nope..nope+v] per
    // position and append in order: same cache state as the sequential path
    let p0 = cache.positions(cfg);
    for t in 0..n {
        let k_rot = &c[t * c_dim + cfg.mla_kva..t * c_dim + cfg.mla_kva + cfg.mla_rope];
        let kbt = &kb[t * kvb..(t + 1) * kvb];
        let mut k_new = vec![0f32; cfg.mla_heads * cfg.mla_qh()];
        let mut v_new = vec![0f32; cfg.mla_heads * cfg.mla_v];
        for h in 0..cfg.mla_heads {
            k_new[h * cfg.mla_qh()..h * cfg.mla_qh() + cfg.mla_nope]
                .copy_from_slice(&kbt[h * (cfg.mla_nope + cfg.mla_v)..h * (cfg.mla_nope + cfg.mla_v) + cfg.mla_nope]);
            k_new[h * cfg.mla_qh() + cfg.mla_nope..(h + 1) * cfg.mla_qh()].copy_from_slice(k_rot);
            v_new[h * cfg.mla_v..(h + 1) * cfg.mla_v].copy_from_slice(
                &kbt[h * (cfg.mla_nope + cfg.mla_v) + cfg.mla_nope..(h + 1) * (cfg.mla_nope + cfg.mla_v)],
            );
        }
        cache.push(cfg, &k_new, &v_new);
    }
    // causal attention per query position, scale qh^-0.5
    let scale = (cfg.mla_qh() as f32).powf(-0.5);
    let mut attn = vec![0f32; n * cfg.mla_heads * cfg.mla_v];
    let flash = !no_flash();
    for t in 0..n {
        let pos = p0 + t;
        let qt = &q[t * qb..(t + 1) * qb];
        for h in 0..cfg.mla_heads {
            let qh = &qt[h * cfg.mla_qh()..(h + 1) * cfg.mla_qh()];
            let oh = &mut attn[(t * cfg.mla_heads + h) * cfg.mla_v..(t * cfg.mla_heads + h + 1) * cfg.mla_v];
            if cache.q8 {
                if flash {
                    mla_attn_flash_q8(cfg, cache, qh, h, pos, scale, oh);
                } else {
                    mla_attn_ref_q8(cfg, cache, qh, h, pos, scale, oh);
                }
            } else if flash {
                mla_attn_flash(cfg, &cache.k, &cache.v, qh, h, pos, scale, oh);
            } else {
                mla_attn_ref(cfg, &cache.k, &cache.v, qh, h, pos, scale, oh);
            }
        }
    }
    // output gate + o_proj (g_proj is [H*v, d]: H*v == d only in the micro
    // config; real K3 is [12288, 7168])
    let hv = cfg.mla_hv();
    let mut g = vec![0f32; n * hv];
    gemm_batch(Model::t(data, &w.g_proj), hv, cfg.d, x, n, &mut g);
    for i in 0..n * hv {
        attn[i] *= sigmoid(g[i]);
    }
    let mut out = vec![0f32; n * cfg.d];
    gemm_batch(Model::t(data, &w.o_proj), cfg.d, hv, &attn, n, &mut out);
    prof.t_mla += tm.elapsed().as_secs_f64();
    out
}

// ── MoE ──
