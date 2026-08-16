// Qwen conversational state snapshots (save / resume).
//
// Unlike K3 (MKMEM001), the Qwen full-attention layers use rotary
// positions, so the absolute position is part of the state. The linear
// layers carry fixed-size recurrent and convolution states, the
// full-attention layers an append-only key/value history, and the optional
// MTP draft head its own key/value history. The logits after the last
// ingested token are stored so that resuming with an empty prompt is a
// pure continuation.
//
// Adapter packs: the K3 format refuses to save with packs because its
// fingerprint cannot prove their identity; the Qwen runtime knows the
// composed pack-set SHA-256, so it is stored and must match on load.
//
// File layout (integers little-endian, floats little-endian f32):
//   magic       : 8 bytes "MKMEMQW1"
//   fingerprint : u32 n_layers, d, vocab, lin_k_heads, lin_v_heads,
//                 lin_k_dim, lin_v_dim, conv_kernel, n_heads, n_kv_heads,
//                 head_dim, dense_inter, n_experts, mtp_layers
//   adapters    : u8 flag, then 32 bytes pack-set SHA-256 (zero if none)
//   position    : u64 tokens ingested
//   per layer   : u8 kind (0 = linear, 1 = full), then
//                 linear -> state, conv; full -> k, v
//                 (each Vec<f32> as u32 length + raw f32 payload)
//   mtp cache   : k, v (empty vectors when the model has no MTP head)
//   logits      : u32 length + raw f32 payload

use super::memory_pack::{get_fixed, put_vec, Reader};
use crate::model::qwen::{QwenCache, QwenModel};

const MAGIC: &[u8; 8] = b"MKMEMQW1";

fn fingerprint(model: &QwenModel) -> [usize; 14] {
    let c = &model.cfg;
    [
        c.n_layers,
        c.d,
        c.vocab,
        c.lin_k_heads,
        c.lin_v_heads,
        c.lin_k_dim,
        c.lin_v_dim,
        c.conv_kernel,
        c.n_heads,
        c.n_kv_heads,
        c.head_dim,
        c.dense_inter,
        c.n_experts,
        c.mtp_layers,
    ]
}

fn pack_digest_bytes(model: &QwenModel) -> Result<(u8, [u8; 32]), String> {
    let mut digest = [0u8; 32];
    match model.adapter_set_sha256() {
        None => Ok((0, digest)),
        Some(hex) => {
            if hex.len() != 64 {
                return Err("adapter set digest is not a SHA-256".to_string());
            }
            for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
                let hi = (chunk[0] as char).to_digit(16).ok_or("bad digest hex")?;
                let lo = (chunk[1] as char).to_digit(16).ok_or("bad digest hex")?;
                digest[i] = (hi * 16 + lo) as u8;
            }
            Ok((1, digest))
        }
    }
}

/// Serializes the current state + the logits after the last ingested token.
pub fn serialize(model: &QwenModel) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    for v in fingerprint(model) {
        out.extend_from_slice(&(v as u32).to_le_bytes());
    }
    let (flag, digest) = pack_digest_bytes(model)?;
    out.push(flag);
    out.extend_from_slice(&digest);
    out.extend_from_slice(&(model.pos as u64).to_le_bytes());
    for cache in &model.caches {
        match cache {
            QwenCache::Linear(c) => {
                out.push(0);
                put_vec(&mut out, &c.state);
                put_vec(&mut out, &c.conv);
            }
            QwenCache::Full(c) => {
                out.push(1);
                put_vec(&mut out, &c.k);
                put_vec(&mut out, &c.v);
            }
        }
    }
    put_vec(&mut out, &model.mtp_cache.k);
    put_vec(&mut out, &model.mtp_cache.v);
    put_vec(&mut out, &model.last_logits);
    Ok(out)
}

/// Snapshots the state to a file.
pub fn save(model: &QwenModel, path: &str) -> Result<(), String> {
    if model.last_logits.is_empty() {
        return Err("nothing to save: no token has been ingested".to_string());
    }
    let bytes = serialize(model)?;
    std::fs::write(path, bytes).map_err(|e| format!("cannot write {}: {}", path, e))
}

/// Restores the state into `model` and returns the stored logits. Refuses
/// to load when the fingerprint or the adapter-pack set does not match.
pub fn load(model: &mut QwenModel, path: &str) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {}", path, e))?;
    load_slice(model, &bytes, path)
}

/// In-memory variant of `load` (prefix-cache entries embed a MKMEMQW1
/// image); `label` only names the source in error messages.
pub fn load_slice(model: &mut QwenModel, bytes: &[u8], label: &str) -> Result<Vec<f32>, String> {
    let path = label;
    let mut r = Reader { b: bytes, p: 0 };
    if r.take(8)? != MAGIC {
        return Err(format!("{}: not a Qwen .mkmem file (bad magic)", path));
    }
    let want = fingerprint(model);
    let mut got = [0usize; 14];
    for slot in got.iter_mut() {
        *slot = r.u32()? as usize;
    }
    if got != want {
        return Err(format!(
            "{}: this snapshot belongs to a different model (fingerprint {:?} != {:?})",
            path, got, want
        ));
    }
    let (want_flag, want_digest) = pack_digest_bytes(model)?;
    let flag = r.u8()?;
    let digest = r.take(32)?;
    if flag != want_flag || digest != want_digest {
        return Err(format!(
            "{}: snapshot adapter-pack set does not match the loaded packs",
            path
        ));
    }
    let pos = u64::from_le_bytes(r.take(8)?.try_into().unwrap()) as usize;

    let c = model.cfg.clone();
    let kv_width = c.n_kv_heads * c.head_dim;
    let lin_state = c.lin_v_heads * c.lin_k_dim * c.lin_v_dim;
    let conv_dim = (c.lin_k_heads * c.lin_k_dim) * 2 + c.lin_v_heads * c.lin_v_dim;
    let lin_conv = conv_dim * (c.conv_kernel - 1);
    for (l, cache) in model.caches.iter_mut().enumerate() {
        let kind = r.u8()?;
        match cache {
            QwenCache::Linear(cell) => {
                if kind != 0 {
                    return Err(format!("{}: layer {} tagged full, expected linear", path, l));
                }
                cell.state = get_fixed(&mut r, lin_state, path)?;
                cell.conv = get_fixed(&mut r, lin_conv, path)?;
            }
            QwenCache::Full(cell) => {
                if kind != 1 {
                    return Err(format!("{}: layer {} tagged linear, expected full", path, l));
                }
                let k = r.vec_f32()?;
                let v = r.vec_f32()?;
                if k.len() != v.len() || k.len() % kv_width != 0 || k.len() / kv_width != pos {
                    return Err(format!("{}: corrupt full-attention cache at layer {}", path, l));
                }
                cell.len = k.len() / kv_width;
                cell.k = k;
                cell.v = v;
            }
        }
    }
    let mtp_k = r.vec_f32()?;
    let mtp_v = r.vec_f32()?;
    if mtp_k.len() != mtp_v.len() || mtp_k.len() % kv_width != 0 {
        return Err(format!("{}: corrupt MTP cache", path));
    }
    if c.mtp_layers == 0 && !mtp_k.is_empty() {
        return Err(format!("{}: snapshot carries an MTP cache, model has none", path));
    }
    model.mtp_cache.len = mtp_k.len() / kv_width;
    model.mtp_cache.k = mtp_k;
    model.mtp_cache.v = mtp_v;
    let logits = get_fixed(&mut r, c.vocab, path)?;
    if r.p != r.b.len() {
        return Err(format!("{}: trailing bytes after the snapshot", path));
    }
    model.pos = pos;
    model.last_logits = logits.clone();
    Ok(logits)
}
