// mkmem - .mkmem conversational state snapshots (save / resume / fork).
//
// K3 has no positional encoding anywhere (KDA uses learned decay, MLA is
// NoPE), so the layer caches alone fully determine the continuation: no
// position needs to be stored. The logits after the last ingested token are
// stored too, so that resuming with an empty prompt is a pure continuation
// (no token has to be re-ingested).
//
// File layout (integers little-endian, floats little-endian f32):
//   magic       : 8 bytes "MKMEM001"
//   fingerprint : u32 n_layers, u32 d, u32 kda_heads, u32 kda_dim, u32 vocab,
//                 then n_layers x u8 layer kind (0 = KDA, 1 = MLA)
//   per layer   : KDA -> u8 0, then conv_q, conv_k, conv_v, s
//                 MLA -> u8 1, then k, v
//                 (each Vec<f32> as u32 length + raw f32 payload)
//   logits      : u32 length + raw f32 payload (model vocab)

use crate::model::{Cache, Model};

const MAGIC: &[u8; 8] = b"MKMEM001";

/// Serializes the layer caches + the logits after the last ingested token.
pub fn save(model: &Model, logits: &[f32], path: &str) -> std::io::Result<()> {
    let cfg = &model.cfg;
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    for v in [cfg.n_layers, cfg.d, cfg.kda_heads, cfg.kda_dim, cfg.vocab] {
        out.extend_from_slice(&(v as u32).to_le_bytes());
    }
    for l in 0..cfg.n_layers {
        out.push(cfg.is_mla(l) as u8);
    }
    for c in &model.caches {
        match c {
            Cache::Kda(k) => {
                out.push(0);
                put_vec(&mut out, &k.conv_q);
                put_vec(&mut out, &k.conv_k);
                put_vec(&mut out, &k.conv_v);
                put_vec(&mut out, &k.s);
            }
            Cache::Mla(m) => {
                out.push(1);
                put_vec(&mut out, &m.k);
                put_vec(&mut out, &m.v);
            }
        }
    }
    put_vec(&mut out, logits);
    std::fs::write(path, out)
}

/// Restores the caches into `model` and returns the stored logits.
/// Refuses to load when the fingerprint does not match the loaded model.
pub fn load(model: &mut Model, path: &str) -> Result<Vec<f32>, String> {
    let b = std::fs::read(path).map_err(|e| format!("cannot read {}: {}", path, e))?;
    let mut r = Reader { b: &b, p: 0 };
    if r.take(8)? != MAGIC {
        return Err(format!("{}: not a .mkmem file (bad magic)", path));
    }
    let cfg = &model.cfg;
    let fp = [r.u32()? as usize, r.u32()? as usize, r.u32()? as usize, r.u32()? as usize, r.u32()? as usize];
    let want = [cfg.n_layers, cfg.d, cfg.kda_heads, cfg.kda_dim, cfg.vocab];
    let kinds = r.take(cfg.n_layers)?.to_vec();
    let kinds_match = (0..cfg.n_layers).all(|l| kinds[l] == cfg.is_mla(l) as u8);
    if fp != want || !kinds_match {
        return Err(format!(
            "{}: this .mkmem belongs to a different model (fingerprint n_layers={} d={} kda_heads={} kda_dim={} vocab={} does not match the loaded model)",
            path, fp[0], fp[1], fp[2], fp[3], fp[4]
        ));
    }
    for c in &mut model.caches {
        match c {
            Cache::Kda(k) => {
                if r.u8()? != 0 {
                    return Err(format!("{}: corrupt .mkmem (KDA layer tagged as MLA)", path));
                }
                let conv_len = 3 * cfg.kda_proj();
                k.conv_q = get_fixed(&mut r, conv_len, path)?;
                k.conv_k = get_fixed(&mut r, conv_len, path)?;
                k.conv_v = get_fixed(&mut r, conv_len, path)?;
                k.s = get_fixed(&mut r, cfg.kda_heads * cfg.kda_dim * cfg.kda_dim, path)?;
            }
            Cache::Mla(m) => {
                if r.u8()? != 1 {
                    return Err(format!("{}: corrupt .mkmem (MLA layer tagged as KDA)", path));
                }
                m.k = r.vec_f32()?;
                m.v = r.vec_f32()?;
            }
        }
    }
    r.vec_f32()
}

// ── merge (experimental): KDA state additivity ──
//
// `microkimi mkmem-merge A.mkmem B.mkmem [C.mkmem ...] --out AB.mkmem` writes
// a standard .mkmem that loads with the normal path. Design choice: the KDA
// recurrent state s is the only part that is a true accumulator (the
// recurrence S += (beta k) x delta makes it a sum-like memory), so the merge
// SUMS the s vectors of all inputs element-wise. The short conv windows are
// not accumulators and per-position MLA latents cannot be summed (they are a
// sequence, not a state), so conv_q/conv_k/conv_v, the MLA k/v caches and the
// stored logits are all taken from the FIRST file unchanged.

/// A parsed .mkmem file (header kept verbatim so a merge output is
/// byte-identical in layout to what `save` writes).
struct MemFile {
    header: Vec<u8>, // magic + fingerprint + layer kinds
    layers: Vec<LayerMem>,
    logits: Vec<f32>,
}

enum LayerMem {
    Kda { conv_q: Vec<f32>, conv_k: Vec<f32>, conv_v: Vec<f32>, s: Vec<f32> },
    Mla { k: Vec<f32>, v: Vec<f32> },
}

fn parse(path: &str) -> Result<MemFile, String> {
    let b = std::fs::read(path).map_err(|e| format!("cannot read {}: {}", path, e))?;
    let mut r = Reader { b: &b, p: 0 };
    if r.take(8)? != MAGIC {
        return Err(format!("{}: not a .mkmem file (bad magic)", path));
    }
    // header = magic + 5 x u32 fingerprint + n_layers x u8 kinds
    let fp: Vec<u32> = (0..5).map(|_| r.u32()).collect::<Result<_, _>>()?;
    let n_layers = fp[0] as usize;
    let kinds = r.take(n_layers)?.to_vec();
    let mut header = Vec::new();
    header.extend_from_slice(MAGIC);
    for v in &fp {
        header.extend_from_slice(&v.to_le_bytes());
    }
    header.extend_from_slice(&kinds);
    let mut layers = Vec::new();
    for (l, &kind) in kinds.iter().enumerate() {
        let tag = r.u8()?;
        match (kind, tag) {
            (0, 0) => layers.push(LayerMem::Kda {
                conv_q: r.vec_f32()?,
                conv_k: r.vec_f32()?,
                conv_v: r.vec_f32()?,
                s: r.vec_f32()?,
            }),
            (1, 1) => layers.push(LayerMem::Mla { k: r.vec_f32()?, v: r.vec_f32()? }),
            _ => return Err(format!("{}: corrupt .mkmem (layer {} kind/tag mismatch)", path, l)),
        }
    }
    let logits = r.vec_f32()?;
    Ok(MemFile { header, layers, logits })
}

fn write_mem(m: &MemFile, path: &str) -> std::io::Result<()> {
    let mut out = m.header.clone();
    for l in &m.layers {
        match l {
            LayerMem::Kda { conv_q, conv_k, conv_v, s } => {
                out.push(0);
                put_vec(&mut out, conv_q);
                put_vec(&mut out, conv_k);
                put_vec(&mut out, conv_v);
                put_vec(&mut out, s);
            }
            LayerMem::Mla { k, v } => {
                out.push(1);
                put_vec(&mut out, k);
                put_vec(&mut out, v);
            }
        }
    }
    put_vec(&mut out, &m.logits);
    std::fs::write(path, out)
}

/// Deterministic Fisher-Yates shuffle (xorshift64), used to build the
/// shuffled-garbage control: same values, same energy, destroyed structure.
fn shuffle(v: &mut [f32], seed: u64) {
    let mut x = seed | 1;
    for i in (1..v.len()).rev() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let j = (x % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
}

/// Merges N .mkmem files into one: KDA s = element-wise sum over all inputs,
/// everything else from the first input. `shuffle_idx` (1-based) marks an
/// input whose s vectors are shuffled before summing (garbage control).
/// With `avg`, the summed s is divided by N afterwards: a plain sum doubles
/// the state energy, which is itself off-distribution for the recurrence.
pub fn merge(paths: &[String], out: &str, shuffle_idx: Option<usize>, avg: bool) -> Result<(), String> {
    if paths.len() < 2 {
        return Err("mkmem-merge needs at least 2 input files".to_string());
    }
    let mut mems: Vec<MemFile> = Vec::new();
    for p in paths {
        mems.push(parse(p)?);
    }
    for (i, m) in mems.iter().enumerate().skip(1) {
        if m.header != mems[0].header {
            return Err(format!("{}: fingerprint differs from {} - cannot merge states of different models", paths[i], paths[0]));
        }
    }
    let mut base = mems.remove(0);
    for (i, m) in mems.into_iter().enumerate() {
        let idx = i + 2; // 1-based position in the original argument list
        for (bl, ml) in base.layers.iter_mut().zip(m.layers.into_iter()) {
            if let (LayerMem::Kda { s: bs, .. }, LayerMem::Kda { s: mut ms, .. }) = (bl, ml) {
                if bs.len() != ms.len() {
                    return Err(format!("{}: KDA state length mismatch", paths[idx - 1]));
                }
                if shuffle_idx == Some(idx) {
                    shuffle(&mut ms, 0x9E37_79B9_7F4A_7C15u64.wrapping_add(idx as u64));
                }
                for (a, b) in bs.iter_mut().zip(ms.iter()) {
                    *a += b;
                }
            }
        }
    }
    if avg {
        let n = paths.len() as f32;
        for l in base.layers.iter_mut() {
            if let LayerMem::Kda { s, .. } = l {
                for x in s.iter_mut() {
                    *x /= n;
                }
            }
        }
    }
    write_mem(&base, out).map_err(|e| format!("cannot write {}: {}", out, e))?;
    Ok(())
}

fn put_vec(out: &mut Vec<u8>, v: &[f32]) {
    out.extend_from_slice(&(v.len() as u32).to_le_bytes());
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

/// Reads a Vec<f32> whose length is fixed by the config (KDA caches).
fn get_fixed(r: &mut Reader, want: usize, path: &str) -> Result<Vec<f32>, String> {
    let v = r.vec_f32()?;
    if v.len() != want {
        return Err(format!("{}: corrupt .mkmem (cache length {} != expected {})", path, v.len(), want));
    }
    Ok(v)
}

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.p + n > self.b.len() {
            return Err("truncated .mkmem file".to_string());
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn vec_f32(&mut self) -> Result<Vec<f32>, String> {
        let n = self.u32()? as usize;
        let raw = self.take(n * 4)?;
        Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }
}
