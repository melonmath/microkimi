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
