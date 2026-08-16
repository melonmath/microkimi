// timeline.rs - version control for conversations (Qwen models).
//
// A conversation state is a commit: the MKMEMQW1 image after some token
// stream, content-addressed by SHA-256 and linked to the state it grew
// from. The DAG this builds supports the operations version control
// taught us to want, but on model minds instead of files:
//
//   fork   - continue from ANY past state, bit-exactly (the engine's
//            prefill is deterministic, so a fork is a real checkout, not
//            an approximation);
//   diff   - run one prompt from two states and report where the two
//            universes diverge;
//   merge  - combine two branches through their lowest common ancestor.
//
// The merge is the alien part, and it is only possible on this
// architecture. Eighteen of twenty-four layers are gated delta-rule
// linear attention whose state is a sum of decayed outer products - a
// LINEAR object. For those layers the three-way merge is literal
// arithmetic:  S_merged = S_a + S_b - S_ancestor,  the inclusion-
// exclusion that keeps the shared history counted once. The six full-
// attention layers are append-only key/value logs; the merge keeps the
// ancestor prefix once and appends both branch suffixes (their keys
// carry the rotary positions they were computed at - branch B's suffix
// overlaps branch A's position range, a declared approximation measured
// in QWEN.md rather than hidden). The MTP draft cache does not survive a
// merge and is cleared; positions add as pos_a + (pos_b - pos_anc).
//
// Node file <id>.tln in `<model>.timelines/`:
//   magic    : 8 bytes "MKTLN001"
//   parent   : 32 bytes (zero = root)
//   n_tokens : u32, then n_tokens x u32 covering token stream
//   payload  : MKMEMQW1 image (src/memory/qwen_state.rs)
// The id is the SHA-256 of the whole body, verified on every read: a
// corrupt node is refused, never silently used.

use crate::model::qwen::{QwenCache, QwenModel};

const MAGIC: &[u8; 8] = b"MKTLN001";

fn max_nodes() -> usize {
    std::env::var("MICROKIMI_TLN_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64)
}

pub struct TimelineStore {
    dir: String,
}

pub struct TimelineNode {
    pub id: String,
    pub parent: Option<String>,
    pub tokens: Vec<u32>,
    pub payload: Vec<u8>,
}

pub struct NodeMeta {
    pub id: String,
    pub parent: Option<String>,
    pub n_tokens: usize,
}

impl TimelineStore {
    /// Opens `<model>.timelines/` beside the model, creating it on first
    /// use. None when the directory cannot be created.
    pub fn open(model_path: &str) -> Option<TimelineStore> {
        let dir = format!("{}.timelines", model_path);
        match std::fs::create_dir_all(&dir) {
            Ok(()) => Some(TimelineStore { dir }),
            Err(e) => {
                eprintln!("timelines: {} not usable ({})", dir, e);
                None
            }
        }
    }

    fn path(&self, id: &str) -> std::path::PathBuf {
        std::path::Path::new(&self.dir).join(format!("{}.tln", id))
    }

    fn encode(parent: Option<&str>, tokens: &[u32], payload: &[u8]) -> Result<Vec<u8>, String> {
        let mut body = Vec::with_capacity(44 + tokens.len() * 4 + payload.len());
        body.extend_from_slice(MAGIC);
        match parent {
            None => body.extend_from_slice(&[0u8; 32]),
            Some(hex) => body.extend_from_slice(&parse_hex32(hex)?),
        }
        body.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
        for &t in tokens {
            body.extend_from_slice(&t.to_le_bytes());
        }
        body.extend_from_slice(payload);
        Ok(body)
    }

    /// Stores a node and returns its content id. Idempotent: an existing
    /// identical node is a no-op. Fails closed at the node cap.
    pub fn put(
        &self,
        parent: Option<&str>,
        tokens: &[u32],
        payload: &[u8],
    ) -> Result<String, String> {
        let body = Self::encode(parent, tokens, payload)?;
        let id = crate::sha256::hex(&crate::sha256::digest(&body));
        let path = self.path(&id);
        if path.exists() {
            return Ok(id);
        }
        if self.list().len() >= max_nodes() {
            return Err(format!(
                "timeline store at its {}-node cap; delete entries in {} or raise MICROKIMI_TLN_MAX",
                max_nodes(),
                self.dir
            ));
        }
        let tmp = format!("{}.tmp{}", path.display(), std::process::id());
        std::fs::write(&tmp, &body).map_err(|e| format!("cannot write node: {}", e))?;
        std::fs::rename(&tmp, &path).map_err(|e| format!("cannot commit node: {}", e))?;
        Ok(id)
    }

    /// Reads and verifies one node (the id must equal the body hash).
    pub fn get(&self, id: &str) -> Result<TimelineNode, String> {
        let body = std::fs::read(self.path(id)).map_err(|_| format!("unknown state {}", id))?;
        let actual = crate::sha256::hex(&crate::sha256::digest(&body));
        if actual != id {
            return Err(format!("state {} is corrupt (content hash mismatch)", id));
        }
        if body.len() < 44 || &body[..8] != MAGIC {
            return Err(format!("state {} is not a timeline node", id));
        }
        let parent_raw = &body[8..40];
        let parent = if parent_raw.iter().all(|&b| b == 0) {
            None
        } else {
            Some(crate::sha256::hex(parent_raw.try_into().unwrap()))
        };
        let n = u32::from_le_bytes(body[40..44].try_into().unwrap()) as usize;
        if body.len() < 44 + n * 4 {
            return Err(format!("state {} is truncated", id));
        }
        let tokens = body[44..44 + n * 4]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Ok(TimelineNode {
            id: id.to_string(),
            parent,
            tokens,
            payload: body[44 + n * 4..].to_vec(),
        })
    }

    /// Every stored node (header only), unordered.
    pub fn list(&self) -> Vec<NodeMeta> {
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("tln") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if let Ok(node) = self.get(stem) {
                out.push(NodeMeta {
                    id: node.id,
                    parent: node.parent,
                    n_tokens: node.tokens.len(),
                });
            }
        }
        out
    }

    /// Lowest common ancestor of two nodes: the first ancestor of `b`
    /// (walking up, starting at `b` itself) that is also an ancestor of
    /// `a` (including `a`).
    pub fn lowest_common_ancestor(&self, a: &str, b: &str) -> Result<String, String> {
        let mut seen = std::collections::HashSet::new();
        let mut cursor = Some(a.to_string());
        while let Some(id) = cursor {
            seen.insert(id.clone());
            cursor = self.get(&id)?.parent;
        }
        let mut cursor = Some(b.to_string());
        while let Some(id) = cursor {
            if seen.contains(&id) {
                return Ok(id);
            }
            cursor = self.get(&id)?.parent;
        }
        Err("the two states share no common ancestor".to_string())
    }
}

fn parse_hex32(hex: &str) -> Result<[u8; 32], String> {
    if hex.len() != 64 || !crate::sha256::is_lower_hex_digest(hex) {
        return Err(format!("{} is not a state id", hex));
    }
    let mut out = [0u8; 32];
    for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
        let hi = (chunk[0] as char).to_digit(16).unwrap();
        let lo = (chunk[1] as char).to_digit(16).unwrap();
        out[i] = (hi * 16 + lo) as u8;
    }
    Ok(out)
}

/// Extracted mutable-free copy of a model's caches (one branch of a merge).
struct BranchState {
    lin: Vec<(Vec<f32>, Vec<f32>)>,
    full: Vec<(Vec<f32>, Vec<f32>)>,
    pos: usize,
    last_logits: Vec<f32>,
}

fn extract(model: &QwenModel) -> BranchState {
    let mut lin = Vec::new();
    let mut full = Vec::new();
    for cache in &model.caches {
        match cache {
            QwenCache::Linear(c) => lin.push((c.state.clone(), c.conv.clone())),
            QwenCache::Full(c) => full.push((c.k.clone(), c.v.clone())),
        }
    }
    BranchState {
        lin,
        full,
        pos: model.pos,
        last_logits: model.last_logits.clone(),
    }
}

/// Three-way merge of two branches through their lowest common ancestor.
/// Loads the three states through `model` (its live state is clobbered),
/// writes the merged state into `model`, stores it as a new node with
/// parent `a`, and returns the new id.
///
/// Soundness by layer kind: linear states merge exactly
/// (S_a + S_b - S_anc); branch A's convolution window is kept (the
/// continuation follows A locally); full-attention caches keep the
/// ancestor prefix once and append both suffixes, B's keys at their
/// original rotary positions (declared approximation); the MTP cache is
/// cleared. Fails closed unless the ancestor's token stream is a strict
/// prefix of both branches.
pub fn merge_nodes(
    store: &TimelineStore,
    model: &mut QwenModel,
    a_id: &str,
    b_id: &str,
) -> Result<String, String> {
    if a_id == b_id {
        return Ok(a_id.to_string());
    }
    let anc_id = store.lowest_common_ancestor(a_id, b_id)?;
    if anc_id == a_id || anc_id == b_id {
        return Err("one state is an ancestor of the other: fork before merging".to_string());
    }
    let a_node = store.get(a_id)?;
    let b_node = store.get(b_id)?;
    let anc_node = store.get(&anc_id)?;
    for (name, node) in [("a", &a_node), ("b", &b_node)] {
        if node.tokens.len() <= anc_node.tokens.len()
            || node.tokens[..anc_node.tokens.len()] != anc_node.tokens[..]
        {
            return Err(format!(
                "branch {} does not extend the common ancestor's token stream",
                name
            ));
        }
    }

    crate::memory::qwen_state::load_slice(model, &anc_node.payload, "merge ancestor")?;
    let anc = extract(model);
    crate::memory::qwen_state::load_slice(model, &a_node.payload, "merge branch a")?;
    let a = extract(model);
    crate::memory::qwen_state::load_slice(model, &b_node.payload, "merge branch b")?;
    let b = extract(model);

    let kv_width = model.cfg.n_kv_heads * model.cfg.head_dim;
    let anc_rows = anc.pos * kv_width;
    let merged_pos = a.pos + (b.pos - anc.pos);

    let (mut li, mut fi) = (0usize, 0usize);
    for cache in model.caches.iter_mut() {
        match cache {
            QwenCache::Linear(c) => {
                let (sa, conv_a) = &a.lin[li];
                let (sb, _) = &b.lin[li];
                let (sanc, _) = &anc.lin[li];
                c.state = sa
                    .iter()
                    .zip(sb)
                    .zip(sanc)
                    .map(|((&x, &y), &z)| x + y - z)
                    .collect();
                c.conv = conv_a.clone();
                li += 1;
            }
            QwenCache::Full(c) => {
                let (ka, va) = &a.full[fi];
                let (kb, vb) = &b.full[fi];
                let mut k = ka.clone();
                let mut v = va.clone();
                k.extend_from_slice(&kb[anc_rows..]);
                v.extend_from_slice(&vb[anc_rows..]);
                c.len = merged_pos;
                c.k = k;
                c.v = v;
                fi += 1;
            }
        }
    }
    model.mtp_cache.k.clear();
    model.mtp_cache.v.clear();
    model.mtp_cache.len = 0;
    model.pos = merged_pos;
    model.last_logits = a.last_logits;

    let mut tokens = a_node.tokens.clone();
    tokens.extend_from_slice(&b_node.tokens[anc_node.tokens.len()..]);
    let payload = crate::memory::qwen_state::serialize(model)?;
    store.put(Some(a_id), &tokens, &payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_dense() -> crate::config::QwenConfig {
        let mut c = crate::config::QwenConfig::qwen38_dense();
        c.n_layers = 4;
        c.d = 32;
        c.vocab = 64;
        c.n_heads = 2;
        c.n_kv_heads = 1;
        c.head_dim = 16;
        c.lin_k_heads = 1;
        c.lin_v_heads = 1;
        c.lin_k_dim = 32;
        c.lin_v_dim = 32;
        c.dense_inter = 64;
        c
    }

    fn store(name: &str) -> TimelineStore {
        let dir = std::env::temp_dir().join(format!(
            "microkimi_tln_test_{}_{}",
            std::process::id(),
            name
        ));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        TimelineStore {
            dir: dir.to_string_lossy().into_owned(),
        }
    }

    fn commit(
        store: &TimelineStore,
        model: &QwenModel,
        parent: Option<&str>,
        tokens: &[u32],
    ) -> String {
        let payload = crate::memory::qwen_state::serialize(model).unwrap();
        store.put(parent, tokens, &payload).unwrap()
    }

    #[test]
    fn store_roundtrip_is_content_addressed_and_verified() {
        let s = store("roundtrip");
        let c = tiny_dense();
        let path = crate::model::qwen::test_fixture(&c);
        let mut model = QwenModel::load(&path);
        model.prefill(&[3, 5, 7]);
        let id = commit(&s, &model, None, &[3, 5, 7]);
        let again = commit(&s, &model, None, &[3, 5, 7]);
        assert_eq!(id, again, "identical nodes must dedup");
        let node = s.get(&id).unwrap();
        assert_eq!(node.tokens, vec![3, 5, 7]);
        assert!(node.parent.is_none());

        // corruption is refused
        let victim = s.path(&id);
        let mut bytes = std::fs::read(&victim).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        std::fs::write(&victim, bytes).unwrap();
        assert!(s.get(&id).is_err());

        drop(model);
        std::fs::remove_dir_all(&s.dir).ok();
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn lca_walks_the_dag() {
        let s = store("lca");
        let c = tiny_dense();
        let path = crate::model::qwen::test_fixture(&c);
        let mut model = QwenModel::load(&path);
        model.prefill(&[3]);
        let root = commit(&s, &model, None, &[3]);
        model.prefill(&[5]);
        let mid = commit(&s, &model, Some(&root), &[3, 5]);
        let mut model_b = QwenModel::load(&path);
        model_b.prefill(&[3, 5, 7]);
        let left = commit(&s, &model_b, Some(&mid), &[3, 5, 7]);
        model_b.reset();
        model_b.prefill(&[3, 5, 9]);
        let right = commit(&s, &model_b, Some(&mid), &[3, 5, 9]);
        assert_eq!(s.lowest_common_ancestor(&left, &right).unwrap(), mid);
        assert_eq!(s.lowest_common_ancestor(&left, &mid).unwrap(), mid);
        drop((model, model_b));
        std::fs::remove_dir_all(&s.dir).ok();
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn three_way_merge_is_exact_on_linear_states() {
        let s = store("merge");
        let c = tiny_dense();
        let path = crate::model::qwen::test_fixture(&c);

        let mut model = QwenModel::load(&path);
        model.prefill(&[3, 5]);
        let anc_state = extract(&model);
        let anc = commit(&s, &model, None, &[3, 5]);

        model.prefill(&[7, 11]);
        let a_state = extract(&model);
        let a = commit(&s, &model, Some(&anc), &[3, 5, 7, 11]);

        model.reset();
        model.prefill(&[3, 5]);
        model.prefill(&[9]);
        let b_state = extract(&model);
        let b = commit(&s, &model, Some(&anc), &[3, 5, 9]);

        let merged_id = merge_nodes(&s, &mut model, &a, &b).unwrap();
        let merged = s.get(&merged_id).unwrap();
        assert_eq!(merged.tokens, vec![3, 5, 7, 11, 9]);
        assert_eq!(merged.parent.as_deref(), Some(a.as_str()));

        // the linear states obey inclusion-exclusion to the bit
        crate::memory::qwen_state::load_slice(&mut model, &merged.payload, "check").unwrap();
        let got = extract(&model);
        for layer in 0..got.lin.len() {
            for i in 0..got.lin[layer].0.len() {
                let expect =
                    a_state.lin[layer].0[i] + b_state.lin[layer].0[i] - anc_state.lin[layer].0[i];
                assert_eq!(got.lin[layer].0[i], expect, "layer {} slot {}", layer, i);
            }
        }
        // position arithmetic and KV length agree
        assert_eq!(got.pos, 4 + (3 - 2));
        let kv_width = c.n_kv_heads * c.head_dim;
        assert_eq!(got.full[0].0.len(), got.pos * kv_width);

        // ancestor-of relation refuses to merge
        assert!(merge_nodes(&s, &mut model, &anc, &a).is_err());

        drop(model);
        std::fs::remove_dir_all(&s.dir).ok();
        std::fs::remove_file(path).ok();
    }
}
