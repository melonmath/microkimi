// Safetensors input for `microkimi slice`: local files/directories and
// HuggingFace URLs. Only the NEEDED bytes are ever read: the index.json is
// resolved into tensor -> (shard, byte range), shard headers are fetched
// lazily via HTTP range requests (build.rs style), and tensor data is
// streamed per row-chunk. A full K3 shard is never downloaded.
//
// Name translation (safetensors -> .bin logical names):
//   language_model.model.layers.3.self_attn.q_a_proj.weight -> layers.3.self_attn.q_a_proj.weight
//   language_model.lm_head.weight                           -> lm_head.weight
//   ...experts.{e}.{w}.weight_packed + .weight_scale        -> ...experts.{e}.{w} (one MXFP4 blob)
//   model.layers.0.self_attn.q_proj.weight (dense models)   -> layers.0.self_attn.q_proj.weight
// Vision tower / mm_projector tensors are skipped entirely.
//
// Two passes read the non-expert tensors when --hidden is given (scoring then
// writing). For remote inputs the converted f32 bytes are cached on disk
// (next to the output, removed on exit) so every byte is fetched exactly
// once. Without --hidden the write pass is the only reader and everything
// streams with no cache at all.
//
// Expert ranking fetches ONLY the weight_scale tensors (1/17 of the expert
// bytes): ||W||^2 = sum over groups of 2^(2*(s-127)) * sum(lut^2), and the
// e2m1 codebook factor has the same expectation for every expert, so the
// scale energy ranks the same way at a fraction of the bandwidth.

use crate::config::Config;
use crate::json::Json;
use crate::safetensors::{self, TensorInfo};
use crate::slice::DirEntry;
use crate::weights::{blob_size, DTYPE_F32, DTYPE_MXFP4};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum StArch {
    K3,
    Dense,
}

#[derive(Clone)]
enum ShardLoc {
    Local(std::path::PathBuf),
    Remote(String), // full URL of the shard
}

/// Byte range of one raw tensor inside its shard (absolute offsets).
#[derive(Clone)]
struct TRef {
    shard: usize,
    start: u64,
    len: u64,
    dtype: String, // safetensors dtype: "BF16", "F32", "U8"
}

#[derive(Clone)]
enum EntrySrc {
    Unresolved, // entry of a pruned layer or not yet resolved
    F32(TRef),  // bf16/f32 storage, converted to f32 on read
    Expert { packed: TRef, scale: TRef },
    Alias(usize), // tied lm_head: same data as another entry
}

#[derive(Clone, Copy, PartialEq)]
enum LayerKind {
    Kda,
    Mla,
}

pub struct StDir {
    pub config: Config,
    pub arch: StArch,
    pub entries: Vec<DirEntry>, // logical .bin-style directory (dims filled by resolve)
    srcs: Vec<EntrySrc>,        // parallel to entries
    pub(crate) index: HashMap<String, usize>,
    weight_map: HashMap<String, String>, // original safetensors name -> shard file name
    orig: HashMap<String, String>,       // logical name -> original safetensors name
    shards: Vec<ShardLoc>,
    shard_idx: HashMap<String, usize>,
    headers: Mutex<HashMap<usize, (u64, HashMap<String, TensorInfo>)>>,
    remote: bool,
    cache_dir: Option<std::path::PathBuf>,
    cache_enabled: AtomicBool,
    n_layers: usize,
    layer_kind: Vec<Option<LayerKind>>, // attention kind per layer
    layer_moe: Vec<bool>,
    n_experts: usize,
    hf_config: Option<Json>, // text_config of the HF config.json (when found)
    resolved: AtomicBool,
}

/// Squeezes size-1 dims out of a safetensors shape: [12288,1,4] -> [12288,4],
/// [1,7168] -> [7168] (the .bin convention used by `microkimi build`).
fn squeeze(shape: &[usize]) -> Vec<u32> {
    let v: Vec<u32> = shape.iter().filter(|&&d| d != 1).map(|&d| d as u32).collect();
    if v.is_empty() {
        vec![1]
    } else {
        v
    }
}

fn gb(n: u64) -> String {
    format!("{:.2} GB", n as f64 / 1e9)
}

/// Local-mirror read for a remote safetensors shard. The mirror directory is
/// $MICROKIMI_MIRROR, defaulting to /mnt/k3 when that exists. A file is only
/// used when it is complete for the requested range (shards may still be
/// downloading): short file -> None -> caller falls back to HTTP.
fn mirror_range(url: &str, start: u64, len: u64) -> Option<Vec<u8>> {
    use std::sync::OnceLock;
    static DIR: OnceLock<Option<String>> = OnceLock::new();
    let dir = DIR.get_or_init(|| {
        if let Ok(d) = std::env::var("MICROKIMI_MIRROR") {
            return Some(d);
        }
        if std::path::Path::new("/mnt/k3").is_dir() {
            Some("/mnt/k3".to_string())
        } else {
            None
        }
    });
    let dir = dir.as_ref()?;
    let fname = url.rsplit('/').next()?;
    let path = format!("{}/{}", dir, fname);
    let f = std::fs::File::open(&path).ok()?;
    if f.metadata().ok()?.len() < start + len {
        return None; // partial download: serve from HTTP instead
    }
    static LOGGED: OnceLock<()> = OnceLock::new();
    LOGGED.get_or_init(|| {
        eprintln!("  mirror: serving shard reads from {} (HTTP fallback for missing/partial)", dir);
    });
    use std::os::unix::fs::FileExt;
    let mut buf = vec![0u8; len as usize];
    f.read_exact_at(&mut buf, start).ok()?;
    Some(buf)
}

impl StDir {
    pub fn open(path: &str, cache_hint: &str) -> StDir {
        let remote = path.starts_with("http://") || path.starts_with("https://");
        let base_url;
        let local_dir;
        let single_file;
        if remote {
            // accept both .../org/repo and .../org/repo/resolve/main
            let b = path.trim_end_matches('/');
            base_url = if b.contains("/resolve/") { b.to_string() } else { format!("{}/resolve/main", b) };
            local_dir = None;
            single_file = None;
        } else {
            let p = std::path::Path::new(path);
            if p.is_dir() {
                base_url = String::new();
                local_dir = Some(p.to_path_buf());
                single_file = None;
            } else {
                base_url = String::new();
                local_dir = p.parent().map(|d| d.to_path_buf());
                single_file = Some(p.to_path_buf());
            }
        }

        // ── weight_map (tensor -> shard) ──
        let mut weight_map: HashMap<String, String> = HashMap::new();
        if let Some(f) = &single_file {
            let fname = f.file_name().unwrap().to_string_lossy().into_owned();
            // header only: the file itself can be ~1 GB
            use std::os::unix::fs::FileExt;
            let fh = std::fs::File::open(f).unwrap_or_else(|e| panic!("{} unreadable: {}", path, e));
            let mut b8 = [0u8; 8];
            fh.read_exact_at(&mut b8, 0).unwrap();
            let hlen = u64::from_le_bytes(b8) as usize;
            let mut buf = vec![0u8; 8 + hlen];
            buf[0..8].copy_from_slice(&b8);
            fh.read_exact_at(&mut buf[8..], 8).unwrap();
            let (_, map) = safetensors::parse_header(&buf);
            for name in map.keys() {
                weight_map.insert(name.clone(), fname.clone());
            }
        } else {
            let idx_bytes = if remote {
                crate::http::fetch(&format!("{}/model.safetensors.index.json", base_url))
                    .unwrap_or_else(|| panic!("cannot fetch {}/model.safetensors.index.json", base_url))
            } else {
                let dir = local_dir.clone().unwrap();
                let idx = dir.join("model.safetensors.index.json");
                if idx.exists() {
                    std::fs::read(&idx).unwrap()
                } else {
                    // directory with a single .safetensors shard
                    let shard = std::fs::read_dir(&dir)
                        .unwrap()
                        .flatten()
                        .map(|e| e.path())
                        .find(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
                        .unwrap_or_else(|| panic!("{}: no model.safetensors.index.json and no .safetensors file", path));
                    return Self::open(&shard.to_string_lossy(), cache_hint);
                }
            };
            let parsed = crate::json::parse(&idx_bytes);
            if let Some(Json::Obj(pairs)) = parsed.get("weight_map") {
                for (k, v) in pairs {
                    if let Some(s) = v.as_str() {
                        weight_map.insert(k.clone(), s.to_string());
                    }
                }
            }
            assert!(!weight_map.is_empty(), "{}: empty safetensors index", path);
        }

        // ── name translation + arch detection ──
        let mut logical: Vec<(String, String)> = Vec::new(); // (logical, original)
        let mut saw_k3_prefix = false;
        let mut skipped = 0usize;
        for name in weight_map.keys() {
            if let Some(rest) = name.strip_prefix("language_model.model.") {
                saw_k3_prefix = true;
                logical.push((rest.to_string(), name.clone()));
            } else if name.as_str() == "language_model.lm_head.weight" {
                saw_k3_prefix = true;
                logical.push(("lm_head.weight".to_string(), name.clone()));
            } else if let Some(rest) = name.strip_prefix("model.") {
                if rest.ends_with("rotary_emb.inv_freq") {
                    skipped += 1;
                    continue; // precomputed rope tables are not weights
                }
                logical.push((rest.to_string(), name.clone()));
            } else if name.starts_with("vision_tower.") || name.starts_with("mm_projector.") {
                skipped += 1;
            } else if name == "lm_head.weight" {
                logical.push((name.clone(), name.clone()));
            } else if name.contains("block_sparse_moe") || name.contains("self_attn.A_log") {
                logical.push((name.clone(), name.clone())); // bare K3 names
            } else {
                skipped += 1;
            }
        }
        let arch = if saw_k3_prefix || logical.iter().any(|(l, _)| l.contains("block_sparse_moe") || l.contains("self_attn.A_log")) {
            StArch::K3
        } else {
            StArch::Dense
        };
        println!(
            "safetensors source: {} ({} tensors mapped, {} skipped: vision/projector/rope)",
            if remote { "remote URL (range requests)" } else { "local" },
            logical.len(),
            skipped
        );

        // ── per-layer structure from names (no shard headers needed) ──
        let mut max_layer: Option<usize> = None;
        let mut has_names: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (l, _) in &logical {
            has_names.insert(l.clone());
            if let Some((i, _)) = crate::slice::split_layer(l) {
                max_layer = Some(max_layer.map_or(i, |m: usize| m.max(i)));
            }
        }
        let n_layers = max_layer.map(|m| m + 1).unwrap_or(0);
        let mut layer_kind: Vec<Option<LayerKind>> = vec![None; n_layers];
        let mut layer_moe = vec![false; n_layers];
        let mut n_experts = 0usize;
        for l in 0..n_layers {
            let p = format!("layers.{}.", l);
            if has_names.contains(format!("{}self_attn.q_a_proj.weight", p).as_str()) {
                layer_kind[l] = Some(LayerKind::Mla);
            } else if has_names.contains(format!("{}self_attn.q_proj.weight", p).as_str()) {
                layer_kind[l] = Some(LayerKind::Kda);
            }
            if has_names.contains(format!("{}block_sparse_moe.gate.weight", p).as_str()) {
                layer_moe[l] = true;
            }
        }
        // expert count: distinct expert ids on the first MoE layer
        if let Some(ml) = layer_moe.iter().position(|&m| m) {
            let pfx = format!("layers.{}.block_sparse_moe.experts.", ml);
            let mut ids = std::collections::HashSet::new();
            for (l, _) in &logical {
                if let Some(rest) = l.strip_prefix(&pfx) {
                    if let Some(dot) = rest.find('.') {
                        if let Ok(e) = rest[..dot].parse::<usize>() {
                            ids.insert(e);
                        }
                    }
                }
            }
            n_experts = ids.len();
        }

        // ── HF config.json (scalars + cross-check; optional) ──
        let hf_config: Option<Json> = if remote {
            crate::http::fetch(&format!("{}/config.json", base_url)).map(|b| crate::json::parse(&b))
        } else {
            local_dir
                .as_ref()
                .map(|d| d.join("config.json"))
                .filter(|p| p.exists())
                .map(|p| crate::json::parse(&std::fs::read(&p).unwrap()))
        };
        let hf_config = hf_config.map(|j| j.get("text_config").cloned().unwrap_or(j));

        // ── logical entry list (sorted, deterministic) ──
        // merge expert weight_packed + weight_scale pairs into one entry
        let mut names: Vec<String> = Vec::new();
        let mut orig: HashMap<String, String> = HashMap::new();
        for (l, o) in logical {
            if let Some(base) = l.strip_suffix(".weight_scale") {
                if l.contains(".experts.") {
                    continue; // folded into the .weight_packed entry
                }
                let _ = base;
            }
            if let Some(base) = l.strip_suffix(".weight_packed") {
                assert!(
                    has_names.contains(format!("{}.weight_scale", base).as_str()),
                    "{}: weight_packed without weight_scale",
                    l
                );
                names.push(base.to_string());
                orig.insert(base.to_string(), o.trim_end_matches(".weight_packed").to_string());
            } else {
                names.push(l.clone());
                orig.insert(l, o);
            }
        }
        let sort_key = |name: &str| -> (i64, u8, u64, String) {
            match crate::slice::split_layer(name) {
                None => (-1, 0, 0, name.to_string()),
                Some((l, rest)) => {
                    if let Some(tail) = rest.strip_prefix("block_sparse_moe.experts.") {
                        let dot = tail.find('.').unwrap();
                        let e: u64 = tail[..dot].parse().unwrap();
                        (l as i64, 1, e, tail[dot..].to_string())
                    } else {
                        (l as i64, 0, 0, rest.to_string())
                    }
                }
            }
        };
        names.sort_by_cached_key(|n: &String| sort_key(n));
        // tied embeddings (dense models without lm_head): alias of embed
        if arch == StArch::Dense && !names.iter().any(|n| n == "lm_head.weight") {
            let pos = names.iter().position(|n| n == "norm.weight").unwrap_or(names.len());
            names.insert(pos, "lm_head.weight".to_string());
        }
        let index: HashMap<String, usize> = names.iter().enumerate().map(|(i, n)| (n.clone(), i)).collect();
        let entries: Vec<DirEntry> = names
            .iter()
            .map(|n| DirEntry { name: n.clone(), dtype: DTYPE_F32, dims: Vec::new(), offset: 0, size: 0 })
            .collect();
        let srcs = vec![EntrySrc::Unresolved; entries.len()];

        // ── shards ──
        let mut shards: Vec<ShardLoc> = Vec::new();
        let mut shard_idx: HashMap<String, usize> = HashMap::new();
        for s in weight_map.values() {
            if !shard_idx.contains_key(s) {
                let loc = if remote {
                    ShardLoc::Remote(format!("{}/{}", base_url, s))
                } else {
                    match &single_file {
                        Some(f) => ShardLoc::Local(f.clone()),
                        None => ShardLoc::Local(local_dir.as_ref().unwrap().join(s)),
                    }
                };
                shard_idx.insert(s.clone(), shards.len());
                shards.push(loc);
            }
        }

        let cache_dir = if remote {
            let d = std::path::PathBuf::from(format!("{}.slicecache", cache_hint));
            std::fs::create_dir_all(&d).ok();
            Some(d)
        } else {
            None
        };

        let mut config = Config::microkimi(); // K3 counts; dims filled by resolve()
        config.n_layers = n_layers;
        config.n_experts = n_experts;
        if let Some(j) = &hf_config {
            let num = |k: &str, dflt: f64| j.get(k).and_then(|x| x.as_num()).unwrap_or(dflt);
            config.top_k = num("num_experts_per_token", config.top_k as f64) as usize;
            config.n_shared = num("num_shared_experts", config.n_shared as f64) as usize;
            config.first_k_dense = num("first_k_dense_replace", config.first_k_dense as f64) as usize;
            config.attn_res_block = num("attn_res_block_size", config.attn_res_block as f64) as usize;
            config.rms_eps = num("rms_norm_eps", config.rms_eps as f64) as f32;
            config.gate_lb = num("gate_lower_bound", config.gate_lb as f64) as f32;
            if let Some(la) = j.get("linear_attn_config") {
                config.gate_lb = la.get("gate_lower_bound").and_then(|x| x.as_num()).map(|x| x as f32).unwrap_or(config.gate_lb);
            }
        }
        config.mla_layers = Some((0..n_layers).filter(|&l| layer_kind[l] == Some(LayerKind::Mla)).collect());
        config.dense_layers = Some((0..n_layers).filter(|&l| !layer_moe[l]).collect());

        let dir = StDir {
            config,
            arch,
            entries,
            srcs,
            index,
            weight_map,
            orig,
            shards,
            shard_idx,
            headers: Mutex::new(HashMap::new()),
            remote,
            cache_dir,
            cache_enabled: AtomicBool::new(false),
            n_layers,
            layer_kind,
            layer_moe,
            n_experts,
            hf_config,
            resolved: AtomicBool::new(false),
        };
        dir.audit_names();
        dir
    }

    /// Every tensor name the slicer relies on must exist in the index; runs at
    /// open, before any data byte is fetched (this is the index.json audit).
    fn audit_names(&self) {
        let names: std::collections::HashSet<&str> = self.entries.iter().map(|e| e.name.as_str()).collect();
        let mut missing: Vec<String> = Vec::new();
        let mut check = |n: String| {
            if !names.contains(n.as_str()) {
                missing.push(n);
            }
        };
        match self.arch {
            StArch::K3 => {
                for g in ["embed_tokens.weight", "lm_head.weight", "norm.weight", "output_attn_res_norm.weight", "output_attn_res_proj.weight"] {
                    check(g.to_string());
                }
                const COMMON: &[&str] = &[
                    "input_layernorm.weight",
                    "post_attention_layernorm.weight",
                    "self_attention_res_norm.weight",
                    "self_attention_res_proj.weight",
                    "mlp_res_norm.weight",
                    "mlp_res_proj.weight",
                ];
                const KDA: &[&str] = &[
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                    "self_attn.g_proj.weight",
                    "self_attn.o_proj.weight",
                    "self_attn.q_conv1d.weight",
                    "self_attn.k_conv1d.weight",
                    "self_attn.v_conv1d.weight",
                    "self_attn.f_a_proj.weight",
                    "self_attn.f_b_proj.weight",
                    "self_attn.A_log",
                    "self_attn.dt_bias",
                    "self_attn.b_proj.weight",
                    "self_attn.o_norm.weight",
                ];
                const MLA: &[&str] = &[
                    "self_attn.q_a_proj.weight",
                    "self_attn.q_a_layernorm.weight",
                    "self_attn.q_b_proj.weight",
                    "self_attn.kv_a_proj_with_mqa.weight",
                    "self_attn.kv_a_layernorm.weight",
                    "self_attn.kv_b_proj.weight",
                    "self_attn.g_proj.weight",
                    "self_attn.o_proj.weight",
                ];
                const MOE: &[&str] = &[
                    "block_sparse_moe.gate.weight",
                    "block_sparse_moe.gate.e_score_correction_bias",
                    "block_sparse_moe.routed_expert_down_proj.weight",
                    "block_sparse_moe.routed_expert_up_proj.weight",
                    "block_sparse_moe.routed_expert_norm.weight",
                    "block_sparse_moe.shared_experts.gate_proj.weight",
                    "block_sparse_moe.shared_experts.up_proj.weight",
                    "block_sparse_moe.shared_experts.down_proj.weight",
                ];
                const DENSE: &[&str] = &["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"];
                for l in 0..self.n_layers {
                    let p = format!("layers.{}.", l);
                    for c in COMMON {
                        check(format!("{}{}", p, c));
                    }
                    match self.layer_kind[l] {
                        Some(LayerKind::Kda) => KDA.iter().for_each(|t| check(format!("{}{}", p, t))),
                        Some(LayerKind::Mla) => MLA.iter().for_each(|t| check(format!("{}{}", p, t))),
                        None => check(format!("{}<no attention tensors>", p)),
                    }
                    if self.layer_moe[l] {
                        MOE.iter().for_each(|t| check(format!("{}{}", p, t)));
                        for e in 0..self.n_experts {
                            for w in ["w1", "w2", "w3"] {
                                check(format!("{}block_sparse_moe.experts.{}.{}", p, e, w));
                            }
                        }
                    } else {
                        DENSE.iter().for_each(|t| check(format!("{}{}", p, t)));
                    }
                }
            }
            StArch::Dense => {
                for g in ["embed_tokens.weight", "lm_head.weight", "norm.weight"] {
                    check(g.to_string());
                }
                for l in 0..self.n_layers {
                    let p = format!("layers.{}.", l);
                    for t in [
                        "input_layernorm.weight",
                        "post_attention_layernorm.weight",
                        "self_attn.q_proj.weight",
                        "self_attn.k_proj.weight",
                        "self_attn.v_proj.weight",
                        "self_attn.o_proj.weight",
                        "mlp.gate_proj.weight",
                        "mlp.up_proj.weight",
                        "mlp.down_proj.weight",
                    ] {
                        check(format!("{}{}", p, t));
                    }
                }
            }
        }
        assert!(missing.is_empty(), "safetensors index audit FAILED, {} missing tensors, e.g.: {:?}", missing.len(), &missing[..missing.len().min(8)]);
        println!(
            "safetensors index audit: {} logical tensors, {} layers, arch {:?} - all expected names present",
            self.entries.len(),
            self.n_layers,
            self.arch
        );
    }

    pub fn enable_caching(&self) {
        self.cache_enabled.store(true, Ordering::Relaxed);
    }

    pub fn is_remote(&self) -> bool {
        self.remote
    }

    fn header(&self, shard: usize) -> std::sync::MutexGuard<'_, HashMap<usize, (u64, HashMap<String, TensorInfo>)>> {
        let mut h = self.headers.lock().unwrap();
        if !h.contains_key(&shard) {
            let loaded = match &self.shards[shard] {
                ShardLoc::Local(p) => {
                    let f = std::fs::File::open(p).unwrap_or_else(|e| panic!("{:?} unreadable: {}", p, e));
                    use std::os::unix::fs::FileExt;
                    let mut b8 = [0u8; 8];
                    f.read_exact_at(&mut b8, 0).unwrap();
                    let hlen = u64::from_le_bytes(b8);
                    let mut buf = vec![0u8; 8 + hlen as usize];
                    f.read_exact_at(&mut buf[8..], 8).unwrap();
                    buf[0..8].copy_from_slice(&b8);
                    safetensors::parse_header(&buf)
                }
                ShardLoc::Remote(url) => {
                    let first = crate::http::fetch_range(url, Some((0, 7))).unwrap_or_else(|| panic!("cannot fetch header of {}", url));
                    let hlen = u64::from_le_bytes(first[0..8].try_into().unwrap());
                    let head = crate::http::fetch_range(url, Some((8, 8 + hlen - 1))).unwrap();
                    safetensors::parse_header(&[first, head].concat())
                }
            };
            h.insert(shard, loaded);
        }
        h
    }

    fn tref(&self, orig_name: &str) -> TRef {
        let shard_name = self.weight_map.get(orig_name).unwrap_or_else(|| panic!("{} not in weight_map", orig_name));
        let shard = self.shard_idx[shard_name];
        let h = self.header(shard);
        let (hlen, map) = h.get(&shard).unwrap();
        let info = map.get(orig_name).unwrap_or_else(|| panic!("{} not in shard {}", orig_name, shard_name));
        TRef {
            shard,
            start: 8 + hlen + info.offsets.0,
            len: info.offsets.1 - info.offsets.0,
            dtype: info.dtype.clone(),
        }
    }

    /// Fills dims + byte sources for the global tensors and the kept layers,
    /// then derives the real architecture dims from the actual shapes.
    pub fn resolve(&mut self, kept_layers: &[usize]) {
        if self.resolved.swap(true, Ordering::Relaxed) {
            return;
        }
        let keep = |name: &str| -> bool {
            match crate::slice::split_layer(name) {
                None => true,
                Some((l, _)) => kept_layers.contains(&l),
            }
        };
        // sample layers for dim derivation: prefer kept ones, fall back to any
        let sample = |pred: &dyn Fn(usize) -> bool| (0..self.n_layers).find(|&l| kept_layers.contains(&l) && pred(l)).or_else(|| (0..self.n_layers).find(|&l| pred(l)));
        let kda_sample = sample(&|l| self.layer_kind[l] == Some(LayerKind::Kda));
        let mla_sample = sample(&|l| self.layer_kind[l] == Some(LayerKind::Mla));
        let moe_sample = sample(&|l| self.layer_moe[l]);
        let dense_sample = sample(&|l| !self.layer_moe[l]);
        let mut extra: Vec<String> = Vec::new();
        if let Some(l) = kda_sample {
            for t in ["self_attn.q_proj.weight", "self_attn.b_proj.weight", "self_attn.o_norm.weight", "self_attn.q_conv1d.weight", "self_attn.f_a_proj.weight"] {
                extra.push(format!("layers.{}.{}", l, t));
            }
        }
        if let Some(l) = mla_sample {
            for t in ["self_attn.q_a_proj.weight", "self_attn.q_b_proj.weight", "self_attn.kv_a_proj_with_mqa.weight", "self_attn.kv_a_layernorm.weight", "self_attn.kv_b_proj.weight"] {
                extra.push(format!("layers.{}.{}", l, t));
            }
        }
        if let Some(l) = moe_sample {
            let p = format!("layers.{}.block_sparse_moe.", l);
            for t in ["gate.weight", "routed_expert_norm.weight", "shared_experts.gate_proj.weight", "experts.0.w1", "experts.0.w2", "experts.0.w3"] {
                extra.push(format!("{}{}", p, t));
            }
        }
        if let Some(l) = dense_sample {
            extra.push(format!("layers.{}.mlp.gate_proj.weight", l));
        }
        let wanted = |name: &str| keep(name) || extra.iter().any(|e| e == name);

        let mut shapes: HashMap<String, Vec<u32>> = HashMap::new(); // logical -> dims (samples + wanted)
        let tied_lm_head = self.arch == StArch::Dense && !self.weight_map.contains_key("lm_head.weight");
        for i in 0..self.entries.len() {
            let name = self.entries[i].name.clone();
            if tied_lm_head && name == "lm_head.weight" {
                continue; // alias, filled after the loop
            }
            if !wanted(&name) {
                self.srcs[i] = EntrySrc::Unresolved; // pruned layer: never touched
                continue;
            }
            let orig = self.orig[&name].clone();
            if self.weight_map.contains_key(&format!("{}.weight_packed", orig)) {
                // merged MXFP4 expert (packed and scale can live in different shards)
                let packed = self.tref(&format!("{}.weight_packed", orig));
                let scale = self.tref(&format!("{}.weight_scale", orig));
                let ps = {
                    let h = self.header(packed.shard);
                    h[&packed.shard].1[&format!("{}.weight_packed", orig)].shape.clone()
                };
                let ss = {
                    let h = self.header(scale.shard);
                    h[&scale.shard].1[&format!("{}.weight_scale", orig)].shape.clone()
                };
                let (r, c) = (ps[0], ps[1] * 2);
                assert_eq!(ss, vec![r, c / 32], "{}: unexpected scale shape", name);
                assert_eq!(packed.dtype, "U8");
                assert_eq!(scale.dtype, "U8");
                assert_eq!(packed.len, (r * c / 2) as u64, "{}: packed size mismatch", name);
                assert_eq!(scale.len, (r * c / 32) as u64, "{}: scale size mismatch", name);
                let dims = vec![r as u32, c as u32];
                self.entries[i].dtype = DTYPE_MXFP4;
                self.entries[i].size = blob_size(DTYPE_MXFP4, &dims);
                self.entries[i].dims = dims.clone();
                self.srcs[i] = EntrySrc::Expert { packed, scale };
                shapes.insert(name, dims);
            } else {
                let t = self.tref(&orig);
                assert!(t.dtype == "BF16" || t.dtype == "F32", "{}: unhandled dtype {}", name, t.dtype);
                let h = self.header(t.shard);
                let shape = h.get(&t.shard).unwrap().1[&orig].shape.clone();
                drop(h);
                let dims = squeeze(&shape);
                self.entries[i].dtype = DTYPE_F32;
                self.entries[i].size = blob_size(DTYPE_F32, &dims);
                self.entries[i].dims = dims.clone();
                self.srcs[i] = EntrySrc::F32(t);
                shapes.insert(name, dims);
            }
        }
        // tied lm_head alias
        if tied_lm_head {
            let ie = self.index["embed_tokens.weight"];
            let il = self.index["lm_head.weight"];
            self.entries[il].dims = self.entries[ie].dims.clone();
            self.entries[il].size = self.entries[ie].size;
            self.srcs[il] = EntrySrc::Alias(ie);
            let d = self.entries[ie].dims.clone();
            shapes.insert("lm_head.weight".to_string(), d);
        }
        self.derive_config(&shapes, kept_layers);
        self.print_audit(&shapes);
    }

    /// Real architecture dims from the actual tensor shapes (ground truth),
    /// cross-checked against the HF config.json when available.
    fn derive_config(&mut self, shapes: &HashMap<String, Vec<u32>>, kept_layers: &[usize]) {
        let c = &mut self.config;
        let embed = shapes.get("embed_tokens.weight").expect("embed_tokens.weight missing");
        c.vocab = embed[0] as usize;
        c.d = embed[1] as usize;
        c.n_layers = self.n_layers;
        c.n_experts = self.n_experts;
        // per-layer types restricted to the model (used by is_mla/is_moe)
        let _ = kept_layers;

        let get = |n: &str| shapes.get(n).cloned();
        match self.arch {
            StArch::K3 => {
                let kl = (0..self.n_layers).find(|&l| get(&format!("layers.{}.self_attn.q_proj.weight", l)).is_some());
                if let Some(l) = kl {
                    let p = format!("layers.{}.", l);
                    let kda_proj = get(&format!("{}self_attn.q_proj.weight", p)).unwrap()[0] as usize;
                    c.kda_heads = get(&format!("{}self_attn.b_proj.weight", p)).unwrap()[0] as usize;
                    c.kda_dim = get(&format!("{}self_attn.o_norm.weight", p)).unwrap()[0] as usize;
                    assert_eq!(c.kda_heads * c.kda_dim, kda_proj, "kda dims inconsistent");
                    c.kda_conv = get(&format!("{}self_attn.q_conv1d.weight", p)).unwrap()[1] as usize;
                    c.kda_fa = get(&format!("{}self_attn.f_a_proj.weight", p)).unwrap()[0] as usize;
                }
                let ml = (0..self.n_layers).find(|&l| get(&format!("layers.{}.self_attn.q_a_proj.weight", l)).is_some());
                if let Some(l) = ml {
                    let p = format!("layers.{}.", l);
                    c.mla_qa = get(&format!("{}self_attn.q_a_proj.weight", p)).unwrap()[0] as usize;
                    c.mla_kva = get(&format!("{}self_attn.kv_a_layernorm.weight", p)).unwrap()[0] as usize;
                    let kv_a_rows = get(&format!("{}self_attn.kv_a_proj_with_mqa.weight", p)).unwrap()[0] as usize;
                    c.mla_rope = kv_a_rows - c.mla_kva;
                    let qb_rows = get(&format!("{}self_attn.q_b_proj.weight", p)).unwrap()[0] as usize;
                    let kvb_rows = get(&format!("{}self_attn.kv_b_proj.weight", p)).unwrap()[0] as usize;
                    // q_b = H*(nope+rope), kv_b = H*(nope+v). Prefer the HF
                    // config values; otherwise assume v == kda_dim and solve.
                    let (h, nope, v) = if let Some(j) = &self.hf_config {
                        let num = |k: &str| j.get(k).and_then(|x| x.as_num()).map(|n| n as usize);
                        (num("num_attention_heads"), num("qk_nope_head_dim"), num("v_head_dim"))
                    } else {
                        (None, None, None)
                    };
                    match (h, nope, v) {
                        (Some(h), Some(nope), Some(v)) => {
                            c.mla_heads = h;
                            c.mla_nope = nope;
                            c.mla_v = v;
                        }
                        _ => {
                            c.mla_v = c.kda_dim;
                            assert_eq!((kvb_rows as isize - qb_rows as isize) % (c.mla_v as isize - c.mla_rope as isize), 0, "MLA dims underdetermined without config.json");
                            c.mla_heads = (kvb_rows - qb_rows) / (c.mla_v - c.mla_rope);
                            assert_eq!(qb_rows % (c.mla_v + c.mla_rope), 0, "MLA q_b rows inconsistent");
                            c.mla_nope = qb_rows / c.mla_heads - c.mla_rope;
                        }
                    }
                    assert_eq!(c.mla_qb(), qb_rows, "MLA q_b rows mismatch");
                    assert_eq!(c.mla_kvb(), kvb_rows, "MLA kv_b rows mismatch");
                }
                let mo = (0..self.n_layers).find(|&l| get(&format!("layers.{}.block_sparse_moe.gate.weight", l)).is_some());
                if let Some(l) = mo {
                    let p = format!("layers.{}.block_sparse_moe.", l);
                    assert_eq!(get(&format!("{}gate.weight", p)).unwrap()[0] as usize, c.n_experts, "router rows != expert count");
                    c.routed_hidden = get(&format!("{}routed_expert_norm.weight", p)).unwrap()[0] as usize;
                    c.shared_inter = get(&format!("{}shared_experts.gate_proj.weight", p)).unwrap()[0] as usize;
                    let w1 = get(&format!("{}experts.0.w1", p)).unwrap();
                    c.moe_inter = w1[0] as usize;
                    assert_eq!(w1[1] as usize, c.routed_hidden, "expert w1 cols != routed_hidden");
                    let w2 = get(&format!("{}experts.0.w2", p)).unwrap();
                    assert_eq!((w2[0] as usize, w2[1] as usize), (c.routed_hidden, c.moe_inter), "expert w2 shape mismatch");
                }
                let dl = (0..self.n_layers).find(|&l| get(&format!("layers.{}.mlp.gate_proj.weight", l)).is_some());
                if let Some(l) = dl {
                    c.dense_inter = get(&format!("layers.{}.mlp.gate_proj.weight", l)).unwrap()[0] as usize;
                }
                if let Some(j) = &self.hf_config {
                    if let Some(sp_bos) = j.get("bos_token_id").and_then(|x| x.as_num()) {
                        c.bos_id = sp_bos as u32;
                    }
                    if let Some(sp_eos) = j.get("eos_token_id").and_then(|x| x.as_num()) {
                        c.eos_id = sp_eos as u32;
                    }
                }
            }
            StArch::Dense => {
                let dl = (0..self.n_layers).find(|&l| get(&format!("layers.{}.mlp.gate_proj.weight", l)).is_some());
                c.dense_inter = dl.and_then(|l| get(&format!("layers.{}.mlp.gate_proj.weight", l))).map(|d| d[0] as usize).unwrap_or(0);
                c.n_experts = 0;
                c.top_k = 0;
                c.n_shared = 0;
                c.first_k_dense = self.n_layers; // no MoE anywhere
                c.mla_layers = Some(Vec::new());
                c.dense_layers = Some((0..self.n_layers).collect());
                if let Some(j) = &self.hf_config {
                    let num = |k: &str| j.get(k).and_then(|x| x.as_num());
                    if let Some(b) = num("bos_token_id") {
                        c.bos_id = b as u32;
                    }
                    if let Some(e) = num("eos_token_id") {
                        c.eos_id = e as u32;
                    }
                    if let Some(eps) = num("rms_norm_eps") {
                        c.rms_eps = eps as f32;
                    }
                }
            }
        }
        self.crosscheck();
    }

    /// Prints config.json vs tensor-shape mismatches (shapes always win).
    fn crosscheck(&self) {
        let Some(j) = &self.hf_config else {
            println!("config.json: not available - scalar fields use K3 defaults, dims come from tensor shapes");
            return;
        };
        let c = &self.config;
        let mut bad: Vec<String> = Vec::new();
        let mut n_ok = 0usize;
        let mut chk = |field: &str, want: Option<f64>, got: usize| {
            if let Some(w) = want {
                if w as usize == got {
                    n_ok += 1;
                } else {
                    bad.push(format!("{}: config.json={} shapes={} (shapes win)", field, w as usize, got));
                }
            }
        };
        chk("hidden_size", j.get("hidden_size").and_then(|x| x.as_num()), c.d);
        chk("vocab_size", j.get("vocab_size").and_then(|x| x.as_num()), c.vocab);
        chk("num_hidden_layers", j.get("num_hidden_layers").and_then(|x| x.as_num()), c.n_layers);
        chk("intermediate_size", j.get("intermediate_size").and_then(|x| x.as_num()), c.dense_inter);
        if self.arch == StArch::K3 {
            chk("num_experts", j.get("num_experts").and_then(|x| x.as_num()), c.n_experts);
            chk("q_lora_rank", j.get("q_lora_rank").and_then(|x| x.as_num()), c.mla_qa);
            chk("kv_lora_rank", j.get("kv_lora_rank").and_then(|x| x.as_num()), c.mla_kva);
            chk("qk_nope_head_dim", j.get("qk_nope_head_dim").and_then(|x| x.as_num()), c.mla_nope);
            chk("qk_rope_head_dim", j.get("qk_rope_head_dim").and_then(|x| x.as_num()), c.mla_rope);
            chk("v_head_dim", j.get("v_head_dim").and_then(|x| x.as_num()), c.mla_v);
            chk("num_attention_heads", j.get("num_attention_heads").and_then(|x| x.as_num()), c.mla_heads);
            chk("moe_intermediate_size", j.get("moe_intermediate_size").and_then(|x| x.as_num()), c.moe_inter);
            chk("routed_expert_hidden_size", j.get("routed_expert_hidden_size").and_then(|x| x.as_num()), c.routed_hidden);
            if let Some(la) = j.get("linear_attn_config") {
                chk("linear.num_heads", la.get("num_heads").and_then(|x| x.as_num()), c.kda_heads);
                chk("linear.head_dim", la.get("head_dim").and_then(|x| x.as_num()), c.kda_dim);
                chk("linear.short_conv_kernel_size", la.get("short_conv_kernel_size").and_then(|x| x.as_num()), c.kda_conv);
            }
        }
        if bad.is_empty() {
            println!("config.json cross-check: {} fields match the tensor shapes", n_ok);
        } else {
            for b in &bad {
                println!("  config.json MISMATCH: {}", b);
            }
        }
    }

    fn print_audit(&self, shapes: &HashMap<String, Vec<u32>>) {
        let c = &self.config;
        let n_kda = self.layer_kind.iter().filter(|&&k| k == Some(LayerKind::Kda)).count();
        let n_mla = self.layer_kind.iter().filter(|&&k| k == Some(LayerKind::Mla)).count();
        let n_moe = self.layer_moe.iter().filter(|&&m| m).count();
        match self.arch {
            StArch::K3 => println!(
                "K3: {} layers ({} KDA + {} MLA), {} MoE + {} dense, hidden {}, vocab {}, {} experts top-{} + {} shared",
                c.n_layers, n_kda, n_mla, n_moe, c.n_layers - n_moe, c.d, c.vocab, c.n_experts, c.top_k, c.n_shared
            ),
            StArch::Dense => println!("dense arch: {} layers, hidden {}, vocab {}", c.n_layers, c.d, c.vocab),
        }
        let show = |n: &str| {
            if let Some(d) = shapes.get(n) {
                println!("    {:<58} {:?}", n, d);
            }
        };
        if self.arch == StArch::K3 {
            let kl = (0..self.n_layers).find(|&l| self.layer_kind[l] == Some(LayerKind::Kda)).unwrap_or(0);
            let ml = (0..self.n_layers).find(|&l| self.layer_kind[l] == Some(LayerKind::Mla)).unwrap_or(0);
            let mo = (0..self.n_layers).find(|&l| self.layer_moe[l]).unwrap_or(1);
            show("embed_tokens.weight");
            show("lm_head.weight");
            show(&format!("layers.{}.self_attn.q_proj.weight", kl));
            show(&format!("layers.{}.self_attn.q_conv1d.weight", kl));
            show(&format!("layers.{}.self_attn.q_a_proj.weight", ml));
            show(&format!("layers.{}.self_attn.kv_a_proj_with_mqa.weight", ml));
            show(&format!("layers.{}.self_attn.kv_b_proj.weight", ml));
            show(&format!("layers.{}.block_sparse_moe.gate.weight", mo));
            show(&format!("layers.{}.block_sparse_moe.experts.0.w1", mo));
            show(&format!("layers.{}.block_sparse_moe.experts.0.w2", mo));
            show(&format!("layers.{}.block_sparse_moe.experts.0.w3", mo));
        } else {
            show("embed_tokens.weight");
            show("lm_head.weight");
            show("layers.0.self_attn.q_proj.weight");
            show("layers.0.self_attn.k_proj.weight");
            show("layers.0.mlp.gate_proj.weight");
        }
    }

    fn raw_range(&self, t: &TRef, off: u64, len: u64) -> Vec<u8> {
        match &self.shards[t.shard] {
            ShardLoc::Local(p) => {
                let f = std::fs::File::open(p).unwrap();
                use std::os::unix::fs::FileExt;
                let mut buf = vec![0u8; len as usize];
                f.read_exact_at(&mut buf, t.start + off).unwrap();
                buf
            }
            ShardLoc::Remote(url) => {
                let start = t.start + off;
                if let Some(buf) = mirror_range(url, start, len) {
                    return buf;
                }
                crate::http::fetch_range(url, Some((start, start + len - 1)))
                    .unwrap_or_else(|| panic!("range fetch failed on {}", url))
            }
        }
    }

    fn to_f32(raw: &[u8], dtype: &str) -> Vec<f32> {
        match dtype {
            "BF16" => safetensors::bf16_slice_to_f32(raw),
            "F32" => raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            dt => panic!("unhandled dtype {}", dt),
        }
    }
    fn cache_file(&self, name: &str) -> std::path::PathBuf {
        let safe: String = name.chars().map(|c| if c == '.' || c == '/' { '_' } else { c }).collect();
        self.cache_dir.as_ref().unwrap().join(format!("{}.f32", safe))
    }

    /// Materializes the converted f32 bytes of a tensor into the disk cache
    /// (remote + double pass only). Chunked fetch: peak RAM stays ~64 MB.
    fn materialize(&self, name: &str, t: &TRef, rows: usize, cols: usize) -> std::path::PathBuf {
        let path = self.cache_file(name);
        if path.exists() {
            return path;
        }
        let bpe: u64 = if t.dtype == "BF16" { 2 } else { 4 };
        let row_bytes = cols as u64 * bpe;
        let per = ((1u64 << 26) / row_bytes.max(1)).max(1) as usize; // ~64 MB raw per request
        let tmp = path.with_extension("partial");
        let mut f = std::io::BufWriter::with_capacity(1 << 22, std::fs::File::create(&tmp).unwrap());
        use std::io::Write;
        let mut r0 = 0usize;
        while r0 < rows {
            let r1 = (r0 + per).min(rows);
            let raw = self.raw_range(t, r0 as u64 * row_bytes, (r1 - r0) as u64 * row_bytes);
            let vals = Self::to_f32(&raw, &t.dtype);
            f.write_all(&crate::weights::f32_to_bytes(&vals)).unwrap();
            r0 = r1;
        }
        drop(f);
        std::fs::rename(&tmp, &path).unwrap();
        println!(
            "  cached {} ({} rows, {}, total fetched {})",
            name,
            rows,
            t.dtype,
            gb(crate::http::fetched_bytes())
        );
        path
    }

    /// Rows r0..r1 of an f32-logical tensor (whole row width), converted.
    pub fn f32_rows(&self, e: &DirEntry, r0: usize, r1: usize) -> Vec<f32> {
        let i = self.index[&e.name];
        match &self.srcs[i] {
            EntrySrc::Alias(j) => self.f32_rows(&self.entries[*j], r0, r1),
            EntrySrc::F32(t) => {
                let (rows, cols) = if e.dims.len() <= 1 { (1usize, e.dims[0] as usize) } else { (e.dims[0] as usize, e.dims[1..].iter().map(|&d| d as usize).product()) };
                assert!(r1 <= rows, "{}: row range out of bounds", e.name);
                if self.remote && self.cache_enabled.load(Ordering::Relaxed) {
                    let path = self.materialize(&e.name, t, rows, cols);
                    let f = std::fs::File::open(&path).unwrap();
                    use std::os::unix::fs::FileExt;
                    let mut buf = vec![0u8; (r1 - r0) * cols * 4];
                    f.read_exact_at(&mut buf, (r0 * cols * 4) as u64).unwrap();
                    return Self::to_f32(&buf, "F32");
                }
                let bpe: u64 = if t.dtype == "BF16" { 2 } else { 4 };
                let raw = self.raw_range(t, r0 as u64 * cols as u64 * bpe, (r1 - r0) as u64 * cols as u64 * bpe);
                Self::to_f32(&raw, &t.dtype)
            }
            EntrySrc::Expert { .. } => panic!("{}: f32_rows on an MXFP4 expert", e.name),
            EntrySrc::Unresolved => panic!("{}: tensor of a pruned layer accessed", e.name),
        }
    }

    /// .bin-layout blob for Copy/Expert roles: converted f32 bytes, or the
    /// MXFP4 packed ++ scales concatenation (fetched once, kept experts only).
    pub fn raw_blob(&self, e: &DirEntry) -> Vec<u8> {
        let i = self.index[&e.name];
        match &self.srcs[i] {
            EntrySrc::Alias(j) => self.raw_blob(&self.entries[*j]),
            EntrySrc::F32(t) => {
                let raw = self.raw_range(t, 0, t.len);
                let vals = Self::to_f32(&raw, &t.dtype);
                crate::weights::f32_to_bytes(&vals)
            }
            EntrySrc::Expert { packed, scale } => {
                let mut blob = self.raw_range(packed, 0, packed.len);
                blob.extend_from_slice(&self.raw_range(scale, 0, scale.len));
                blob
            }
            EntrySrc::Unresolved => panic!("{}: tensor of a pruned layer accessed", e.name),
        }
    }

    /// Only the scale bytes of an MXFP4 expert (1/17 of the data): enough to
    /// rank experts by weight magnitude without fetching the packed nibbles.
    pub fn expert_scales(&self, e: &DirEntry) -> Vec<u8> {
        let i = self.index[&e.name];
        match &self.srcs[i] {
            EntrySrc::Expert { scale, .. } => self.raw_range(scale, 0, scale.len),
            _ => panic!("{}: not an MXFP4 expert", e.name),
        }
    }
}

impl Drop for StDir {
    fn drop(&mut self) {
        if let Some(d) = &self.cache_dir {
            std::fs::remove_dir_all(d).ok();
        }
    }
}
