//! External model adapter packs (`MKADAPT1`).
//!
//! A pack contains standard low-rank updates for named fp32 matrices:
//!
//!     W <- W + scale * B * A
//!
//! Packs are bound to the exact base model by SHA-256. They are folded into
//! MAP_PRIVATE pages (or the in-memory fallback) during load, so the base file
//! stays unchanged and inference has no adapter-branch overhead. Several
//! packs compose additively in pack-digest order. Packed expert tensors are
//! intentionally rejected because adding a float delta would require a new
//! quantization decision.

use crate::json::Json;
use crate::model::pool::{pool, Job, MPtr, SPtr};
use crate::quant::weights::{BinFile, DTYPE_F32};
use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

const MAGIC: &[u8; 8] = b"MKADAPT1";
const MAX_MANIFEST: usize = 16 << 20;
const ROOT_FIELDS: &[&str] = &["format", "name", "base_sha256", "fold", "targets"];
const TARGET_FIELDS: &[&str] = &[
    "tensor", "out", "in", "rank", "scale", "a_offset", "a_bytes", "a_sha256", "b_offset",
    "b_bytes", "b_sha256",
];

#[derive(Debug)]
struct Update {
    tensor: String,
    out: usize,
    input: usize,
    rank: usize,
    scale: f64,
    a: Vec<f32>,
    b: Vec<f32>,
}

#[derive(Debug)]
struct Pack {
    name: String,
    digest: String,
    base_sha256: String,
    updates: Vec<Update>,
}

#[derive(Debug, Clone, Default)]
pub(super) struct AppliedPacks {
    pub(super) count: usize,
    pub(super) set_sha256: Option<String>,
}

impl AppliedPacks {
    pub(super) fn is_empty(&self) -> bool {
        self.count == 0
    }
}

fn object<'a>(value: &'a Json, where_: &str) -> Result<&'a Vec<(String, Json)>, String> {
    let Json::Obj(pairs) = value else {
        return Err(format!("{} must be an object", where_));
    };
    let mut seen = HashSet::new();
    for (key, _) in pairs {
        if !seen.insert(key) {
            return Err(format!("{} contains duplicate field {:?}", where_, key));
        }
    }
    Ok(pairs)
}

fn exact_fields(pairs: &[(String, Json)], allowed: &[&str], where_: &str) -> Result<(), String> {
    let present: HashSet<&str> = pairs.iter().map(|(key, _)| key.as_str()).collect();
    for &field in allowed {
        if !present.contains(field) {
            return Err(format!("{} is missing field {:?}", where_, field));
        }
    }
    let unknown: Vec<&str> = present
        .into_iter()
        .filter(|field| !allowed.contains(field))
        .collect();
    if !unknown.is_empty() {
        return Err(format!(
            "{} contains unsupported fields: {}",
            where_,
            unknown.join(", ")
        ));
    }
    Ok(())
}

fn field<'a>(pairs: &'a [(String, Json)], key: &str, where_: &str) -> Result<&'a Json, String> {
    pairs
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value)
        .ok_or_else(|| format!("{} is missing field {:?}", where_, key))
}

fn string(pairs: &[(String, Json)], key: &str, where_: &str) -> Result<String, String> {
    let value = field(pairs, key, where_)?
        .as_str()
        .ok_or_else(|| format!("{}.{} must be a string", where_, key))?;
    if value.is_empty() || value.len() > 1024 || value.chars().any(char::is_control) {
        return Err(format!(
            "{}.{} must be a non-empty string without control characters",
            where_, key
        ));
    }
    Ok(value.to_string())
}

fn number(pairs: &[(String, Json)], key: &str, where_: &str) -> Result<f64, String> {
    let value = field(pairs, key, where_)?
        .as_num()
        .ok_or_else(|| format!("{}.{} must be a number", where_, key))?;
    if !value.is_finite() {
        return Err(format!("{}.{} must be finite", where_, key));
    }
    Ok(value)
}

fn usize_field(pairs: &[(String, Json)], key: &str, where_: &str) -> Result<usize, String> {
    let value = number(pairs, key, where_)?;
    if value < 0.0 || value.fract() != 0.0 || value > usize::MAX as f64 {
        return Err(format!("{}.{} must be a non-negative integer", where_, key));
    }
    Ok(value as usize)
}

fn checked_bytes(a: usize, b: usize, where_: &str) -> Result<usize, String> {
    a.checked_mul(b)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| format!("{} dimensions overflow", where_))
}

unsafe fn mutable_f32_slice<'a>(pointer: MPtr, len: usize) -> &'a mut [f32] {
    unsafe { std::slice::from_raw_parts_mut(pointer.0, len) }
}

fn decode_f32(blob: &[u8], where_: &str) -> Result<Vec<f32>, String> {
    if blob.len() % 4 != 0 {
        return Err(format!("{} byte length is not divisible by four", where_));
    }
    let values: Vec<f32> = blob
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{} contains a non-finite factor", where_));
    }
    Ok(values)
}

fn parse_pack(path: &str) -> Result<Pack, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {}", path, e))?;
    if bytes.len() < 12 || &bytes[..8] != MAGIC {
        return Err(format!("{}: bad magic (expected MKADAPT1)", path));
    }
    let manifest_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    if manifest_len == 0 || manifest_len > MAX_MANIFEST || 12 + manifest_len > bytes.len() {
        return Err(format!(
            "{}: invalid manifest length {}",
            path, manifest_len
        ));
    }
    let manifest_bytes = &bytes[12..12 + manifest_len];
    let manifest_text = std::str::from_utf8(manifest_bytes)
        .map_err(|_| format!("{}: manifest is not UTF-8", path))?;
    if manifest_text.trim_end().len() != manifest_text.len() {
        return Err(format!(
            "{}: manifest must not contain trailing whitespace",
            path
        ));
    }
    let root = crate::json::parse_complete(manifest_bytes);
    let root_pairs = object(&root, "manifest")?;
    exact_fields(root_pairs, ROOT_FIELDS, "manifest")?;
    if usize_field(root_pairs, "format", "manifest")? != 1 {
        return Err(format!("{}: unsupported adapter-pack format", path));
    }
    let name = string(root_pairs, "name", "manifest")?;
    if name.len() > 128 {
        return Err("manifest.name is longer than 128 bytes".to_string());
    }
    let base_sha256 = string(root_pairs, "base_sha256", "manifest")?;
    if !crate::sha256::is_lower_hex_digest(&base_sha256) {
        return Err("manifest.base_sha256 must be a lowercase SHA-256 digest".to_string());
    }
    if string(root_pairs, "fold", "manifest")? != "f32_ba_v1" {
        return Err("manifest.fold must be f32_ba_v1".to_string());
    }
    let targets = field(root_pairs, "targets", "manifest")?
        .as_arr()
        .ok_or_else(|| "manifest.targets must be an array".to_string())?;
    if targets.is_empty() {
        return Err("manifest.targets must not be empty".to_string());
    }
    let payload = &bytes[12 + manifest_len..];
    let mut cursor = 0usize;
    let mut previous = None::<String>;
    let mut updates = Vec::with_capacity(targets.len());
    for (index, value) in targets.iter().enumerate() {
        let where_ = format!("manifest.targets[{}]", index);
        let pairs = object(value, &where_)?;
        exact_fields(pairs, TARGET_FIELDS, &where_)?;
        let tensor = string(pairs, "tensor", &where_)?;
        if previous.as_ref().is_some_and(|old| old >= &tensor) {
            return Err("manifest targets must be unique and sorted by tensor name".to_string());
        }
        previous = Some(tensor.clone());
        let out = usize_field(pairs, "out", &where_)?;
        let input = usize_field(pairs, "in", &where_)?;
        let rank = usize_field(pairs, "rank", &where_)?;
        if out == 0 || input == 0 || rank == 0 {
            return Err(format!("{} dimensions must be positive", where_));
        }
        let scale = number(pairs, "scale", &where_)?;
        if scale == 0.0 || scale.abs() > 1.0e6 {
            return Err(format!(
                "{}.scale must be nonzero with magnitude <= 1e6",
                where_
            ));
        }
        let a_offset = usize_field(pairs, "a_offset", &where_)?;
        let a_bytes = usize_field(pairs, "a_bytes", &where_)?;
        let b_offset = usize_field(pairs, "b_offset", &where_)?;
        let b_bytes = usize_field(pairs, "b_bytes", &where_)?;
        if a_offset != cursor || a_bytes != checked_bytes(rank, input, &where_)? {
            return Err(format!("{} has a non-canonical A payload range", where_));
        }
        cursor = cursor
            .checked_add(a_bytes)
            .ok_or_else(|| "payload offset overflow".to_string())?;
        if b_offset != cursor || b_bytes != checked_bytes(out, rank, &where_)? {
            return Err(format!("{} has a non-canonical B payload range", where_));
        }
        cursor = cursor
            .checked_add(b_bytes)
            .ok_or_else(|| "payload offset overflow".to_string())?;
        if cursor > payload.len() {
            return Err(format!("{} payload extends beyond the file", where_));
        }
        let a_blob = &payload[a_offset..a_offset + a_bytes];
        let b_blob = &payload[b_offset..b_offset + b_bytes];
        let a_hash = string(pairs, "a_sha256", &where_)?;
        let b_hash = string(pairs, "b_sha256", &where_)?;
        if !crate::sha256::is_lower_hex_digest(&a_hash)
            || crate::sha256::hex(&crate::sha256::digest(a_blob)) != a_hash
        {
            return Err(format!("{} A SHA-256 mismatch", where_));
        }
        if !crate::sha256::is_lower_hex_digest(&b_hash)
            || crate::sha256::hex(&crate::sha256::digest(b_blob)) != b_hash
        {
            return Err(format!("{} B SHA-256 mismatch", where_));
        }
        updates.push(Update {
            tensor,
            out,
            input,
            rank,
            scale,
            a: decode_f32(a_blob, &format!("{} A", where_))?,
            b: decode_f32(b_blob, &format!("{} B", where_))?,
        });
    }
    if cursor != payload.len() {
        return Err(format!(
            "{}: trailing bytes after the canonical payload",
            path
        ));
    }
    Ok(Pack {
        name,
        digest: crate::sha256::hex(&crate::sha256::digest(&bytes)),
        base_sha256,
        updates,
    })
}

/// Verifies and folds external packs into the private model mapping.
/// The on-disk model is never opened for writing.
pub(super) fn apply_packs(
    model_path: &str,
    bin: &mut BinFile,
    pack_paths: &[String],
) -> Result<AppliedPacks, String> {
    if pack_paths.is_empty() {
        return Ok(AppliedPacks::default());
    }
    let mut packs: Vec<Pack> = pack_paths
        .iter()
        .map(|path| parse_pack(path))
        .collect::<Result<_, _>>()?;
    packs.sort_by(|left, right| left.digest.cmp(&right.digest));
    for pair in packs.windows(2) {
        if pair[0].digest == pair[1].digest {
            return Err(format!(
                "adapter pack {} was supplied more than once",
                pair[0].name
            ));
        }
    }

    println!(
        "adapter packs: hashing base model for {} pack(s)",
        packs.len()
    );
    let base_sha256 = crate::sha256::hex(&crate::sha256::digest_file(model_path)?);
    for pack in &packs {
        if pack.base_sha256 != base_sha256 {
            return Err(format!(
                "{} belongs to base {}, loaded model is {}",
                pack.name, pack.base_sha256, base_sha256
            ));
        }
    }

    let mut groups: BTreeMap<String, Vec<(usize, usize)>> = BTreeMap::new();
    for (pack_index, pack) in packs.iter().enumerate() {
        println!(
            "adapter pack: {} ({} targets, sha256 {}...)",
            pack.name,
            pack.updates.len(),
            &pack.digest[..12]
        );
        for (update_index, update) in pack.updates.iter().enumerate() {
            let entry = bin.entries.get(&update.tensor).ok_or_else(|| {
                format!("{}: base model has no tensor {}", pack.name, update.tensor)
            })?;
            if entry.dtype != DTYPE_F32 {
                return Err(format!(
                    "{}: target {} has dtype {}, only fp32 targets are supported",
                    pack.name, update.tensor, entry.dtype
                ));
            }
            if entry.dims != vec![update.out as u32, update.input as u32] {
                return Err(format!(
                    "{}: target {} declares [{}, {}], base has {:?}",
                    pack.name, update.tensor, update.out, update.input, entry.dims
                ));
            }
            groups
                .entry(update.tensor.clone())
                .or_default()
                .push((pack_index, update_index));
        }
    }

    for (tensor, refs) in &groups {
        let entry = bin.entries.get(tensor).unwrap().clone();
        let start = entry.offset as usize;
        let end = start
            .checked_add(entry.size as usize)
            .ok_or_else(|| format!("{} byte range overflows", tensor))?;
        let data = bin.data.bytes_mut();
        if end > data.len() {
            return Err(format!(
                "{} byte range is outside the loaded model backing",
                tensor
            ));
        }
        let (prefix, weight, suffix) = unsafe { data[start..end].align_to_mut::<f32>() };
        if !prefix.is_empty() || !suffix.is_empty() {
            return Err(format!("{} is not aligned as fp32", tensor));
        }
        let first = &packs[refs[0].0].updates[refs[0].1];
        let (out, input) = (first.out, first.input);
        if weight.len() != out * input {
            return Err(format!(
                "{} payload size does not match its dimensions",
                tensor
            ));
        }
        let updates: Vec<(usize, f64, SPtr, SPtr)> = refs
            .iter()
            .map(|&(pack_index, update_index)| {
                let update = &packs[pack_index].updates[update_index];
                (
                    update.rank,
                    update.scale,
                    SPtr(update.a.as_ptr()),
                    SPtr(update.b.as_ptr()),
                )
            })
            .collect();
        let failed = Arc::new(AtomicBool::new(false));
        let job_count = pool().workers.min(out).max(1);
        let rows_per_job = out.div_ceil(job_count);
        let mut jobs: Vec<Job> = Vec::with_capacity(job_count);
        for row_start in (0..out).step_by(rows_per_job) {
            let row_end = (row_start + rows_per_job).min(out);
            let row_count = row_end - row_start;
            let output = MPtr(unsafe { weight.as_mut_ptr().add(row_start * input) });
            let updates = updates.clone();
            let failed = failed.clone();
            jobs.push(Box::new(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    // Every job owns a disjoint range of output rows. Factor
                    // pointers stay valid until the pool barrier below returns.
                    let output = unsafe { mutable_f32_slice(output, row_count * input) };
                    let mut accumulated = vec![0.0f64; input];
                    for local_row in 0..row_count {
                        let row = row_start + local_row;
                        let row_weight = &mut output[local_row * input..(local_row + 1) * input];
                        for column in 0..input {
                            accumulated[column] = row_weight[column] as f64;
                        }
                        for &(rank, scale, a_ptr, b_ptr) in &updates {
                            let a = unsafe { std::slice::from_raw_parts(a_ptr.0, rank * input) };
                            let b = unsafe { std::slice::from_raw_parts(b_ptr.0, out * rank) };
                            for rank_index in 0..rank {
                                let coefficient = scale * b[row * rank + rank_index] as f64;
                                let a_row = &a[rank_index * input..(rank_index + 1) * input];
                                for column in 0..input {
                                    accumulated[column] += coefficient * a_row[column] as f64;
                                }
                            }
                        }
                        for column in 0..input {
                            let value = accumulated[column] as f32;
                            if !value.is_finite() {
                                failed.store(true, Ordering::Relaxed);
                            }
                            row_weight[column] = value;
                        }
                    }
                }));
                if result.is_err() {
                    failed.store(true, Ordering::Relaxed);
                }
            }));
        }
        pool().run(jobs);
        if failed.load(Ordering::Relaxed) {
            return Err(format!("{} fold produced a non-finite weight", tensor));
        }
    }

    let mut set_hasher = crate::sha256::Sha256::new();
    set_hasher.update(b"microkimi.adapter-set.v1\0");
    set_hasher.update(base_sha256.as_bytes());
    for pack in &packs {
        set_hasher.update(pack.digest.as_bytes());
    }
    let set_sha256 = crate::sha256::hex(&set_hasher.finalize());
    let updates: usize = packs.iter().map(|pack| pack.updates.len()).sum();
    println!(
        "adapter packs: folded {} updates into {} tensors (set {}..., base file unchanged)",
        updates,
        groups.len(),
        &set_sha256[..12]
    );
    Ok(AppliedPacks {
        count: packs.len(),
        set_sha256: Some(set_sha256),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::quant::weights::{f32_to_bytes, Backing, Entry};
    use std::collections::HashMap;

    fn temp(name: &str) -> String {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir()
            .join(format!(
                "microkimi_adapter_{}_{}_{}",
                std::process::id(),
                nonce,
                name
            ))
            .to_string_lossy()
            .into_owned()
    }

    fn pack_bytes(
        name: &str,
        base_sha: &str,
        tensor: &str,
        scale: f64,
        a: &[f32],
        b: &[f32],
        out: usize,
        input: usize,
        rank: usize,
    ) -> Vec<u8> {
        let ab = f32_to_bytes(a);
        let bb = f32_to_bytes(b);
        let manifest = format!(
            "{{\"format\":1,\"name\":\"{}\",\"base_sha256\":\"{}\",\"fold\":\"f32_ba_v1\",\"targets\":[{{\"tensor\":\"{}\",\"out\":{},\"in\":{},\"rank\":{},\"scale\":{},\"a_offset\":0,\"a_bytes\":{},\"a_sha256\":\"{}\",\"b_offset\":{},\"b_bytes\":{},\"b_sha256\":\"{}\"}}]}}",
            name,
            base_sha,
            tensor,
            out,
            input,
            rank,
            scale,
            ab.len(),
            crate::sha256::hex(&crate::sha256::digest(&ab)),
            ab.len(),
            bb.len(),
            crate::sha256::hex(&crate::sha256::digest(&bb)),
        );
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&(manifest.len() as u32).to_le_bytes());
        bytes.extend_from_slice(manifest.as_bytes());
        bytes.extend_from_slice(&ab);
        bytes.extend_from_slice(&bb);
        bytes
    }

    fn bin(weight: &[f32]) -> BinFile {
        BinFile {
            data: Backing::Vec(f32_to_bytes(weight)),
            entries: HashMap::from([(
                "linear.weight".to_string(),
                Entry {
                    dtype: DTYPE_F32,
                    dims: vec![2, 3],
                    offset: 0,
                    size: 24,
                },
            )]),
            config: Config::microkimi(),
        }
    }

    fn values(bin: &BinFile) -> Vec<f32> {
        bin.data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn folds_two_packs_in_digest_order_and_keeps_the_base_file() {
        let base_path = temp("base.bin");
        let p1 = temp("one.mkap");
        let p2 = temp("two.mkap");
        let base_file = b"the exact base model bytes";
        std::fs::write(&base_path, base_file).unwrap();
        let base_sha = crate::sha256::hex(&crate::sha256::digest(base_file));
        std::fs::write(
            &p1,
            pack_bytes(
                "one",
                &base_sha,
                "linear.weight",
                0.5,
                &[1.0, 2.0, 3.0],
                &[4.0, 5.0],
                2,
                3,
                1,
            ),
        )
        .unwrap();
        std::fs::write(
            &p2,
            pack_bytes(
                "two",
                &base_sha,
                "linear.weight",
                -0.25,
                &[2.0, 0.0, 1.0],
                &[1.0, -2.0],
                2,
                3,
                1,
            ),
        )
        .unwrap();
        let initial = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let mut left = bin(&initial);
        let applied = apply_packs(&base_path, &mut left, &[p1.clone(), p2.clone()]).unwrap();
        assert_eq!(applied.count, 2);
        assert!(applied.set_sha256.is_some());
        let mut right = bin(&initial);
        apply_packs(&base_path, &mut right, &[p2.clone(), p1.clone()]).unwrap();
        assert_eq!(values(&left), values(&right));
        let got = values(&left);
        let want = [11.5, 24.0, 35.75, 43.5, 55.0, 68.0];
        for (actual, expected) in got.iter().zip(want) {
            assert!(
                (actual - expected).abs() < 1e-6,
                "{} vs {}",
                actual,
                expected
            );
        }
        assert_eq!(std::fs::read(&base_path).unwrap(), base_file);
        for path in [base_path, p1, p2] {
            std::fs::remove_file(path).ok();
        }
    }

    #[test]
    fn rejects_a_pack_for_another_base() {
        let base_path = temp("wrong-base.bin");
        let pack_path = temp("wrong.mkap");
        std::fs::write(&base_path, b"base A").unwrap();
        let other = crate::sha256::hex(&crate::sha256::digest(b"base B"));
        std::fs::write(
            &pack_path,
            pack_bytes(
                "wrong",
                &other,
                "linear.weight",
                1.0,
                &[1.0, 1.0, 1.0],
                &[1.0, 1.0],
                2,
                3,
                1,
            ),
        )
        .unwrap();
        let error = apply_packs(&base_path, &mut bin(&[0.0; 6]), &[pack_path.clone()]).unwrap_err();
        assert!(error.contains("belongs to base"), "{}", error);
        std::fs::remove_file(base_path).ok();
        std::fs::remove_file(pack_path).ok();
    }

    #[test]
    fn rejects_corrupted_factor_bytes() {
        let base_path = temp("corrupt-base.bin");
        let pack_path = temp("corrupt.mkap");
        std::fs::write(&base_path, b"base").unwrap();
        let base_sha = crate::sha256::hex(&crate::sha256::digest(b"base"));
        let mut bytes = pack_bytes(
            "corrupt",
            &base_sha,
            "linear.weight",
            1.0,
            &[1.0, 1.0, 1.0],
            &[1.0, 1.0],
            2,
            3,
            1,
        );
        *bytes.last_mut().unwrap() ^= 1;
        std::fs::write(&pack_path, bytes).unwrap();
        let error = parse_pack(&pack_path).unwrap_err();
        assert!(error.contains("B SHA-256 mismatch"), "{}", error);
        std::fs::remove_file(base_path).ok();
        std::fs::remove_file(pack_path).ok();
    }

    #[test]
    fn model_load_folds_into_private_mapping_without_changing_the_file() {
        let base_path = temp("model.bin");
        let pack_path = temp("model.mkap");
        crate::model::testbin::write(&base_path);
        let before = std::fs::read(&base_path).unwrap();
        let base_sha = crate::sha256::hex(&crate::sha256::digest(&before));
        let cfg = crate::model::testbin::config();
        let out = cfg.kda_proj();
        let input = cfg.d;
        let a = vec![0.25f32; input];
        let b: Vec<f32> = (0..out).map(|index| (index as f32 + 1.0) * 0.01).collect();
        std::fs::write(
            &pack_path,
            pack_bytes(
                "integration",
                &base_sha,
                "layers.0.self_attn.q_proj.weight",
                1.0,
                &a,
                &b,
                out,
                input,
                1,
            ),
        )
        .unwrap();

        let mut base = crate::model::Model::load(&base_path);
        let mut adapted =
            crate::model::Model::load_with_adapters(&base_path, std::slice::from_ref(&pack_path));
        assert!(adapted.has_adapter_packs());
        let base_logits = base.forward(7, 0);
        let adapted_logits = adapted.forward(7, 0);
        let moved: f32 = base_logits
            .iter()
            .zip(&adapted_logits)
            .map(|(left, right)| (left - right).abs())
            .sum();
        assert!(moved > 1e-6, "adapter did not change model output");
        assert_eq!(std::fs::read(&base_path).unwrap(), before);
        std::fs::remove_file(base_path).ok();
        std::fs::remove_file(pack_path).ok();
    }
}
