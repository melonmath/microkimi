// safetensors reading (JSON header + data ranges), for local files
// (a model repo in the local HF cache) and remote shard headers (K3,
// via range requests). Read-only: tensors are located and sliced out,
// never converted here (conversion lives in quant/weights.rs).

use crate::json::{self, Json};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub offsets: (u64, u64), // relative to the start of the data section
}

/// Parses a safetensors header contained in `bytes` (which must start with
/// the 8 header_len bytes). Returns (header_len, name → info map).
pub fn parse_header(bytes: &[u8]) -> (u64, HashMap<String, TensorInfo>) {
    assert!(bytes.len() >= 8, "safetensors: file too small");
    let hlen = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    let header = json::parse(&bytes[8..8 + hlen as usize]);
    let mut map = HashMap::new();
    if let Json::Obj(pairs) = header {
        for (name, meta) in pairs {
            if name == "__metadata__" {
                continue;
            }
            let dtype = meta
                .get("dtype")
                .and_then(|d| d.as_str())
                .unwrap_or("?")
                .to_string();
            let shape: Vec<usize> = meta
                .get("shape")
                .and_then(|s| s.as_arr())
                .map(|a| a.iter().filter_map(|x| x.as_num().map(|n| n as usize)).collect())
                .unwrap_or_default();
            let offs: Vec<u64> = meta
                .get("data_offsets")
                .and_then(|s| s.as_arr())
                .map(|a| a.iter().filter_map(|x| x.as_num().map(|n| n as u64)).collect())
                .unwrap_or_default();
            map.insert(
                name,
                TensorInfo {
                    dtype,
                    shape,
                    offsets: (offs[0], offs[1]),
                },
            );
        }
    }
    (hlen, map)
}

/// bf16 → f32 : f32::from_bits(bits << 16)
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Converts a bf16 buffer (little-endian) into a Vec<f32>.
pub fn bf16_slice_to_f32(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(2)
        .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}
