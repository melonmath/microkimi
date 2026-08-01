// microkimi .bin format (microkimi-debug.bin, nanokimi-0.2b.bin, ...):
//   magic "MKIM0001" (8 bytes)
//   u32 n_tensors
//   directory × n : u16 name_len | name (utf-8) | u8 dtype (0=f32, 1=mxfp4, 2=i32) |
//                   u8 n_dims | u32 dims[n_dims] | u64 offset | u64 size
//   raw data (offsets from start of file, 64-aligned)
// mxfp4 dtype: blob = packed (R×C/2 bytes) then scales (R×C/32), dims = logical [R,C].
// MKIM0002: u32 config_len + JSON config right after the magic.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, Write};

pub const MAGIC: &[u8; 8] = b"MKIM0001";
pub const MAGIC_V2: &[u8; 8] = b"MKIM0002";
pub const DTYPE_F32: u8 = 0;
pub const DTYPE_MXFP4: u8 = 1;
pub const DTYPE_I32: u8 = 2;

#[derive(Debug, Clone)]
pub struct Entry {
    pub dtype: u8,
    pub dims: Vec<u32>,
    pub offset: u64,
    pub size: u64,
}

/// Size in bytes of a data blob.
pub fn blob_size(dtype: u8, dims: &[u32]) -> u64 {
    let n: u64 = dims.iter().map(|&d| d as u64).product();
    match dtype {
        DTYPE_F32 => n * 4,
        DTYPE_I32 => n * 4,
        DTYPE_MXFP4 => {
            let (r, c) = (dims[0] as u64, dims[1] as u64);
            r * c / 2 + r * c / 32
        }
        _ => panic!("unknown dtype {}", dtype),
    }
}

fn dir_entry_size(name: &str, dims: &[u32]) -> u64 {
    2 + name.len() as u64 + 1 + 1 + 4 * dims.len() as u64 + 8 + 8
}

// ── writing (build) ──

pub struct BinWriter {
    pub names_order: Vec<(String, u8, Vec<u32>)>, // (name, dtype, dims) in write order
}

impl BinWriter {
    pub fn new() -> Self {
        BinWriter { names_order: Vec::new() }
    }

    pub fn add(&mut self, name: &str, dtype: u8, dims: Vec<u32>) {
        self.names_order.push((name.to_string(), dtype, dims));
    }

    /// Computes the 64-aligned offsets in write order; `prefix` = number of
    /// bytes before the u32 tensor count (8 for MKIM0001, 8+4+config_len for
    /// MKIM0002).
    fn layout(&self, prefix: u64) -> Vec<u64> {
        let dir_size: u64 = self
            .names_order
            .iter()
            .map(|(n, _, d)| dir_entry_size(n, d))
            .sum();
        let data_start = prefix + 4 + dir_size;
        let mut offsets = Vec::with_capacity(self.names_order.len());
        let mut pos = data_start;
        for (_, dtype, dims) in &self.names_order {
            pos = pos.div_ceil(64) * 64;
            offsets.push(pos);
            pos += blob_size(*dtype, dims);
        }
        offsets
    }

    fn write_directory(&self, f: &mut std::fs::File, offsets: &[u64]) {
        f.write_all(&(self.names_order.len() as u32).to_le_bytes()).unwrap();
        for ((name, dtype, dims), &off) in self.names_order.iter().zip(offsets) {
            f.write_all(&(name.len() as u16).to_le_bytes()).unwrap();
            f.write_all(name.as_bytes()).unwrap();
            f.write_all(&[*dtype]).unwrap();
            f.write_all(&(dims.len() as u8).to_le_bytes()).unwrap();
            for d in dims {
                f.write_all(&d.to_le_bytes()).unwrap();
            }
            f.write_all(&off.to_le_bytes()).unwrap();
            f.write_all(&blob_size(*dtype, dims).to_le_bytes()).unwrap();
        }
    }

    /// Computes the directory (64-aligned offsets) and writes it with the header.
    /// Returns the offsets in the same order as names_order.
    pub fn write_header(&self, f: &mut std::fs::File) -> Vec<u64> {
        let offsets = self.layout(8);
        f.write_all(MAGIC).unwrap();
        self.write_directory(f, &offsets);
        offsets
    }

    /// MKIM0002 header: magic + u32 config_len + JSON config + directory.
    pub fn write_header_v2(&self, f: &mut std::fs::File, config_json: &str) -> Vec<u64> {
        let offsets = self.layout(8 + 4 + config_json.len() as u64);
        f.write_all(MAGIC_V2).unwrap();
        f.write_all(&(config_json.len() as u32).to_le_bytes()).unwrap();
        f.write_all(config_json.as_bytes()).unwrap();
        self.write_directory(f, &offsets);
        offsets
    }

    /// Writes a blob at its offset (with alignment padding before).
    pub fn write_blob_at(&self, f: &mut std::fs::File, offset: u64, data: &[u8]) {
        let cur = f.stream_position().unwrap();
        if cur < offset {
            f.write_all(&vec![0u8; (offset - cur) as usize]).unwrap();
        } else if cur > offset {
            f.seek(SeekFrom::Start(offset)).unwrap();
        }
        f.write_all(data).unwrap();
    }
}

// ── reading (inference) ──

pub struct BinFile {
    pub data: Vec<u8>,
    pub entries: HashMap<String, Entry>,
    pub config: crate::config::Config,
}

/// Reads ONLY the header (magic + config block) without loading the whole
/// file — used to pick the right tokenizer before loading the model.
pub fn read_config(path: &str) -> crate::config::Config {
    let mut f = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("{} unreadable: {} (run `microkimi build` first)", path, e));
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic).unwrap();
    if magic == *MAGIC {
        return crate::config::Config::microkimi();
    }
    assert_eq!(magic, *MAGIC_V2, "bad magic in {} (expected MKIM0001 or MKIM0002)", path);
    let mut clen = [0u8; 4];
    f.read_exact(&mut clen).unwrap();
    let clen = u32::from_le_bytes(clen) as usize;
    let mut cbuf = vec![0u8; clen];
    f.read_exact(&mut cbuf).unwrap();
    crate::config::Config::from_json(&crate::json::parse(&cbuf))
}

impl BinFile {
    pub fn open(path: &str) -> Self {
        let mut f = std::fs::File::open(path)
            .unwrap_or_else(|e| panic!("{} unreadable: {} (run `microkimi build` first)", path, e));
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic).unwrap();
        let config = if magic == *MAGIC {
            crate::config::Config::microkimi() // MKIM0001: implicit microkimi config
        } else if magic == *MAGIC_V2 {
            // MKIM0002: u32 config_len + explicit JSON config
            let mut clen = [0u8; 4];
            f.read_exact(&mut clen).unwrap();
            let clen = u32::from_le_bytes(clen) as usize;
            let mut cbuf = vec![0u8; clen];
            f.read_exact(&mut cbuf).unwrap();
            crate::config::Config::from_json(&crate::json::parse(&cbuf))
        } else {
            panic!("bad magic in {} (expected MKIM0001 or MKIM0002)", path)
        };
        let mut nbuf = [0u8; 4];
        f.read_exact(&mut nbuf).unwrap();
        let n = u32::from_le_bytes(nbuf) as usize;
        let mut entries = HashMap::with_capacity(n);
        for _ in 0..n {
            let mut nlen = [0u8; 2];
            f.read_exact(&mut nlen).unwrap();
            let nlen = u16::from_le_bytes(nlen) as usize;
            let mut name = vec![0u8; nlen];
            f.read_exact(&mut name).unwrap();
            let mut fixed = [0u8; 2];
            f.read_exact(&mut fixed).unwrap();
            let (dtype, n_dims) = (fixed[0], fixed[1] as usize);
            let mut dims = vec![0u32; n_dims];
            for d in dims.iter_mut() {
                let mut b = [0u8; 4];
                f.read_exact(&mut b).unwrap();
                *d = u32::from_le_bytes(b);
            }
            let mut b16 = [0u8; 16];
            f.read_exact(&mut b16).unwrap();
            let offset = u64::from_le_bytes(b16[0..8].try_into().unwrap());
            let size = u64::from_le_bytes(b16[8..16].try_into().unwrap());
            entries.insert(String::from_utf8(name).unwrap(), Entry { dtype, dims, offset, size });
        }
        drop(f);
        let data = std::fs::read(path).unwrap();
        BinFile { data, entries, config }
    }

    fn blob(&self, name: &str) -> &[u8] {
        let e = self
            .entries
            .get(name)
            .unwrap_or_else(|| panic!("missing tensor: {}", name));
        &self.data[e.offset as usize..(e.offset + e.size) as usize]
    }

    /// f32 tensor → slice over the file bytes (lazy conversion elsewhere).
    #[allow(dead_code)]
    pub fn f32_bytes(&self, name: &str) -> &[u8] {
        let e = self.entries.get(name).unwrap_or_else(|| panic!("missing tensor: {}", name));
        assert_eq!(e.dtype, DTYPE_F32, "{} is not f32", name);
        self.blob(name)
    }

    /// f32 tensor → Vec<f32> (copy).
    #[allow(dead_code)]
    pub fn f32_vec(&self, name: &str) -> Vec<f32> {
        self.f32_bytes(name)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// mxfp4 tensor → (packed, scales, rows, cols).
    #[allow(dead_code)]
    pub fn mxfp4_parts(&self, name: &str) -> (&[u8], &[u8], usize, usize) {
        let e = self.entries.get(name).unwrap_or_else(|| panic!("missing tensor: {}", name));
        assert_eq!(e.dtype, DTYPE_MXFP4, "{} is not mxfp4", name);
        let (r, c) = (e.dims[0] as usize, e.dims[1] as usize);
        let b = self.blob(name);
        let np = r * c / 2;
        (&b[..np], &b[np..], r, c)
    }

    /// i32 tensor → Vec<i32> (copy).
    #[allow(dead_code)]
    pub fn i32_vec(&self, name: &str) -> Vec<i32> {
        let e = self.entries.get(name).unwrap_or_else(|| panic!("missing tensor: {}", name));
        assert_eq!(e.dtype, DTYPE_I32, "{} is not i32", name);
        self.blob(name)
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

pub fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}
