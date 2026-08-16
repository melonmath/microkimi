// imatrix - activation importance statistics for quantization calibration.
//
// The quality of a low-bit quantization depends on which inputs matter: a
// weight column that multiplies large activations deserves more fidelity.
// `microkimi calibrate` runs a small corpus through the model and accumulates
// the second moment (x^2) of the inputs of the quantized MoE expert matrices,
// per input column:
//   hidden[layer][c] : sum over tokens of h[c]^2, h = routed expert input
//                      (w1/w3 input, dim routed_hidden)
//   inter[layer][c]  : sum over selected experts/tokens of act[c]^2, act =
//                      SiTU(w1 h, w3 h) (w2 input, dim moe_inter)
// `microkimi slice --cold-vq N --imatrix imatrix.bin` then uses these
// statistics to weight the VQ codebook training and the nearest-centroid
// assignment (see quant.rs). The hooks in model.rs are no-ops (one atomic
// load) unless a calibration is running, so normal inference is bit-exact.
//
// File format (integers little-endian, sums little-endian f64):
//   magic       : 8 bytes "IMAT0001"
//   u32 n_layers, u32 routed_hidden, u32 moe_inter, u64 tokens
//   u32 n_moe_layers, then per MoE layer (ascending):
//       u32 layer_idx, routed_hidden f64 (hidden sums), moe_inter f64 (inter sums)

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

pub const MAGIC: &[u8; 8] = b"IMAT0001";

/// Accumulated second moments for one model (see the file format above).
pub struct Imatrix {
    pub n_layers: usize,
    pub routed_hidden: usize,
    pub moe_inter: usize,
    pub tokens: u64,
    pub layers: Vec<u32>,      // MoE layer indices, ascending
    pub hidden: Vec<Vec<f64>>, // per MoE layer: sum of h^2 per input column
    pub inter: Vec<Vec<f64>>,  // per MoE layer: sum of act^2 per input column
}

impl Imatrix {
    fn slot(&self, layer: usize) -> Option<usize> {
        self.layers.iter().position(|&l| l as usize == layer)
    }

    /// Column weights for one expert matrix of `layer` ("w1"/"w3" -> hidden,
    /// "w2" -> inter), normalized to mean 1 so the weighted error stays on
    /// the same scale as the unweighted one.
    pub fn col_weights(&self, layer: usize, wn: &str) -> Option<Vec<f32>> {
        let i = self.slot(layer)?;
        let src = if wn == "w2" { &self.inter[i] } else { &self.hidden[i] };
        let mean = src.iter().sum::<f64>() / src.len() as f64;
        if mean == 0.0 {
            return None;
        }
        Some(src.iter().map(|&v| (v / mean) as f32).collect())
    }
}

static ACTIVE: AtomicBool = AtomicBool::new(false);
static ACC: Mutex<Option<Imatrix>> = Mutex::new(None);

/// Arms the accumulator (calibration start). `moe_layers` ascending.
pub fn start(cfg: &crate::config::Config, moe_layers: Vec<u32>) {
    start_dims(cfg.n_layers, cfg.routed_hidden, cfg.moe_inter, moe_layers);
}

/// Dimension-explicit accumulator start. The K3 MoE path stores the routed
/// expert input width and expert intermediate width; the Qwen dense path
/// stores the hidden size (gate/up input) and the dense MLP width (down
/// input) in the same two slots.
pub fn start_dims(n_layers: usize, hidden_w: usize, inter_w: usize, layers: Vec<u32>) {
    let n = layers.len();
    *ACC.lock().unwrap() = Some(Imatrix {
        n_layers,
        routed_hidden: hidden_w,
        moe_inter: inter_w,
        tokens: 0,
        layers,
        hidden: vec![vec![0.0; hidden_w]; n],
        inter: vec![vec![0.0; inter_w]; n],
    });
    ACTIVE.store(true, Ordering::Relaxed);
}

/// model.rs hook: routed expert input h of one MoE layer (one call per token
/// per MoE layer). No-op unless a calibration is running.
pub fn record_hidden(layer: usize, h: &[f32]) {
    if !ACTIVE.load(Ordering::Relaxed) {
        return;
    }
    if let Some(im) = ACC.lock().unwrap().as_mut() {
        if let Some(i) = im.slot(layer) {
            for (a, &v) in im.hidden[i].iter_mut().zip(h) {
                *a += (v as f64) * (v as f64);
            }
        }
    }
}

/// model.rs hook: SiTU expert activation (w2 input), one call per selected
/// expert per token. No-op unless a calibration is running.
pub fn record_inter(layer: usize, act: &[f32]) {
    if !ACTIVE.load(Ordering::Relaxed) {
        return;
    }
    if let Some(im) = ACC.lock().unwrap().as_mut() {
        if let Some(i) = im.slot(layer) {
            for (a, &v) in im.inter[i].iter_mut().zip(act) {
                *a += (v as f64) * (v as f64);
            }
        }
    }
}

/// Counts one processed token (called by the calibration loop).
pub fn tick() {
    if let Some(im) = ACC.lock().unwrap().as_mut() {
        im.tokens += 1;
    }
}

/// Disarms the accumulator and writes the statistics file.
pub fn stop_and_save(path: &str) -> Result<(), String> {
    ACTIVE.store(false, Ordering::Relaxed);
    let im = ACC.lock().unwrap().take().ok_or("no calibration running")?;
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    for v in [im.n_layers as u32, im.routed_hidden as u32, im.moe_inter as u32] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&im.tokens.to_le_bytes());
    out.extend_from_slice(&(im.layers.len() as u32).to_le_bytes());
    for (i, &l) in im.layers.iter().enumerate() {
        out.extend_from_slice(&l.to_le_bytes());
        for &v in im.hidden[i].iter().chain(im.inter[i].iter()) {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(path, &out).map_err(|e| format!("cannot write {}: {}", path, e))?;
    Ok(())
}

/// Loads an imatrix file, checking the magic and the internal lengths.
pub fn load(path: &str) -> Result<Imatrix, String> {
    let b = std::fs::read(path).map_err(|e| format!("cannot read {}: {}", path, e))?;
    let mut r = Reader { b: &b, p: 0 };
    if r.take(8)? != MAGIC {
        return Err(format!("{}: not an imatrix file (bad magic)", path));
    }
    let n_layers = r.u32()? as usize;
    let routed_hidden = r.u32()? as usize;
    let moe_inter = r.u32()? as usize;
    let tokens = r.u64()?;
    let n_moe = r.u32()? as usize;
    let mut layers = Vec::with_capacity(n_moe);
    let mut hidden = Vec::with_capacity(n_moe);
    let mut inter = Vec::with_capacity(n_moe);
    for _ in 0..n_moe {
        layers.push(r.u32()?);
        hidden.push(r.f64s(routed_hidden)?);
        inter.push(r.f64s(moe_inter)?);
    }
    Ok(Imatrix { n_layers, routed_hidden, moe_inter, tokens, layers, hidden, inter })
}

/// `microkimi calibrate --model X.bin --text corpus.txt --out imatrix.bin
/// [--vocab V.json] [--max-tokens N]`: raw-encodes the corpus, forwards it
/// token by token (the model.rs hooks accumulate the expert input moments),
/// then writes the imatrix file.
pub fn calibrate_cmd(args: &[String]) {
    let tl = std::time::Instant::now();
    let Some(text_path) = crate::value_flag(args, "--text") else {
        eprintln!("error: calibrate requires --text corpus.txt");
        std::process::exit(1);
    };
    let Some(out) = crate::value_flag(args, "--out") else {
        eprintln!("error: calibrate requires --out imatrix.bin");
        std::process::exit(1);
    };
    let text = match std::fs::read_to_string(&text_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: cannot read {}: {}", text_path, e);
            std::process::exit(1);
        }
    };
    let mp = crate::model_flag(args).unwrap_or_else(crate::bin_path);
    if crate::quant::weights::read_config(&mp).ds.is_some() {
        eprintln!("error: calibrate is only supported for K3 and Qwen models (not DeepSeek-V4)");
        std::process::exit(1);
    }
    if let Some(qcfg) = crate::quant::weights::read_config(&mp).qwen.clone() {
        return calibrate_qwen(args, &mp, &qcfg, &text_path, &text, &out, tl);
    }
    let tok = crate::load_any_tokenizer(&mp, crate::vocab_flag(args), crate::quant::weights::read_config(&mp).vocab);
    let mut model = crate::model::Model::load(&mp);
    crate::check_tok_compat(&tok, &model);
    let cfg_n_layers = model.cfg.n_layers;
    let moe_layers: Vec<u32> = (0..cfg_n_layers).filter(|&l| model.cfg.is_moe(l)).map(|l| l as u32).collect();
    // The BPE encoder is slow on large inputs, so cap the text BEFORE
    // encoding when --max-tokens is given (6 bytes/token is a safe
    // overestimate for English; ids are truncated to the exact count below).
    let text = match crate::value_flag(args, "--max-tokens").and_then(|s| s.parse::<usize>().ok()) {
        Some(n) if text.len() > n * 6 => {
            let mut cap = n * 6;
            while !text.is_char_boundary(cap) {
                cap -= 1;
            }
            text[..cap].to_string()
        }
        _ => text,
    };
    let mut ids = tok.encode_raw(&text);
    if let Some(n) = crate::value_flag(args, "--max-tokens").and_then(|s| s.parse::<usize>().ok()) {
        ids.truncate(n);
    }
    println!(
        "calibrate: {} ({} bytes -> {} tokens), {} MoE layers x ({} hidden + {} inter)",
        text_path,
        text.len(),
        ids.len(),
        moe_layers.len(),
        model.cfg.routed_hidden,
        model.cfg.moe_inter
    );
    start(&model.cfg, moe_layers);
    model.reset_cache();
    let tp = std::time::Instant::now();
    // --chunk N (default 512): restart the context every N tokens. Decode
    // attention costs O(position) per token, so one long contiguous context
    // makes calibration quadratic; the activation statistics do not need a
    // contiguous context (same chunking convention as llama.cpp imatrix).
    let chunk: usize = crate::value_flag(args, "--chunk").and_then(|s| s.parse().ok()).unwrap_or(512);
    let mut pos = 0usize;
    for (done, &id) in ids.iter().enumerate() {
        if pos == chunk {
            model.reset_cache();
            pos = 0;
        }
        model.forward(id, pos);
        pos += 1;
        tick();
        if (done + 1) % 2000 == 0 {
            println!("  {} tokens ({:.1?})", done + 1, tp.elapsed());
        }
    }
    if let Err(e) = stop_and_save(&out) {
        eprintln!("error: {}", e);
        std::process::exit(1);
    }
    let size = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
    println!(
        "calibrate: {} tokens in {:.1?} (load {:.1?}) - imatrix saved to {} ({:.1} KB)",
        ids.len(),
        tp.elapsed(),
        tl.elapsed(),
        out,
        size as f64 / 1024.0
    );
}

/// Qwen dense calibration: raw-encodes the corpus with the Qwen tokenizer
/// and forwards it token by token; the packed_dense_mlp hooks accumulate
/// the per-layer input moments of the MLP matrices (gate/up input = hidden
/// stream, down input = SiLU-gated activation). MoE Qwen decoders are
/// rejected: their per-expert activations are not accumulated.
fn calibrate_qwen(
    args: &[String],
    mp: &str,
    qcfg: &crate::config::QwenConfig,
    text_path: &str,
    text: &str,
    out: &str,
    tl: std::time::Instant,
) {
    if !qcfg.is_dense() {
        eprintln!("error: calibrate supports the dense Qwen variant only (routed experts are not accumulated)");
        std::process::exit(1);
    }
    let tok = crate::load_qwen_any_tokenizer(mp, crate::vocab_flag(args), qcfg.vocab);
    let mut model = crate::model::qwen::QwenModel::load(mp);
    let text_owned = match crate::value_flag(args, "--max-tokens").and_then(|s| s.parse::<usize>().ok()) {
        Some(n) if text.len() > n * 6 => {
            let mut cap = n * 6;
            while !text.is_char_boundary(cap) {
                cap -= 1;
            }
            text[..cap].to_string()
        }
        _ => text.to_string(),
    };
    let mut ids = tok.encode_raw(&text_owned);
    if let Some(n) = crate::value_flag(args, "--max-tokens").and_then(|s| s.parse::<usize>().ok()) {
        ids.truncate(n);
    }
    println!(
        "calibrate: {} ({} bytes -> {} tokens), {} dense layers x ({} hidden + {} inter)",
        text_path,
        text_owned.len(),
        ids.len(),
        qcfg.n_layers,
        qcfg.d,
        qcfg.dense_inter
    );
    start_dims(
        qcfg.n_layers,
        qcfg.d,
        qcfg.dense_inter,
        (0..qcfg.n_layers as u32).collect(),
    );
    model.reset();
    let tp = std::time::Instant::now();
    let chunk: usize = crate::value_flag(args, "--chunk").and_then(|s| s.parse().ok()).unwrap_or(512);
    for (done, &id) in ids.iter().enumerate() {
        if done > 0 && done % chunk == 0 {
            model.reset();
        }
        model.forward(id);
        tick();
        if (done + 1) % 2000 == 0 {
            println!("  {} tokens ({:.1?})", done + 1, tp.elapsed());
        }
    }
    if let Err(e) = stop_and_save(out) {
        eprintln!("error: {}", e);
        std::process::exit(1);
    }
    let size = std::fs::metadata(out).map(|m| m.len()).unwrap_or(0);
    println!(
        "calibrate: {} tokens in {:.1?} (load {:.1?}) - imatrix saved to {} ({:.1} KB)",
        ids.len(),
        tp.elapsed(),
        tl.elapsed(),
        out,
        size as f64 / 1024.0
    );
}

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.p + n > self.b.len() {
            return Err("truncated imatrix file".to_string());
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn f64s(&mut self, n: usize) -> Result<Vec<f64>, String> {
        let raw = self.take(n * 8)?;
        Ok(raw.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect())
    }
}
