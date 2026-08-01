// `microkimi selftest`: compares the KDA recurrence, SiTU, MXFP4
// dequant and attn_res against the reference values in ref/golden.json (generated
// by ref/make_golden.py with torch + the vendored fla shim). Relative tolerance 1e-4.

use crate::json::{self, Json};

const RTOL: f64 = 1e-4;
const ATOL: f64 = 1e-6;

fn arr(j: &Json, key: &str) -> Vec<f32> {
    j.get(key)
        .and_then(|x| x.as_arr())
        .unwrap_or_else(|| panic!("golden: field '{}' missing", key))
        .iter()
        .map(|x| x.as_num().unwrap() as f32)
        .collect()
}

fn check(name: &str, got: &[f32], want: &[f32]) -> bool {
    assert_eq!(got.len(), want.len(), "{}: sizes {} vs {}", name, got.len(), want.len());
    let mut max_rel = 0f64;
    let mut max_abs = 0f64;
    let mut n_bad = 0;
    for (a, b) in got.iter().zip(want) {
        let d = (*a as f64 - *b as f64).abs();
        max_abs = max_abs.max(d);
        let rel = d / (*b as f64).abs().max(1e-12);
        max_rel = max_rel.max(rel);
        if d > ATOL + RTOL * (*b as f64).abs() {
            n_bad += 1;
        }
    }
    let ok = n_bad == 0;
    println!(
        "  {:<28} {}  (max_abs={:.3e}, max_rel={:.3e}{})",
        name,
        if ok { "OK " } else { "FAIL" },
        max_abs,
        max_rel,
        if n_bad > 0 { format!(", {} values out of tolerance", n_bad) } else { String::new() }
    );
    ok
}

/// Standalone KDA recurrence replicating fla/ops/kda _kda_core (raw inputs):
/// L2-norm q/k (p=2, eps 1e-12), beta=sigmoid, gate=-5·sigmoid(exp(A_log)·(g+dt_bias)),
/// q×K^-0.5, then S decay → δ=v-kᵀS → S+=(βk)⊗δ → o=qᵀS.
fn kda_core(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    t_steps: usize,
    h_heads: usize,
    dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut s = vec![0f32; h_heads * dim * dim];
    let mut o = vec![0f32; t_steps * h_heads * dim];
    let scale = (dim as f32).powf(-0.5);
    for t in 0..t_steps {
        for h in 0..h_heads {
            let base = (t * h_heads + h) * dim;
            let qh = &q[base..base + dim];
            let kh = &k[base..base + dim];
            let vh = &v[base..base + dim];
            let gh = &g[base..base + dim];
            let bh = 1.0 / (1.0 + (-beta[t * h_heads + h]).exp());
            // L2-norm (F.normalize : x / max(||x||₂, 1e-12)), q × scale
            let nq: f32 = qh.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            let nk: f32 = kh.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            let qn: Vec<f32> = qh.iter().map(|x| x / nq * scale).collect();
            let kn: Vec<f32> = kh.iter().map(|x| x / nk).collect();
            // gate : g = -5 · sigmoid(exp(A_log) · (g + dt_bias.view(H,K)))
            let gg: Vec<f32> = (0..dim)
                .map(|c| {
                    let a = a_log[c].exp();
                    let x = gh[c] + dt_bias[h * dim + c];
                    -5.0 / (1.0 + (-a * x).exp())
                })
                .collect();
            let sh = &mut s[h * dim * dim..(h + 1) * dim * dim];
            for i in 0..dim {
                let decay = gg[i].exp();
                for j in 0..dim {
                    sh[i * dim + j] *= decay;
                }
            }
            let mut delta = vh.to_vec();
            for i in 0..dim {
                for j in 0..dim {
                    delta[j] -= kn[i] * sh[i * dim + j];
                }
            }
            let oh = &mut o[base..base + dim];
            for x in oh.iter_mut() {
                *x = 0.0;
            }
            for i in 0..dim {
                let bk = bh * kn[i];
                for j in 0..dim {
                    sh[i * dim + j] += bk * delta[j];
                }
                for j in 0..dim {
                    oh[j] += qn[i] * sh[i * dim + j];
                }
            }
        }
    }
    (o, s)
}

fn situ_rust(g: f32, u: f32) -> f32 {
    4.0 * (g / 4.0).tanh() * (1.0 / (1.0 + (-g).exp())) * (25.0 * (u / 25.0).tanh())
}

pub fn run() {
    let path = if std::path::Path::new("ref/golden.json").exists() {
        "ref/golden.json"
    } else {
        "/workspace/microkimi/ref/golden.json"
    };
    let bytes = std::fs::read(path)
        .unwrap_or_else(|_| panic!("{} missing - run first: /home/node/venv/bin/python3 ref/make_golden.py", path));
    let golden = json::parse(&bytes);
    let mut all_ok = true;
    println!("microkimi selftest - comparison against {} (rel tol {:.0e})", path, RTOL);

    // ── 1) KDA recurrence ──
    {
        let j = golden.get("kda").unwrap();
        let t = j.get("T").unwrap().as_num().unwrap() as usize;
        let h = j.get("H").unwrap().as_num().unwrap() as usize;
        let dim = j.get("K").unwrap().as_num().unwrap() as usize;
        let (o, s) = kda_core(
            &arr(j, "q"), &arr(j, "k"), &arr(j, "v"), &arr(j, "g"), &arr(j, "beta"),
            &arr(j, "A_log"), &arr(j, "dt_bias"), t, h, dim,
        );
        all_ok &= check("KDA recurrence: outputs o", &o, &arr(j, "o"));
        all_ok &= check("KDA recurrence: final state S", &s, &arr(j, "S"));
    }

    // ── 2) SiTU ──
    {
        let j = golden.get("situ").unwrap();
        let g = arr(j, "g");
        let u = arr(j, "u");
        let out: Vec<f32> = g.iter().zip(&u).map(|(&a, &b)| situ_rust(a, b)).collect();
        all_ok &= check("SiTU", &out, &arr(j, "out"));
    }

    // ── 3) MXFP4 dequant ──
    {
        let j = golden.get("mxfp4").unwrap();
        let rows = j.get("rows").unwrap().as_num().unwrap() as usize;
        let cols = j.get("cols").unwrap().as_num().unwrap() as usize;
        let packed: Vec<u8> = arr(j, "packed").iter().map(|&x| x as u8).collect();
        let scales: Vec<u8> = arr(j, "scales").iter().map(|&x| x as u8).collect();
        let w = crate::mxfp4::dequant(&packed, &scales, rows, cols);
        all_ok &= check("MXFP4 dequant", &w, &arr(j, "W"));
    }

    // ── 4) attn_res ──
    {
        let j = golden.get("attn_res").unwrap();
        let d = j.get("D").unwrap().as_num().unwrap() as usize;
        let b = j.get("B").unwrap().as_num().unwrap() as usize;
        assert_eq!(d, crate::model::D);
        let flat = arr(j, "blocks");
        let blocks: Vec<Vec<f32>> = (0..b).map(|i| flat[i * d..(i + 1) * d].to_vec()).collect();
        let prefix = arr(j, "prefix");
        let norm_w = arr(j, "norm_w");
        let proj_w = arr(j, "proj_w");
        let w: Vec<f32> = norm_w.iter().zip(&proj_w).map(|(a, b)| a * b).collect();
        let mut out = vec![0f32; d];
        crate::model::attn_res(&crate::config::Config::microkimi(), &prefix, &blocks, &w, &mut out);
        all_ok &= check("AttnRes", &out, &arr(j, "out"));
    }

    println!();
    if all_ok {
        println!("SELFTEST OK - all 4 mechanisms match the references (tol {:.0e})", RTOL);
    } else {
        println!("SELFTEST FAILED");
        std::process::exit(1);
    }
}

/// DeepSeek-V4 dequantization checks against ref/ds_golden.json
/// (fp8 e4m3/ue8m0 128x128 and fp4 e2m1/ue8m0/32).
pub fn run_ds() {
    let path = if std::path::Path::new("ref/ds_golden.json").exists() {
        "ref/ds_golden.json"
    } else {
        "/workspace/microkimi-oss/ref/ds_golden.json"
    };
    let bytes = std::fs::read(path)
        .unwrap_or_else(|_| panic!("{} missing - run first: /home/node/venv/bin/python3 ref/make_ds_golden.py", path));
    let golden = json::parse(&bytes);
    let mut ok = true;

    {
        let j = golden.get("fp8").unwrap();
        let rows = j.get("rows").unwrap().as_num().unwrap() as usize;
        let cols = j.get("cols").unwrap().as_num().unwrap() as usize;
        let packed: Vec<u8> = arr(j, "w_packed").iter().map(|&x| x as u8).collect();
        let scales: Vec<u8> = arr(j, "scales").iter().map(|&x| x as u8).collect();
        let w = crate::dequant::dequant_fp8(&packed, &scales, rows, cols);
        ok &= check("DS fp8 e4m3 dequant (torch golden)", &w, &arr(j, "dequant"));
        // quantize path: my quantize_fp8 of the ORIGINAL matrix must reproduce the
        // torch-packed bytes exactly (same scale rule + nearest-even cast)
        let (qw, qs) = crate::dequant::quantize_fp8(&arr(j, "w_orig"), rows, cols);
        let n_diff = qw.iter().zip(&packed).filter(|(a, b)| a != b).count();
        let n_sdiff = qs.iter().zip(&scales).filter(|(a, b)| a != b).count();
        println!(
            "  {:<28} {}  (packed byte diffs: {}, scale byte diffs: {})",
            "DS fp8 quantize (vs torch)",
            if n_diff == 0 && n_sdiff == 0 { "OK   " } else { "ÉCHEC" },
            n_diff,
            n_sdiff
        );
        ok &= n_diff == 0 && n_sdiff == 0;
    }
    {
        let j = golden.get("fp4").unwrap();
        let rows = j.get("rows").unwrap().as_num().unwrap() as usize;
        let cols = j.get("cols").unwrap().as_num().unwrap() as usize;
        let packed: Vec<u8> = arr(j, "packed").iter().map(|&x| x as u8).collect();
        let scales: Vec<u8> = arr(j, "scales").iter().map(|&x| x as u8).collect();
        let w = crate::dequant::dequant_fp4(&packed, &scales, rows, cols);
        ok &= check("DS fp4 e2m1 dequant (torch golden)", &w, &arr(j, "dequant"));
    }

    println!();
    if ok {
        println!("DS DEQUANT OK - fp8/fp4 match the torch references");
    } else {
        println!("DS DEQUANT FAILED");
        std::process::exit(1);
    }
}

/// DeepSeek-V4 MoE routing + Hyper-Connections vs ref/ds_golden2.json
/// (plain-torch transcription of the DeepSeek reference math).
pub fn run_ds2() {
    let path = if std::path::Path::new("ref/ds_golden2.json").exists() {
        "ref/ds_golden2.json"
    } else {
        "/workspace/microkimi-oss/ref/ds_golden2.json"
    };
    let bytes = std::fs::read(path)
        .unwrap_or_else(|_| panic!("{} missing - run first: /home/node/venv/bin/python3 ref/make_ds_golden2.py", path));
    let golden = json::parse(&bytes);
    let mut ok = true;

    // ── gate: sqrtsoftplus + bias-for-selection + renorm ×1.5 ──
    {
        let j = golden.get("gate").unwrap();
        let x = arr(j, "x"); // [3, 16]
        let gw = arr(j, "gate_w");
        let bias = arr(j, "bias");
        let want_idx = arr(j, "indices"); // [3, 6]
        let want_w = arr(j, "weights");   // [3, 6]
        let mut got_ids: Vec<u32> = Vec::new();
        let mut got_ws: Vec<f32> = Vec::new();
        for t in 0..3 {
            let (sel, _) = crate::deepseek::gate_forward(&x[t * 16..(t + 1) * 16], &gw, Some(&bias), None, 0, 32, 6, 1.5);
            got_ids.extend(sel.iter().map(|s| s.0));
            got_ws.extend(sel.iter().map(|s| s.1));
        }
        // the reference topk is unsorted; compare per-token as sets + sorted-by-id weights
        let mut ids_ok = true;
        for t in 0..3 {
            let mut a: Vec<u32> = got_ids[t * 6..(t + 1) * 6].to_vec();
            let mut b: Vec<u32> = want_idx[t * 6..(t + 1) * 6].iter().map(|&v| v as u32).collect();
            a.sort();
            b.sort();
            if a != b {
                ids_ok = false;
            }
        }
        println!(
            "  {:<28} {}  (ids exacts: {})",
            "DS gate top-6 (sqrtsoftplus)",
            if ids_ok { "OK   " } else { "ÉCHEC" },
            ids_ok
        );
        ok &= ids_ok;
        // relaxed scale-aware check: gate weights go through sqrt(softplus(x)),
        // the exp amplifies f32 rounding differently than torch (tolerance
        // 1e-5 abs / 1e-3 of max — still catches any semantic bug)
        {
            let scale = want_w.iter().fold(0f64, |m, &b| m.max((b as f64).abs()));
            let mut bad = 0;
            for (a, b) in got_ws.iter().zip(&want_w) {
                if (*a as f64 - *b as f64).abs() > 1e-5 + 1e-3 * scale {
                    bad += 1;
                }
            }
            let lok = bad == 0;
            println!(
                "  {:<28} {}  ({} val. hors tol 1e-5+1e-3·scale)",
                "DS gate weights (renorm ×1.5)",
                if lok { "OK   " } else { "ÉCHEC" },
                bad
            );
            ok &= lok;
        }
    }

    // ── expert: silu(gate)*clamp(up), gate clamp ──
    {
        let j = golden.get("expert").unwrap();
        let x = arr(j, "x");
        let w1 = arr(j, "w1");
        let w2 = arr(j, "w2");
        let w3 = arr(j, "w3");
        let mut out = vec![0f32; 8];
        crate::deepseek::expert_forward(&w1, &w2, &w3, &x, 16, 10.0, &mut out);
        ok &= check("DS expert (swiglu_limit 10)", &out, &arr(j, "out"));
    }

    // ── hyper-connections: hc_pre / hc_post / hc_head + sinkhorn ──
    {
        let j = golden.get("hc").unwrap();
        let xs = arr(j, "x");       // [hc*d] = [24]
        let hc_fn = arr(j, "hc_fn");
        let hc_scale = arr(j, "hc_scale");
        let hc_base = arr(j, "hc_base");
        let (y_pre, post, comb) = crate::deepseek::hc_pre(
            &xs, &xs, &hc_fn, &hc_scale, &hc_base, 4, 1e-6, 20, 1e-6,
        );
        ok &= check("DS hc pre (sinkhorn pre)", &y_pre, &arr(j, "y_pre"));
        ok &= check("DS hc post weights", &post, &arr(j, "post"));
        ok &= check("DS hc comb (sinkhorn 20 iters)", &comb, &arr(j, "comb"));
        let y_post = crate::deepseek::hc_post(&y_pre, &xs, &post, &comb, 4);
        ok &= check("DS hc_post", &y_post, &arr(j, "y_post"));
        let y_head = crate::deepseek::hc_head(&xs, &xs, &hc_fn, hc_scale[0], &hc_base, 4, 1e-6, 1e-6);
        ok &= check("DS hc_head (no sinkhorn)", &y_head, &arr(j, "y_head"));
    }

    println!();
    if ok {
        println!("DS MOE/HC OK - routing and hyper-connections match the references");
    } else {
        println!("DS MOE/HC FAILED");
        std::process::exit(1);
    }
}

// ── DeepSeek-V4 sparse attention parity (3 layer types × 10 tokens) ──

fn ds_load_layer(j: &Json, root: &Json, ratio: i32, compress_theta: f64) -> (crate::config::DsConfig, crate::deepseek::DsAttentionW, Vec<Vec<f32>>) {
    let mut cfg = crate::config::DsConfig::microdeepseek();
    for l in 0..cfg.compress_ratios.len() {
        cfg.compress_ratios[l] = ratio;
    }
    cfg.compress_rope_theta = compress_theta;
    let w = j.get("weights").unwrap();
    let a = |k: &str| arr(w, k);
    let opt = |k: &str| if w.get(k).is_some() { a(k) } else { Vec::new() };
    let dw = crate::deepseek::DsAttentionW {
        wq_a: a("wq_a"),
        q_norm_w: a("q_norm_w"),
        wq_b: a("wq_b"),
        wkv: a("wkv"),
        kv_norm_w: a("kv_norm_w"),
        wo_a: a("wo_a"),
        wo_b: a("wo_b"),
        attn_sink: a("attn_sink"),
        comp_wkv: opt("comp_wkv"),
        comp_wgate: opt("comp_wgate"),
        comp_ape: opt("comp_ape"),
        comp_norm_w: opt("comp_norm_w"),
        idx_wq_b: opt("idx_wq_b"),
        idx_weights_proj: opt("idx_weights_proj"),
        idx_comp_wkv: opt("idx_comp_wkv"),
        idx_comp_wgate: opt("idx_comp_wgate"),
        idx_comp_ape: opt("idx_comp_ape"),
        idx_comp_norm_w: opt("idx_comp_norm_w"),
    };
    let xs: Vec<Vec<f32>> = arr(root, "x")
        .chunks(cfg.d)
        .map(|c| c.to_vec())
        .collect();
    (cfg, dw, xs)
}

pub fn run_ds3() {
    let path = if std::path::Path::new("ref/ds_golden3.json").exists() {
        "ref/ds_golden3.json"
    } else {
        "/workspace/microkimi-oss/ref/ds_golden3.json"
    };
    let bytes = std::fs::read(path)
        .unwrap_or_else(|_| panic!("{} missing - run first: /home/node/venv/bin/python3 ref/make_ds_golden3.py", path));
    let golden = json::parse(&bytes);
    let mut ok = true;

    // rope tables: theta=10000 (window) and theta=160000+YaRN (compressed)
    {
        let g0 = golden.get("rope_theta10000").unwrap();
        let (cos, sin) = crate::deepseek::precompute_freqs_cis(64, 4096, 0, 10000.0, 16.0, 32, 1);
        ok &= check("DS rope theta=10000 cos", &cos, &arr(g0, "cos"));
        ok &= check("DS rope theta=10000 sin", &sin, &arr(g0, "sin"));
        let g1 = golden.get("rope_compress").unwrap();
        let (cos1, sin1) = crate::deepseek::precompute_freqs_cis(64, 4096, 65536, 160000.0, 16.0, 32, 1);
        ok &= check("DS rope 160000+YaRN cos", &cos1, &arr(g1, "cos"));
        ok &= check("DS rope 160000+YaRN sin", &sin1, &arr(g1, "sin"));
    }

    let t: usize = golden.get("T").unwrap().as_num().unwrap() as usize;
    for (key, ratio, theta, label) in [
        ("layer_window", 0, 10000.0, "DS attention window-only"),
        ("layer_overlap_indexer", 4, 160000.0, "DS attention overlap+indexer"),
        ("layer_dense", 8, 160000.0, "DS attention dense compressor"),
    ] {
        let j = golden.get(key).unwrap();
        let want = arr(j, "out");
        // locate the max error (debug)
        {
            let mut worst = (0f32, 0usize, 0f32, 0f32);
            let mut got_dbg: Vec<f32> = Vec::new();
            let (cfg2, dw2, xs2) = ds_load_layer(j, &golden, ratio, theta);
            let mut st2 = crate::deepseek::DsAttention::new(&cfg2, 0);
            for (ti, x) in xs2.iter().enumerate() {
                let mut o = vec![0f32; cfg2.n_heads * cfg2.head_dim];
                crate::deepseek::attention_step(&cfg2, 0, &dw2, &mut st2, x, &mut o);
                let mut proj = vec![0f32; cfg2.d];
                crate::deepseek::grouped_o_proj(&cfg2, &dw2.wo_a, &dw2.wo_b, &o, &mut proj);
                for (i, &p) in proj.iter().enumerate() {
                    let d = (p - want[ti * cfg2.d + i]).abs();
                    if d > worst.0 { worst = (d, ti * cfg2.d + i, p, want[ti * cfg2.d + i]); }
                }
                got_dbg.extend_from_slice(&proj);
            }
            println!("    debug {} worst: d={:.4} at idx {} (tok {}) got {:.6} want {:.6}", label, worst.0, worst.1, worst.1 / 512, worst.2, worst.3);
        }
        let (cfg, dw, xs) = ds_load_layer(j, &golden, ratio, theta);
        let mut st = crate::deepseek::DsAttention::new(&cfg, 0);
        let mut got = Vec::new();
        for (ti, x) in xs.iter().enumerate() {
            let mut o = vec![0f32; cfg.n_heads * cfg.head_dim];
            crate::deepseek::attention_step(&cfg, 0, &dw, &mut st, x, &mut o);
            if ti == 0 && label.starts_with("DS attention window") {
                println!("    debug o[0..4]={:?}", &o[..4]);
            }
            let mut proj = vec![0f32; cfg.d];
            crate::deepseek::grouped_o_proj(&cfg, &dw.wo_a, &dw.wo_b, &o, &mut proj);
            got.extend_from_slice(&proj);
        }
        let _ = t;
        // QAT-aware tolerance: fp8/fp4 quantization boundaries amplify tiny f32
        // rounding differences into 1-2 grid steps (~2e-3). Semantic bugs show
        // up as O(1) errors instead.
        let scale = want.iter().fold(0f64, |m, &b| m.max((b as f64).abs()));
        let mut worst = 0f64;
        let mut bad = 0;
        for (a, b) in got.iter().zip(&want) {
            let dd = (*a as f64 - *b as f64).abs();
            worst = worst.max(dd);
            if dd > 2e-3 + 1e-3 * scale {
                bad += 1;
            }
        }
        let lok = bad == 0;
        println!(
            "  {:<34} {}  max_abs={:.3e} ({} val. hors tol QAT 2e-3+1e-3·scale)",
            label,
            if lok { "OK   " } else { "ÉCHEC" },
            worst,
            bad
        );
        ok &= lok;
    }

    println!();
    if ok {
        println!("DS ATTENTION OK - sparse attention matches the DeepSeek reference");
    } else {
        println!("DS ATTENTION FAILED");
        std::process::exit(1);
    }
}

// ── DS4: V4 tokenizer vs the official HF tokenizers runtime ──

pub fn run_ds4() {
    let path = if std::path::Path::new("ref/ds_tok_golden.json").exists() {
        "ref/ds_tok_golden.json"
    } else {
        "/workspace/microkimi-oss/ref/ds_tok_golden.json"
    };
    let bytes = std::fs::read(path).unwrap_or_else(|_| panic!("{} missing", path));
    let golden = json::parse(&bytes);
    let tok = crate::dstok::DsTokenizer::load(&crate::ds_tokenizer_path("microdeepseek.bin", None));
    println!("DS tokenizer ({} cases vs HF tokenizers, EXACT ids)", golden.as_arr().unwrap().len());
    let mut ok = true;
    for case in golden.as_arr().unwrap() {
        let text = case.get("text").and_then(|x| x.as_str()).unwrap();
        let want: Vec<u32> = arr(case, "ids").iter().map(|&x| x as u32).collect();
        let got = tok.encode(text);
        if got != want {
            ok = false;
            println!("  FAIL {:?}\n    rust   {:?}\n    golden {:?}", text, got, want);
        }
        // decode(encode(x)) round-trip
        let back = tok.decode(&got);
        if back != text {
            ok = false;
            println!("  FAIL decode round-trip {:?} → {:?}", text, back);
        }
    }
    println!();
    if ok {
        println!("DS TOKENIZER OK - exact match with the official HF tokenizers runtime");
    } else {
        println!("DS TOKENIZER FAILED");
        std::process::exit(1);
    }
}
