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
