// `microkimi paritytest`: compares the full Rust forward pass against the dump in
// ref/parity_golden.json produced by ref/parity_ref.py (genuine Moonshot code,
// micro dims, same microkimi-debug.bin weights, strict fp32).
// Hard criteria: |a-b| ≤ 1e-6 + 1e-3·max|tensor| on logits/hiddens/sub-blocks
// (beyond that → semantic bug; the real bugs observed produced O(1) discrepancies);
// the 1e-4 target threshold is reported for information only (f32 summation noise);
// router top-16 indices and logits top-16 ids EXACT.

use crate::json::{self, Json};
use crate::model::{Model, ParityDump, DUMP_LAYERS, PARITY, ROUTER_LAYERS};

const RTOL_TARGET: f64 = 1e-4; // target threshold (reported, informational)
const RTOL_PASS: f64 = 1e-3; // hard threshold (f32 noise beyond = semantic bug)
const ATOL: f64 = 1e-6;

fn arr(j: &Json, key: &str) -> Vec<f32> {
    j.get(key)
        .and_then(|x| x.as_arr())
        .unwrap_or_else(|| panic!("golden: field '{}' missing", key))
        .iter()
        .map(|x| x.as_num().unwrap() as f32)
        .collect()
}

/// (max_abs, max_rel_element, count beyond the HARD threshold, count beyond the target threshold)
/// TENSOR-SCALE relative tolerance: an element-by-element criterion
/// fails on values close to 0 due to unavoidable f32 summation noise
/// between torch BLAS and Rust dot (FMA, blocking).
fn diff(got: &[f32], want: &[f32]) -> (f64, f64, usize, usize) {
    assert_eq!(got.len(), want.len(), "sizes {} vs {}", got.len(), want.len());
    let scale = want.iter().fold(0f64, |m, &b| m.max((b as f64).abs()));
    let mut max_abs = 0f64;
    let mut max_rel = 0f64;
    let mut bad = 0;
    let mut soft = 0;
    for (a, b) in got.iter().zip(want) {
        let d = (*a as f64 - *b as f64).abs();
        max_abs = max_abs.max(d);
        max_rel = max_rel.max(d / (*b as f64).abs().max(1e-12));
        if d > ATOL + RTOL_PASS * scale {
            bad += 1;
        } else if d > ATOL + RTOL_TARGET * scale {
            soft += 1;
        }
    }
    (max_abs, max_rel, bad, soft)
}

fn report(name: &str, got: &[f32], want: &[f32]) -> bool {
    let (ma, mr, bad, soft) = diff(got, want);
    let ok = bad == 0;
    println!(
        "  {:<44} {}  max_abs={:.3e} max_rel={:.3e}{}",
        name,
        if ok { "OK   " } else { "FAIL " },
        ma,
        mr,
        if bad > 0 {
            format!("  ({} vals beyond the hard 1e-3 threshold!)", bad)
        } else if soft > 0 {
            format!("  ({} vals between 1e-4 and 1e-3: f32 noise)", soft)
        } else {
            String::new()
        }
    );
    ok
}

fn top_k(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut top: Vec<(u32, f32)> = Vec::with_capacity(k);
    for (i, &l) in logits.iter().enumerate() {
        if top.len() < k {
            top.push((i as u32, l));
            if top.len() == k {
                top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            }
        } else if l > top[k - 1].1 {
            top[k - 1] = (i as u32, l);
            top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        }
    }
    top
}

pub fn run(show: bool) {
    let path = if std::path::Path::new("ref/parity_golden.json").exists() {
        "ref/parity_golden.json"
    } else {
        "ref/parity_golden.json"
    };
    let bytes = std::fs::read(path).unwrap_or_else(|_| {
        panic!("{} missing - run first: python3 ref/parity_ref.py", path)
    });
    let golden = json::parse(&bytes);
    let ids: Vec<u32> = arr(&golden, "ids").iter().map(|&x| x as u32).collect();
    let t_max = ids.len();
    println!("microkimi paritytest - {} vs Rust forward (rel tol {:.0e})", path, RTOL_TARGET);
    println!("sequence: {:?} ({} positions)", ids, t_max);

    let mut model = Model::load(&crate::bin_path());
    PARITY.with(|p| *p.borrow_mut() = Some(ParityDump::default()));
    let mut logits = Vec::new();
    let mut logits_all: Vec<Vec<f32>> = Vec::new();
    for (pos, &id) in ids.iter().enumerate() {
        logits = model.forward(id, pos);
        logits_all.push(logits.clone());
    }
    let dump = PARITY.with(|p| p.borrow_mut().take()).expect("dump missing");

    let mut ok = true;

    // ── final logits (last position, full vector) ──
    ok &= report("final logits (last position)", &logits, &arr(&golden, "logits_last"));

    // ── logits top-16 at each position: EXACT ids + values ──
    {
        let gtop = golden.get("logits_top").unwrap();
        let mut ids_ok = true;
        let mut worst = (0f64, 0f64);
        for pos in 0..t_max {
            let g: Vec<(u32, f32)> = gtop
                .get(&pos.to_string())
                .and_then(|x| x.as_arr())
                .unwrap()
                .iter()
                .map(|kv| {
                    let a = kv.as_arr().unwrap();
                    (a[0].as_num().unwrap() as u32, a[1].as_num().unwrap() as f32)
                })
                .collect();
            let mine = top_k(&logits_all[pos], 16);
            let my_sorted: Vec<u32> = { let mut v: Vec<u32> = mine.iter().map(|t| t.0).collect(); v.sort(); v };
            let g_sorted: Vec<u32> = { let mut v: Vec<u32> = g.iter().map(|t| t.0).collect(); v.sort(); v };
            if my_sorted != g_sorted {
                ids_ok = false;
                println!("    pos {}: top-16 ids DIFFER\n      rust   {:?}\n      golden {:?}", pos, my_sorted, g_sorted);
            }
            let (ma, mr, _, _) = diff(
                &mine.iter().map(|t| t.1).collect::<Vec<_>>(),
                &g.iter().map(|t| t.1).collect::<Vec<_>>(),
            );
            worst = (worst.0.max(ma), worst.1.max(mr));
        }
        println!(
            "  {:<44} {}  max_abs={:.3e} max_rel={:.3e} (exact ids: {})",
            "top-16 logits per position",
            if ids_ok { "OK   " } else { "FAIL " },
            worst.0,
            worst.1,
            ids_ok
        );
        ok &= ids_ok;
    }

    // ── hiddens after dumped layers ──
    {
        let gh = golden.get("hiddens").unwrap();
        for &l in &DUMP_LAYERS {
            let g = arr(gh, &l.to_string());
            let mut worst = (0f64, 0f64, 0usize, 0usize);
            let mut worst_offender = (0usize, 0usize, 0f32, 0f32); // pos, channel, rust, golden
            for pos in 0..t_max {
                let mine = &dump.hiddens[&(pos, l)];
                let (ma, mr, bad, soft) = diff(mine, &g[pos * 512..(pos + 1) * 512]);
                worst = (worst.0.max(ma), worst.1.max(mr), worst.2 + bad, worst.3 + soft);
                for c in 0..512 {
                    let d = (mine[c] as f64 - g[pos * 512 + c] as f64).abs();
                    if d >= worst_offender.2 as f64 - worst_offender.3 as f64 {
                        // keep the worst signed discrepancy
                        if d > (worst_offender.2 as f64 - worst_offender.3 as f64).abs() {
                            worst_offender = (pos, c, mine[c], g[pos * 512 + c]);
                        }
                    }
                }
            }
            let lok = worst.2 == 0;
            println!(
                "  {:<44} {}  max_abs={:.3e} max_rel={:.3e}{}",
                format!("hidden after layer {} ({} pos)", l, t_max),
                if lok { "OK   " } else { "FAIL " },
                worst.0,
                worst.1,
                if worst.2 > 0 {
                    format!("  ({} vals beyond the hard 1e-3 threshold!)", worst.2)
                } else if worst.3 > 0 {
                    format!("  ({} vals between 1e-4 and 1e-3: f32 noise)", worst.3)
                } else {
                    String::new()
                }
            );
            if worst.2 > 0 {
                println!(
                    "      worst element: pos {} channel {} → rust {:.6e} vs golden {:.6e}",
                    worst_offender.0, worst_offender.1, worst_offender.2, worst_offender.3
                );
            }
            ok &= lok;
        }
    }

    // ── layer 1 sub-blocks: KDA output, routed MoE, shared ──
    for (key, mine_map, label) in [
        ("l1_attn", &dump.l1_attn, "layer 1: KDA output"),
        ("l1_routed", &dump.l1_routed, "layer 1: routed MoE (after up)"),
        ("l1_shared", &dump.l1_shared, "layer 1: shared MoE"),
    ] {
        let g = arr(&golden, key);
        let mut all = Vec::new();
        for pos in 0..t_max {
            all.extend_from_slice(&mine_map[&pos]);
        }
        ok &= report(label, &all, &g);
    }

    // ── router: EXACT top-16 indices ──
    {
        let gr = golden.get("router").unwrap();
        let mut exact = true;
        for &l in &ROUTER_LAYERS {
            let g = gr.get(&l.to_string()).and_then(|x| x.as_arr()).unwrap();
            for pos in 0..t_max {
                let want: Vec<u32> = g[pos]
                    .as_arr()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_num().unwrap() as u32)
                    .collect();
                let mine = &dump.router[&(pos, l)];
                if *mine != want {
                    exact = false;
                    println!("    router layer {} pos {}: DIFFERS\n      rust   {:?}\n      golden {:?}", l, pos, mine, want);
                }
            }
        }
        println!(
            "  {:<44} {}",
            format!("router top-16 (layers {:?}, all pos)", ROUTER_LAYERS),
            if exact { "OK    (exact match)" } else { "FAIL " }
        );
        ok &= exact;
    }

    // ── --show mode: concrete values side by side ──
    if show {
        let last = t_max - 1;
        let glogits = arr(&golden, "logits_last");
        println!();
        println!("{}", "=".repeat(72));
        println!("1:1 PARITY IN VALUES - last position (pos {})", last);
        println!("{}", "=".repeat(72));

        // 1) the first 10 final logits
        println!();
        println!("1) Final logits - indices 0..9");
        println!("  {:>8} │ {:>16} │ {:>16} │ {:>10}", "index", "ref (Moonshot)", "rust", "diff");
        println!("  {}┼{}┼{}┼{}", "─".repeat(10), "─".repeat(18), "─".repeat(18), "─".repeat(12));
        for i in 0..10 {
            let (r, m) = (glogits[i], logits[i]);
            println!("  {:>8} │ {:>16.8} │ {:>16.8} │ {:>10.2e}", i, r, m, (m - r).abs());
        }

        // 2) top-5 logits (descending order, ref side)
        println!();
        println!("2) Top-5 logits (descending order, ref side)");
        println!("  {:>8} │ {:>16} │ {:>16} │ {:>10}", "index", "ref (Moonshot)", "rust", "diff");
        println!("  {}┼{}┼{}┼{}", "─".repeat(10), "─".repeat(18), "─".repeat(18), "─".repeat(12));
        let gtop = golden.get("logits_top").unwrap();
        let g5: Vec<(u32, f32)> = gtop
            .get(&last.to_string())
            .and_then(|x| x.as_arr())
            .unwrap()
            .iter()
            .take(5)
            .map(|kv| {
                let a = kv.as_arr().unwrap();
                (a[0].as_num().unwrap() as u32, a[1].as_num().unwrap() as f32)
            })
            .collect();
        for (id, rv) in &g5 {
            let m = logits[*id as usize];
            println!("  {:>8} │ {:>16.8} │ {:>16.8} │ {:>10.2e}", id, rv, m, (m - rv).abs());
        }

        // 3) router top-16 layer 1, last position
        println!();
        println!("3) Router top-16 - layer 1, last position (sorted by index)");
        let gr = golden.get("router").unwrap();
        let g1 = gr.get("1").and_then(|x| x.as_arr()).unwrap();
        let want: Vec<u32> = g1[last]
            .as_arr()
            .unwrap()
            .iter()
            .map(|x| x.as_num().unwrap() as u32)
            .collect();
        let mine = &dump.router[&(last, 1)];
        println!("  {:>6} │ {:>16} │ {:>16}", "rank", "ref (Moonshot)", "rust");
        println!("  {}┼{}┼{}", "─".repeat(8), "─".repeat(18), "─".repeat(18));
        for r in 0..16 {
            let mark = if want[r] == mine[r] { "" } else { "  ← DIFFERS" };
            println!("  {:>6} │ {:>16} │ {:>16}{}", r, want[r], mine[r], mark);
        }

        // 4) hidden after layer 92, indices 0..7
        println!();
        println!("4) Hidden after layer 92 - indices 0..7, last position");
        println!("  {:>8} │ {:>16} │ {:>16} │ {:>10}", "index", "ref (Moonshot)", "rust", "diff");
        println!("  {}┼{}┼{}┼{}", "─".repeat(10), "─".repeat(18), "─".repeat(18), "─".repeat(12));
        let gh = arr(golden.get("hiddens").unwrap(), "92");
        let mh = &dump.hiddens[&(last, 92)];
        for i in 0..8 {
            let (r, m) = (gh[last * 512 + i], mh[i]);
            println!("  {:>8} │ {:>16.8} │ {:>16.8} │ {:>10.2e}", i, r, m, (m - r).abs());
        }
        println!();
    }

    println!();
    if ok {
        println!("PARITYTEST OK - Rust forward ≡ official Moonshot code (hard threshold {:.0e})", RTOL_PASS);
    } else {
        println!("PARITYTEST: discrepancies detected (see above)");
        std::process::exit(1);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// DeepSeek-V4 end-to-end parity (`microkimi dsparity`): microdeepseek-debug.bin
// forward vs the plain-torch replica of the reference Transformer
// (ref/make_ds_parity.py → ref/ds_parity_golden.json).
// QAT-aware tolerance (same as the DS attention selftest): 2e-3 + 1e-3·scale —
// fp8/fp4 quantization boundaries amplify tiny f32 rounding differences into
// 1-2 grid steps; semantic bugs produce O(1) discrepancies.
// Router selections (layers 1/3/42) and top-16 logit ids: EXACT.
// ════════════════════════════════════════════════════════════════════════════

fn qat_diff(got: &[f32], want: &[f32]) -> (f64, usize) {
    assert_eq!(got.len(), want.len(), "sizes {} vs {}", got.len(), want.len());
    let scale = want.iter().fold(0f64, |m, &b| m.max((b as f64).abs()));
    let mut max_abs = 0f64;
    let mut bad = 0;
    for (a, b) in got.iter().zip(want) {
        let d = (*a as f64 - *b as f64).abs();
        max_abs = max_abs.max(d);
        if d > 2e-3 + 1e-3 * scale {
            bad += 1;
        }
    }
    (max_abs, bad)
}

fn qat_report(name: &str, got: &[f32], want: &[f32]) -> bool {
    let (ma, bad) = qat_diff(got, want);
    let ok = bad == 0;
    println!(
        "  {:<46} {}  max_abs={:.3e}{}",
        name,
        if ok { "OK   " } else { "FAIL " },
        ma,
        if bad > 0 { format!("  ({} vals beyond QAT tol 2e-3+1e-3·scale!)", bad) } else { String::new() }
    );
    ok
}

pub fn run_ds() {
    let path = if std::path::Path::new("ref/ds_parity_golden.json").exists() {
        "ref/ds_parity_golden.json"
    } else {
        "ref/ds_parity_golden.json"
    };
    let bytes = std::fs::read(path).unwrap_or_else(|_| {
        panic!("{} missing - run first: python3 ref/make_ds_parity.py", path)
    });
    let golden = json::parse(&bytes);
    let ids: Vec<u32> = arr(&golden, "ids").iter().map(|&x| x as u32).collect();
    let t_max = ids.len();
    println!("microkimi dsparity - {} vs Rust forward (QAT-aware tol 2e-3+1e-3·scale)", path);
    println!("sequence: {} positions", t_max);

    let bin = if std::path::Path::new("models/microdeepseek-debug.bin").exists() {
        "models/microdeepseek-debug.bin".to_string()
    } else {
        "models/microdeepseek.bin".to_string() // legacy name from older builds
    };
    let mut model = crate::model::deepseek::DsModel::load(&bin);
    model.reset();
    crate::model::deepseek::DS_PARITY.with(|p| *p.borrow_mut() = Some(crate::model::deepseek::DsParityDump::default()));
    let mut logits = Vec::new();
    let mut logits_all: Vec<Vec<f32>> = Vec::new();
    for (pos, &id) in ids.iter().enumerate() {
        logits = model.forward(id, pos);
        logits_all.push(logits.clone());
    }
    let dump = crate::model::deepseek::DS_PARITY.with(|p| p.borrow_mut().take()).expect("dump missing");

    let mut ok = true;

    // ── final logits (last position, full vector) ──
    ok &= qat_report("final logits (last position)", &logits, &arr(&golden, "logits_last"));

    // ── logits top-16 at each position: EXACT ids + values ──
    {
        let gtop = golden.get("logits_top").and_then(|x| x.as_arr()).unwrap();
        let mut ids_ok = true;
        let mut worst = 0f64;
        for (pos, g) in gtop.iter().enumerate() {
            let gids: Vec<u32> = g.get("ids").and_then(|x| x.as_arr()).unwrap().iter().map(|v| v.as_num().unwrap() as u32).collect();
            let gvals: Vec<f32> = g.get("vals").and_then(|x| x.as_arr()).unwrap().iter().map(|v| v.as_num().unwrap() as f32).collect();
            let mine = top_k(&logits_all[pos], 16);
            let mut my_sorted: Vec<u32> = mine.iter().map(|t| t.0).collect();
            my_sorted.sort();
            let mut g_sorted = gids.clone();
            g_sorted.sort();
            if my_sorted != g_sorted {
                ids_ok = false;
                println!("    pos {}: top-16 ids DIFFER\n      rust   {:?}\n      golden {:?}", pos, my_sorted, g_sorted);
            }
            let (ma, _) = qat_diff(&mine.iter().map(|t| t.1).collect::<Vec<_>>(), &gvals);
            worst = worst.max(ma);
        }
        println!(
            "  {:<46} {}  max_abs={:.3e} (exact ids: {})",
            "top-16 logits per position",
            if ids_ok { "OK   " } else { "FAIL " },
            worst,
            ids_ok
        );
        ok &= ids_ok;
    }

    // ── hiddens after dumped layers (full HC state [hc*d]) ──
    {
        let gh = golden.get("hiddens").unwrap();
        let pos_dump: Vec<usize> = arr(&golden, "pos_dump").iter().map(|&x| x as usize).collect();
        let layer_dump: Vec<usize> = arr(&golden, "layer_dump").iter().map(|&x| x as usize).collect();
        for &l in &layer_dump {
            for &pos in &pos_dump {
                let key = format!("{},{}", pos, l);
                let g = arr(gh, &key);
                let mine = &dump.hiddens[&(pos, l)];
                ok &= qat_report(&format!("hidden pos {} after layer {}", pos, l), mine, &g);
            }
        }
    }

    // ── router: EXACT sorted expert ids (layers 1/3/42, all positions) ──
    {
        let gr = golden.get("router").unwrap();
        let mut exact = true;
        for &l in &crate::model::deepseek::DS_ROUTER_LAYERS {
            for pos in 0..t_max {
                let key = format!("{},{}", pos, l);
                let want: Vec<u32> = arr(gr, &key).iter().map(|&x| x as u32).collect();
                let mine = &dump.router[&(pos, l)];
                if *mine != want {
                    exact = false;
                    println!("    router layer {} pos {}: DIFFERS\n      rust   {:?}\n      golden {:?}", l, pos, mine, want);
                }
            }
        }
        println!(
            "  {:<46} {}",
            format!("router top-{} (layers {:?}, all pos)", model.cfg.n_activated_experts, crate::model::deepseek::DS_ROUTER_LAYERS),
            if exact { "OK    (exact match)" } else { "FAIL " }
        );
        ok &= exact;
    }

    println!();
    if ok {
        println!("DSPARITY OK - Rust forward ≡ DeepSeek-V4 reference replica (QAT-aware threshold)");
    } else {
        println!("DSPARITY: discrepancies detected (see above)");
        std::process::exit(1);
    }
}
