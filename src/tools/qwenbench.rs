//! `microkimi qwenbench --model X.bin [--steps N]`: the standard paired
//! benchmark battery for a converted dense Qwen model, one command, one
//! report. Designed for bare-metal runs (macOS on Apple silicon, Linux):
//! every comparison is paired and in-process where possible, and
//! env-dependent modes run through clean child processes so OnceLock
//! caches never leak across arms.
//!
//! Battery:
//!   decode      f32 vs q8 spine vs fp4 spine (ms/token, paired prompts)
//!   sdot        q8 spine with and without the SDOT kernels
//!   prefill     batched vs sequential ingestion of a ~1k-token prompt
//!   lanes       in-process A/B aggregate throughput at 4 and 8 lanes (q8)
//!   mtp         plain vs chained speculative decode (q8 spine)
//!
//! Compare against llama.cpp on the SAME machine with the same
//! checkpoint quantized to Q8_0 (see BENCH.md for the exact protocol).

use std::process::Command;

fn run_self(args: &[&str], envs: &[(&str, &str)]) -> String {
    let exe = std::env::current_exe().expect("current_exe");
    let mut cmd = Command::new(exe);
    cmd.args(args);
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("child benchmark run");
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

fn ms_per_token(output: &str) -> Option<f64> {
    // matches the generation report "NNN ms/token"
    for line in output.lines().rev() {
        if let Some(idx) = line.find(" ms/token") {
            let head = &line[..idx];
            let num = head.rsplit(|c: char| !(c.is_ascii_digit() || c == '.')).next()?;
            if let Ok(v) = num.parse() {
                return Some(v);
            }
        }
    }
    None
}

fn prefill_ms(output: &str) -> Option<f64> {
    for line in output.lines() {
        if line.contains("tokens (") && line.contains("ms/token)") {
            let start = line.rfind('(')? + 1;
            let end = line.rfind(" ms/token)")?;
            return line[start..end].parse().ok();
        }
    }
    None
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

pub fn run(args: &[String]) {
    let model = crate::value_flag(args, "--model").unwrap_or_else(|| {
        eprintln!("error: qwenbench requires --model MODEL.bin");
        std::process::exit(2);
    });
    let steps: usize = crate::value_flag(args, "--steps")
        .and_then(|v| v.parse().ok())
        .unwrap_or(48);
    let rounds: usize = crate::value_flag(args, "--rounds")
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let cfg = crate::quant::weights::read_config(&model);
    let Some(qcfg) = cfg.qwen.clone() else {
        eprintln!("error: {} is not a Qwen model", model);
        std::process::exit(1);
    };
    println!(
        "qwenbench: {} ({} layers, hidden {}, {})",
        model,
        qcfg.n_layers,
        qcfg.d,
        if qcfg.is_dense() { "dense" } else { "MoE" }
    );
    let prompt = "The industrial revolution transformed European cities because";
    let steps_s = steps.to_string();
    let base: Vec<&str> = vec![
        "run", prompt, "--model", &model, "--raw", "--max-new", &steps_s,
    ];

    // ── decode: f32 vs q8 vs fp4, paired rounds ──
    let mut f32_ms = Vec::new();
    let mut q8_ms = Vec::new();
    let mut fp4_ms = Vec::new();
    for _ in 0..rounds {
        if let Some(v) = ms_per_token(&run_self(&base, &[])) {
            f32_ms.push(v);
        }
        if let Some(v) = ms_per_token(&run_self(&base, &[("MICROKIMI_Q8_SPINE", "1")])) {
            q8_ms.push(v);
        }
        if let Some(v) = ms_per_token(&run_self(&base, &[("MICROKIMI_FP4_SPINE", "1")])) {
            fp4_ms.push(v);
        }
    }
    let report_decode = |label: &str, xs: &Vec<f64>| {
        if xs.is_empty() {
            println!("  {:<12} unavailable", label);
        } else {
            let m = median(xs.clone());
            println!(
                "  {:<12} {:>7.1} ms/token   {:>6.1} tok/s   (rounds: {:?})",
                label,
                m,
                1000.0 / m,
                xs.iter().map(|v| *v as i64).collect::<Vec<_>>()
            );
        }
    };
    println!("decode (single stream, {} tokens):", steps);
    report_decode("f32 spine", &f32_ms);
    report_decode("q8 spine", &q8_ms);
    report_decode("fp4 spine", &fp4_ms);

    // ── sdot A/B under q8 ──
    let mut nosdot_ms = Vec::new();
    for _ in 0..rounds {
        if let Some(v) = ms_per_token(&run_self(
            &base,
            &[("MICROKIMI_Q8_SPINE", "1"), ("MICROKIMI_NO_SDOT", "1")],
        )) {
            nosdot_ms.push(v);
        }
    }
    println!("kernels:");
    report_decode("q8 no-sdot", &nosdot_ms);

    // ── prefill A/B ──
    let long_prompt = "The history of computing spans mechanical calculators, vacuum tubes, transistors, integrated circuits, and modern accelerators. ".repeat(40);
    let pre: Vec<&str> = vec![
        "run", &long_prompt, "--model", &model, "--raw", "--max-new", "2", "--debug",
    ];
    let batched = prefill_ms(&run_self(&pre, &[]));
    let sequential = prefill_ms(&run_self(&pre, &[("MICROKIMI_NO_QWEN_BATCH", "1")]));
    println!("prefill (~1k-token prompt):");
    match (batched, sequential) {
        (Some(b), Some(s)) => println!(
            "  batched {:>6.1} ms/token | sequential {:>6.1} ms/token | {:.1}x",
            b,
            s,
            s / b
        ),
        _ => println!("  unavailable"),
    }

    // ── lanes A/B (q8 spine) ──
    println!("lane-batched aggregate (q8 spine, in-process A/B):");
    for lanes in [4usize, 8] {
        let lanes_s = lanes.to_string();
        let out = run_self(
            &[
                "lanebench", "--model", &model, "--lanes", &lanes_s, "--steps", "16", "--ab",
                "--rounds", "4",
            ],
            &[("MICROKIMI_Q8_SPINE", "1")],
        );
        for line in out.lines() {
            if let Some(rest) = line.strip_prefix("median aggregate speedup at ") {
                println!("  {}", rest.trim());
            }
        }
    }

    // ── MTP (only when the model carries the head) ──
    if qcfg.mtp_layers > 0 {
        let mut plain = Vec::new();
        let mut drafted = Vec::new();
        for _ in 0..rounds {
            if let Some(v) = ms_per_token(&run_self(&base, &[("MICROKIMI_Q8_SPINE", "1")])) {
                plain.push(v);
            }
            let mtp_args: Vec<&str> = base.iter().copied().chain(["--mtp"]).collect();
            if let Some(v) = ms_per_token(&run_self(&mtp_args, &[("MICROKIMI_Q8_SPINE", "1")])) {
                drafted.push(v);
            }
        }
        println!("speculative decode (q8 spine):");
        report_decode("plain", &plain);
        report_decode("mtp chain", &drafted);
    }
    println!(
        "compare with llama.cpp on this machine per BENCH.md (same checkpoint, Q8_0, llama-bench)."
    );
}
