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

    // ── threads A/B under q8: default (P-cores on macOS) vs all cores.
    // The decode is bandwidth-bound; on big.LITTLE parts the E-cluster
    // adds aggregate bandwidth but also adds barrier stragglers - which
    // effect wins is machine-specific, so measure it.
    let all_cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);
    if all_cores > 0 {
        let all_s = all_cores.to_string();
        let mut allcore_ms = Vec::new();
        for _ in 0..rounds {
            if let Some(v) = ms_per_token(&run_self(
                &base,
                &[("MICROKIMI_Q8_SPINE", "1"), ("MICROKIMI_THREADS", &all_s)],
            )) {
                allcore_ms.push(v);
            }
        }
        report_decode(&format!("q8 {}-thread", all_cores), &allcore_ms);
    }

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

    // ── GPU prefill (macOS: MPS GEMM offload, in-process paired child) ──
    #[cfg(target_os = "macos")]
    {
        let out = run_self(&["qwengpubench", "--model", &model, "--rounds", "3"], &[]);
        println!("prefill gpu (MPS GEMM, in-process A/B):");
        let mut shown = false;
        for line in out.lines() {
            // only the report lines; the Metal shader compiler may echo
            // indented source excerpts in warnings, so match prefixes.
            let t = line.trim_start();
            if t.starts_with("gpu ")
                || t.starts_with("split:")
                || t.starts_with("last-position")
                || t.starts_with("precision:")
                || t.starts_with("gpu gemm check")
            {
                println!("  {}", t);
                shown = true;
            }
        }
        if !shown {
            println!("  unavailable ({})", out.lines().last().unwrap_or("no output").trim());
        }
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

/// `microkimi qwengpubench --model X.bin [--rounds N] [--tokens T]`
///
/// In-process paired CPU/GPU batched-prefill benchmark. The GPU arm is
/// the MPS GEMM offload (macOS); the first GPU prefill is a discarded
/// warm-up that pays the one-time dequant/upload of the weight stack,
/// then rounds alternate GPU and CPU on the SAME loaded model with a
/// snapshot restore between runs. Also reports the last-position logits
/// disagreement between the two arms (the offload is not bit-exact: no
/// q8 activation quantization, GPU reassociation).
pub fn gpu_prefill_cmd(args: &[String]) {
    #[cfg(not(target_os = "macos"))]
    {
        let _ = args;
        println!("qwengpubench: macOS only (Metal / MPS)");
    }
    #[cfg(target_os = "macos")]
    {
        let model_path = crate::value_flag(args, "--model").unwrap_or_else(|| {
            eprintln!("error: qwengpubench requires --model MODEL.bin");
            std::process::exit(2);
        });
        let rounds: usize = crate::value_flag(args, "--rounds")
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
        let n_tokens: usize = crate::value_flag(args, "--tokens")
            .and_then(|v| v.parse().ok())
            .unwrap_or(1024);
        crate::model::metal::set_qwen_gpu(true);
        if !crate::model::metal::mps_available() {
            println!("qwengpubench: no usable Metal/MPS context - nothing to measure");
            return;
        }
        let mut model = crate::model::qwen::QwenModel::load(&model_path);
        let vocab = (model.cfg.vocab as u32).min(50_000);
        let prompt: Vec<u32> = (0..n_tokens as u32).map(|i| (i * 7 + 3) % vocab).collect();
        let snap = model.snapshot();

        // warm-up: pays dequant + upload, and yields the GPU-arm logits
        let gpu_out = model.prefill_collect(&prompt, false);
        let gpu_logits = gpu_out.logits.last().cloned().unwrap_or_default();

        crate::model::metal::set_qwen_gpu(false);
        model.restore(&snap);
        let cpu_out = model.prefill_collect(&prompt, false);
        let cpu_logits = cpu_out.logits.last().cloned().unwrap_or_default();
        let scale = cpu_logits.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        let max_rel = gpu_logits
            .iter()
            .zip(&cpu_logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
            / scale;

        let time_prefill = |model: &mut crate::model::qwen::QwenModel| -> f64 {
            model.restore(&snap);
            let t0 = std::time::Instant::now();
            let out = model.prefill_collect(&prompt, false);
            std::hint::black_box(&out.logits);
            t0.elapsed().as_secs_f64() * 1000.0 / n_tokens as f64
        };
        let (mut gpu_ms, mut cpu_ms, mut gemm_ms) = (Vec::new(), Vec::new(), Vec::new());
        let mut gemm_calls = 0u64;
        for _ in 0..rounds {
            crate::model::metal::set_qwen_gpu(true);
            crate::model::metal::gemm_stats_take();
            gpu_ms.push(time_prefill(&mut model));
            let (calls, ms) = crate::model::metal::gemm_stats_take();
            gemm_calls = calls;
            gemm_ms.push(ms / n_tokens as f64);
            crate::model::metal::set_qwen_gpu(false);
            cpu_ms.push(time_prefill(&mut model));
        }
        let (g, c) = (median(gpu_ms.clone()), median(cpu_ms.clone()));
        let gm = median(gemm_ms.clone());
        println!("gpu prefill (MPS GEMM), {} tokens, paired rounds:", n_tokens);
        println!(
            "  gpu {:>6.2} ms/token | cpu batched {:>6.2} ms/token | {:.2}x  (gpu rounds: {:?}, cpu rounds: {:?})",
            g,
            c,
            c / g,
            gpu_ms.iter().map(|v| (v * 10.0).round() / 10.0).collect::<Vec<_>>(),
            cpu_ms.iter().map(|v| (v * 10.0).round() / 10.0).collect::<Vec<_>>()
        );
        println!(
            "  split: {:.2} ms/token inside {} GEMMs | {:.2} ms/token cpu tissue (the phase-2 porting target)",
            gm,
            gemm_calls,
            (g - gm).max(0.0)
        );
        println!("  last-position logits: max rel diff {:.2e} vs the CPU path", max_rel);
        println!(
            "  precision: {} storage in the GEMMs (MICROKIMI_QWEN_GPU_F32=1 for f32)",
            if crate::model::metal::gemm_f16_on() { "f16" } else { "f32" }
        );
    }
}
