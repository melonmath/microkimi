// microkimi - 100% Rust inference engine + weight builder, zero dependencies.
// Micro reimplementation of the Kimi K3 architecture (MoE): same counts
// (93 layers, 69 KDA + 24 MLA, 896 experts top-16 + 2 shared, AttnRes block 12),
// same mechanisms (KDA, MLA NoPE, latent MoE, SiTU, MXFP4, noaux_tc router),
// reduced dims. std only.

mod config;
mod json;
mod memory;
mod model;
mod quant;
mod sha256;
mod stream;
mod tokenizer;
mod tools;
mod unicode_nfc;

use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("help");
    let t0 = Instant::now();

    // unified arch selector: --arch k3 (default) | dsv4
    let arch = args
        .iter()
        .position(|a| a == "--arch")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("k3");

    // env-armed routing statistics sketch (MICROKIMI_ROUTECMS=path.bin on
    // run/chat/prefill/absorb); no-op when the variable is absent
    stream::route_sketch::start_from_env();

    match cmd {
        "build" => {
            if arch == "dsv4" { tools::build_ds::run() } else { tools::build::run() }
        }
        "build-ds" => tools::build_ds::run(), // alias for `build --arch dsv4`
        // Convert a local Qwen3.5-MoE Hugging Face checkpoint to MKIM0002.
        "convert-qwen" => tools::convert_qwen::run(&args),
        // Deterministic JSONL completions with one model/adapter load.
        "complete-batch" => tools::complete_batch::run(&args),
        // microkimi slice --model X.bin --out Y.bin [--hidden N] [--experts N] [--layers "0-11"]
        "slice" => tools::slice::run(&args),
        "slice-qwen-vocab" => tools::slice_qwen::run(&args),
        "serve" => tools::serve::run(&args),
        // microkimi shadow --model X.bin [--out X.shadows]  (VQ1 expert shadows for --stream-fallback)
        "shadow" => stream::shadow::cmd(&args),
        "selftest" => { tools::selftest::run(); tools::selftest::run_ds(); tools::selftest::run_ds2(); tools::selftest::run_ds3(); tools::selftest::run_ds4(); tools::selftest::run_packed_emul(); tools::selftest::run_q8(); tools::selftest::run_flash(); tools::selftest::run_kvq8(); },
        "metaltest" => metaltest_cmd(),
        "metaltest-packed" => metaltest_packed_cmd(),
        "gputest" => gputest_cmd(),
        "dstest" => dstest_cmd(),
        "qwen-dump" => model::qwen::dump_cmd(&args),
        "lanebench" => model::qwen::lanebench_cmd(&args),
        "qwenbench" => tools::qwenbench::run(&args),
        "qwengpubench" => tools::qwenbench::gpu_prefill_cmd(&args),
        "qwen-tok" => model::qwentok::dump_cmd(&args),
        "gpubench" => gpubench_cmd(&args),
        "paritytest" | "parity" => {
            if arch == "dsv4" { tools::parity::run_ds() } else { tools::parity::run(args.iter().any(|a| a == "--show")) }
        }
        "dsparity" => tools::parity::run_ds(), // alias for `parity --arch dsv4`
        "run" => {
            // microkimi run "prompt" [--max-new N] [--model X.bin] [--vocab V.json]
            //                        [--memory mem.mkmem] [--save mem.mkmem]
            //                        [--temp T] [--top-p P] [--seed N]
            //                        [--exit-layer N] [--logit-lens [--lens-probe "TOKEN"]]
            let skip = flag_value_positions(&args, &["--adapter"]);
            let positional: Vec<&String> = args
                .iter()
                .enumerate()
                .skip(2)
                .filter(|(i, a)| !a.starts_with("--") && !skip.contains(i))
                .map(|(_, a)| a)
                .collect();
            let prompt = positional.first().map(|s| s.to_string()).unwrap_or_else(|| "Hello".to_string());
            let max_new = args
                .iter()
                .position(|a| a == "--max-new")
                .and_then(|i| args.get(i + 1))
                .and_then(|s| s.parse().ok())
                .unwrap_or(20);
            model::set_gpu(args.iter().any(|a| a == "--gpu"));
            model::set_dump_hidden(args.iter().any(|a| a == "--dump-hidden"));
            model::set_logit_lens(
                args.iter().any(|a| a == "--logit-lens" || a == "--logit-lens-all"),
                args.iter().any(|a| a == "--logit-lens-all"),
            );
            run_inference(&prompt, max_new, true, &model_flag(&args), vocab_flag(&args), args.iter().any(|a| a == "--debug-routing"), args.iter().any(|a| a == "--raw"), &value_flag(&args, "--memory"), &value_flag(&args, "--save"), &mut sampler_flag(&args), stream_ram_flag(&args), exit_layer_flag(&args), &lens_probe_strings(&args));
        }
        "chat" => {
            model::set_gpu(args.iter().any(|a| a == "--gpu"));
            model::set_logit_lens(
                args.iter().any(|a| a == "--logit-lens" || a == "--logit-lens-all"),
                args.iter().any(|a| a == "--logit-lens-all"),
            );
            chat_loop(&model_flag(&args), vocab_flag(&args), args.iter().any(|a| a == "--debug-routing"), args.iter().any(|a| a == "--raw"), value_flag(&args, "--memory"), value_flag(&args, "--save"), &mut sampler_flag(&args), stream_ram_flag(&args), exit_layer_flag(&args), &lens_probe_strings(&args));
        }
        // microkimi prefill "text" --save mem.mkmem [--model X.bin] [--vocab V.json] [--chat]
        "prefill" => {
            let positional: Vec<&String> = args.iter().skip(2).filter(|a| !a.starts_with("--")).collect();
            let text = positional.first().map(|s| s.to_string()).unwrap_or_default();
            let Some(save) = value_flag(&args, "--save") else {
                eprintln!("error: prefill requires --save mem.mkmem");
                std::process::exit(1);
            };
            model::set_gpu(args.iter().any(|a| a == "--gpu"));
            prefill_cmd(&text, &save, &model_flag(&args), vocab_flag(&args), args.iter().any(|a| a == "--chat"), stream_ram_flag(&args));
        }
        // microkimi absorb file.txt --out pack.mkmem [--model X.bin] [--vocab V.json] [--chat]
        "absorb" => absorb_cmd(&args),
        // microkimi mkmem-merge A.mkmem B.mkmem [C.mkmem ...] --out AB.mkmem [--shuffle N]
        // (experiment) KDA state additivity: s = element-wise sum over inputs,
        // conv/MLA/logits from the first input. --shuffle N shuffles the s of
        // the Nth input (1-based) before summing: shuffled-garbage control.
        "mkmem-merge" => {
            let skip = flag_value_positions(&args, &["--out", "--shuffle"]);
            let paths: Vec<String> = args
                .iter()
                .enumerate()
                .skip(2)
                .filter(|(i, a)| !a.starts_with("--") && !skip.contains(i))
                .map(|(_, a)| a.clone())
                .collect();
            let Some(out) = value_flag(&args, "--out") else {
                eprintln!("error: mkmem-merge requires --out AB.mkmem");
                std::process::exit(1);
            };
            let shuffle_idx = value_flag(&args, "--shuffle").and_then(|s| s.parse().ok());
            let avg = args.iter().any(|a| a == "--avg");
            match memory::memory_pack::merge(&paths, &out, shuffle_idx, avg) {
                Ok(()) => println!("merged {} states -> {} (KDA s summed, conv/MLA/logits from {})", paths.len(), out, paths[0]),
                Err(e) => {
                    eprintln!("error: {}", e);
                    std::process::exit(1);
                }
            }
        }
        // microkimi mkmem-div REF.mkmem X.mkmem [Y.mkmem ...] --prompt "text" [--max-new N]
        // (hidden debug tool) greedy-generates N tokens from the prompt on top
        // of each state and reports the top-1 agreement of X, Y, ... vs REF.
        "mkmem-div" => mkmem_div_cmd(&args),
        // microkimi routestats "prompt" [--model X.bin] [--max-new N] [--out routecms.bin]
        // runs one turn with the count-min routing sketch armed, then saves it
        "routestats" => routestats_cmd(&args),
        // microkimi cmsinfo sketch.bin: top-50 (layer, expert, count) + coverage
        "cmsinfo" => stream::route_sketch::info_cmd(&args),
        // microkimi decay mem.mkmem --half-life H --out mem2.mkmem [--units U]
        // exp2 partial forgetting: KDA s *= 2^(-U/H), conv/MLA/logits untouched
        "decay" => {
            let skip = flag_value_positions(&args, &["--half-life", "--out", "--units"]);
            let positional: Vec<&String> = args
                .iter()
                .enumerate()
                .skip(2)
                .filter(|(i, a)| !a.starts_with("--") && !skip.contains(i))
                .map(|(_, a)| a)
                .collect();
            let (Some(mem), Some(hl), Some(out)) = (
                positional.first(),
                value_flag(&args, "--half-life").and_then(|s| s.parse::<f64>().ok()),
                value_flag(&args, "--out"),
            ) else {
                eprintln!("error: usage: microkimi decay mem.mkmem --half-life H --out mem2.mkmem [--units U]");
                std::process::exit(1);
            };
            let units = value_flag(&args, "--units").and_then(|s| s.parse().ok()).unwrap_or(1.0);
            match memory::memory_pack::decay(mem, hl, units, &out) {
                Ok(f) => println!("decay: {} -> {} ({} units at half-life {}, KDA s scaled by {:.6})", mem, out, units, hl, f),
                Err(e) => {
                    eprintln!("error: {}", e);
                    std::process::exit(1);
                }
            }
        }
        // microkimi merge a.mkmem b.mkmem --alpha A --out m.mkmem
        // (experiment) linear interpolation of the KDA states AND conv windows
        // (same alpha); MLA caches + logits from file A. NOT a semantic merge.
        "merge" => {
            let skip = flag_value_positions(&args, &["--alpha", "--out"]);
            let positional: Vec<&String> = args
                .iter()
                .enumerate()
                .skip(2)
                .filter(|(i, a)| !a.starts_with("--") && !skip.contains(i))
                .map(|(_, a)| a)
                .collect();
            let (Some(a), Some(b), Some(out)) = (positional.first(), positional.get(1), value_flag(&args, "--out")) else {
                eprintln!("error: usage: microkimi merge a.mkmem b.mkmem --alpha A --out m.mkmem");
                std::process::exit(1);
            };
            let alpha = value_flag(&args, "--alpha").and_then(|s| s.parse().ok()).unwrap_or(0.5);
            match memory::memory_pack::merge_interp(a, b, alpha, &out) {
                Ok(()) => println!(
                    "merged {} + {} -> {} (alpha={}, experimental linear blend of two SSM states, not a semantic merge)",
                    a, b, out, alpha
                ),
                Err(e) => {
                    eprintln!("error: {}", e);
                    std::process::exit(1);
                }
            }
        }
        // microkimi streamtest --model https://huggingface.co/org/repo [--cache-dir D] [--stream-disk N]
        "streamtest" => stream::streamtest(&args),
        // microkimi eval --model X.bin [--vocab V.json] [--max-new N] [--ppl-file F] [--json out.json]
        "eval" => tools::eval::run(&args),
        // microkimi calibrate --model X.bin --text corpus.txt --out imatrix.bin [--vocab V.json] [--max-tokens N]
        "calibrate" => quant::imatrix::calibrate_cmd(&args),
        // microkimi mxfp4test --model X.bin [--tensors N]
        // (hidden measurement) e8m0 vs quadratic scale encoding on real tensors
        "mxfp4test" => quant::mxfp4::test_cmd(&args),
        // (hidden bench) matvec kernel timing: 1024x512 and 163840x1024, 100 iters
        "dotbench" => dotbench_cmd(),
        // microkimi cache --info | microkimi cache --clean [--repo X]
        "cache" => stream::cache_cmd(&args),
        // microkimi pck --info | microkimi pck --clean [--model X.bin]
        // prefix cache of the chat (<model>.pck/ by default, MICROKIMI_PCK_DIR)
        "pck" => memory::prefix_cache::cmd(&args),
        // microkimi cachereplay trace.bin [--top-k K] [--predict N]
        "cachereplay" => tools::replay::run(&args),
        // microkimi routebuild store.routes trace.bin [trace2.bin ...]
        "routebuild" => tools::replay::routebuild(&args),
        // debug command (debug helper): prints the tokenization of a text
        "tok" => {
            let tok = tokenizer::Tokenizer::load(&tokenizer_path());
            let text = args.get(2).cloned().unwrap_or_default();
            let any = tokenizer::AnyTokenizer::Full(tok);
            let ids = if text.starts_with("--chat2") {
                let q = args.get(3).cloned().unwrap_or_default();
                any.encode_chat(&[("hello".to_string(), "hi".to_string())], &q)
            } else if text.starts_with("--chat") {
                let q = args.get(3).cloned().unwrap_or_default();
                any.encode_chat_user(&q)
            } else {
                match &any {
                    tokenizer::AnyTokenizer::Full(t) => t.encode(&text),
                    _ => unreachable!(),
                }
            };
            println!("{:?}", ids);
            if !text.starts_with("--") {
                for &id in &ids {
                    println!("  {:6} = {:?}", id, any.decode_id(id));
                }
            }
        }
        _ => {
            println!("microkimi - micro K3 inference engine, zero dependencies");
            println!("usage:");
            println!("Build & slice:");
            println!("  microkimi build                      builds microkimi-debug.bin (K3 fetch + generation)");
            println!("  microkimi build-ds                   builds microdeepseek-debug.bin (DeepSeek-V4 fetch + generation)");
            println!("  microkimi convert-qwen --source DIR --out MODEL.bin [--audit-only] [--imatrix F]");
            println!("                                         converts a local Qwen3.5-family text checkpoint (MoE or dense);");
            println!("                                         f32 spine + MXFP4 experts or dense MLP, bounded conversion RAM;");
            println!("                                         --imatrix: calibration-weighted MXFP4 scales (dense, see calibrate)");
            println!("  microkimi qwenbench --model X.bin [--steps N] [--rounds R]");
            println!("                                         full paired benchmark battery (decode spines, sdot, prefill,");
            println!("                                         lanes, mtp) with one report; protocol vs llama.cpp in BENCH.md");
            println!("  microkimi lanebench --model X.bin [--lanes N] [--steps M]");
            println!("                                         aggregate decode throughput of lane-batched decoding (Qwen)");
            println!("  microkimi complete-batch --model X.bin --input REQUESTS.jsonl --out RESULTS.jsonl");
            println!("                        deterministic greedy completions, one model load; --chat optional;");
            println!("  microkimi slice --model X.bin --out Y.bin [--hidden N] [--experts N] [--layers \"0-11\"]");
            println!("                                         structural pruning (channels / experts / layers)");            println!("      --vocab-top N freqfile             vocabulary pruning: top-N frequent rows + all specials");
            println!("      --vocab-list ids.txt               same, but the kept rows come from an explicit id list");
            println!("                                         (one per line, '#' comments; nano/vocab_cross.py output);");
            println!("                                         mutually exclusive with --vocab-top");            println!("      --merge-experts N                  merge instead of delete: usage-weighted k-means clusters");
            println!("                                         the experts of every MoE layer into N averaged experts");
            println!("                                         (router rows merged by logsumexp, routing mass conserved)");            println!("      --cold-vq N                        precision tiering: top-N experts stay mxfp4, the");
            println!("                                         cold tail becomes VQ1 (0.5 bit, shared codebook)");
            println!("      --imatrix FILE (with --cold-vq)      activation-weighted VQ codebook (see calibrate);");
            println!("                                         --imatrix-score-only: report only, blind codebook");
            println!("      --expert-order=frequency           physical reorder of expert blobs, hottest first");
            println!("      --route-cms SKETCH (with --expert-order)  routing frequency sketch (MICROKIMI_ROUTECMS);");
            println!("                                         hot experts become file-adjacent for stream run fusion");
            println!("      --model also accepts safetensors: model.safetensors, a directory with an index,");
            println!("      or https://huggingface.co/org/repo (range requests: only the needed tensors");
            println!("      and, for expert ranking, only the weight_scale bytes are fetched)");
            println!("  microkimi slice-qwen-vocab --model X.bin --out Y.bin --top N --freqfile F [--vocab T.json]");
            println!("                                         vocabulary pruning for Qwen models: keeps the special block,");
            println!("                                         the 256 byte tokens, the chat-template pieces and the top-N");
            println!("                                         frequent ids; writes qwen.vocabmap.json beside the output");
            println!("                                         (dropped tokens re-encode as byte tokens at runtime)");
            println!("  microkimi shadow --model X.bin [--out X.shadows]");
            println!("                                         VQ1 (0.5-bit) shadows of EVERY expert + one global");
            println!("                                         codebook, the resident low-precision tier served on");
            println!("                                         expert cache misses by --stream-fallback");
            println!("  microkimi calibrate --model X.bin --text corpus.txt --out imatrix.bin [--max-tokens N]");
            println!("                                         activation second moments per expert-matrix input column,");
            println!("                                         consumed by slice --cold-vq --imatrix (weighted VQ)");
            println!("Run:");
            println!("  microkimi run \"prompt\" [--max-new N]  greedy generation with detailed steps");
            println!("  microkimi chat                       interactive with history ('quit' to exit)");
            println!("  microkimi serve --model X.bin [--host 127.0.0.1] [--port 8080] [--mtp] [--max-new N]");
            println!("                                         OpenAI-compatible HTTP endpoint (Qwen models): /v1/chat/completions");
            println!("                                         and /v1/completions with SSE streaming, sampling per request,");
            println!("                                         cross-request chat prefix cache; binds 127.0.0.1 by default");
            println!("  microkimi prefill \"text\" --save mem.mkmem  ingest text, snapshot the state (.mkmem)");
            println!("  microkimi absorb file.txt --out pack.mkmem  ingest a document file, snapshot the state (.mkmem)");
            println!("  run/chat options: --model X.bin --vocab vocab_nano.json (auto if next to the .bin)");
            println!("                    --raw (raw completion, for nanokimi)  --debug-routing  --gpu (Metal, macOS)");
            println!("                    --memory mem.mkmem (resume a state)  --save mem.mkmem (snapshot after the run)");
            println!("                    --adapter skill.mkap (repeatable; hash-bound low-rank packs folded");
            println!("                        into private pages at load, base .bin unchanged; K3 and Qwen;");
            println!("                        currently incompatible with --memory/--save and the chat prefix cache)");
            println!("                    --temp T (0 = greedy, default)  --top-p P (nucleus, default 1.0)  --seed N");
            println!("                    --spec N (n-gram speculative decoding, greedy only)");
            println!("                    --spec-rosa N (suffix-automaton proposer, unbounded context, greedy only)");
            println!("                    --mtp (draft with the converted multi-token-prediction head, Qwen dense, greedy only)");
            println!("                    --mtp-depth N (chained draft length per verification pass, default 4;");
            println!("                        draft argmax through MICROKIMI_MTP_MINIHEAD rows, default 32768, 0 = full head)");
            println!("                    --dry P (DRY anti-repetition penalty, 0 = off)");
            println!("                    --dump-hidden (per-layer hidden-state rms table, collapse diagnostic)");
            println!("                    --logit-lens (top-5 tokens of every layer through final norm + lm_head,");
            println!("                        last prefill position; --logit-lens-all: also on each generated token)");
            println!("                    --exit-layer N (early exit: run decoder layers 0..=N only, then read the");
            println!("                        residual stream through the lens projection - final norm + lm_head,");
            println!("                        no output attn-res mix: greedy output == logit-lens row N top-1;");
            println!("                        N < n_layers-1 required; layers past N are never allocated/fetched)");
            println!("                    --lens-probe \"TOKEN\" (repeatable, with --logit-lens: adds the per-layer");
            println!("                        probability and 1-based logit rank of each probe token; the string");
            println!("                        must be exactly one vocab entry, leading space included)");
            println!("                    a .bin with an embedded seam adapter (apply_lora_bin.py --write-seam) is");
            println!("                        applied exactly after layer seam_after, in prefill and decode alike;");
            println!("                        the load line shows 'seam: adapter rank R after layer N'");
            println!("Stream & cache:");
            println!("                    --stream (lazy expert loading: RAM LRU + disk/HTTP tiers, bit-identical)");
            println!("                    --stream-ram N (expert cache budget in MB, default 512; implies --stream)");
            println!("                    --stream-disk N (remote disk cache budget in MB, default 0 = unlimited;");
            println!("                        expert-only LRU rollover, spine never evicted; env MICROKIMI_STREAM_DISK)");
            println!("                    --stream-predict N (Markov expert prefetch: N predicted experts/layer,");
            println!("                        0 = off, default; output-preserving, only changes fetch timing)");
            println!("                    --stream-fallback (DEGRADED latency mode, default OFF: on an expert cache");
            println!("                        miss, serve the resident 0.5-bit VQ1 shadow immediately and refill the");
            println!("                        full-precision expert in the background - the decode never blocks on the");
            println!("                        disk, but a shadow-served token is NOT bit-identical: a latency mode,");
            println!("                        not a quality mode; needs <model>.shadows, see `microkimi shadow`;");
            println!("                        env MICROKIMI_STREAM_FALLBACK=1 is equivalent)");
            println!("                    env MICROKIMI_DRAFTPREFETCH=0 disables the draft-aware expert prefetch");
            println!("                        (--spec/--spec-rosa + --stream: experts predicted from the drafted");
            println!("                        tokens are prefetched before the verification pass; output-preserving)");
            println!("                    env MICROKIMI_TRACE=trace.bin records the expert request stream (see cachereplay)");
            println!("                    env MICROKIMI_TRACESIM=1 cross-session trace-similarity expert prefetch");
            println!("                        (default OFF: matches the running session's routing against the per-layer");
            println!("                        expert histograms of past sessions kept in <model>.routes and prefetches");
            println!("                        the matched session's experts during the cold start / after a topic rupture;");
            println!("                        output-preserving, only changes fetch timing; see cachereplay --tracesim)");
            println!("                    env MICROKIMI_CACHE=arc|lru|lfu selects the expert cache eviction policy");
            println!("                        (default lfu; arc = T1/T2 + ghost lists, scan-resistant, non-default)");
            println!("                    env MICROKIMI_ROUTECMS=sketch.bin records a count-min sketch of the routing");
            println!("                        decisions of the run (4 x 4096 u32, saved on exit; see routestats/cmsinfo)");
            println!("  microkimi streamtest --model https://huggingface.co/org/repo [--cache-dir D] [--stream-disk N]");
            println!("                                         remote per-tensor cache + LRU budget proof (bandwidth-safe)");
            println!("  microkimi cache --info             per-repo disk cache usage (bytes, tensors, access span)");
            println!("  microkimi cache --clean [--repo X] delete cached tensors (one repo or all), prints freed bytes");
            println!("  microkimi cachereplay trace.bin [--top-k K] [--predict N] [--tracesim store.routes] [--first N]");
            println!("                                         replay a MICROKIMI_TRACE expert-request trace offline:");
            println!("                                         hit-rate vs capacity under LRU, LFU, ARC, Markov prefetch, Belady;");
            println!("                                         --tracesim: cold-start A/B of the cross-session prefetch");
            println!("  microkimi routebuild store.routes trace.bin [trace2.bin ...]");
            println!("                                         append the routing signature of each trace to a .routes store");
            println!("Memory packs:");
            println!("  microkimi decay mem.mkmem --half-life H --out mem2.mkmem [--units U]");
            println!("                                         exp2 partial forgetting: KDA states scaled by 2^(-U/H)");
            println!("  microkimi merge a.mkmem b.mkmem --alpha A --out m.mkmem");
            println!("                                         (experiment) linear blend of two states: alpha*A + (1-alpha)*B");
            println!("                                         on KDA states + conv windows; NOT a semantic merge");
            println!("  microkimi mkmem-merge A.mkmem B.mkmem [C.mkmem ...] --out AB.mkmem [--shuffle N] [--avg]");
            println!("                                         (experiment) KDA state additivity: s summed over inputs");
            println!("  microkimi pck --info [--model X.bin]   chat prefix-cache entries (count, covered tokens, bytes)");
            println!("  microkimi pck --clean [--model X.bin]  purge the chat prefix cache, prints freed bytes");
            println!("                    env MICROKIMI_NO_PCK=1 disables the chat prefix cache (default on: the state");
            println!("                        after each turn's prompt is snapshotted in <model>.pck/ and a turn whose");
            println!("                        prompt extends a cached prefix resumes from the snapshot, bit-identical;");
            println!("                        MICROKIMI_PCK_DIR overrides the cache directory; see `microkimi pck`)");
            println!("Diagnostics:");
            println!("  microkimi selftest                   compares against golden values (ref/golden.json)");
            println!("  microkimi eval --model X.bin [--vocab V.json] [--max-new N] [--ppl-file F] [--json out.json]");
            println!("                        --skip-qa --ppl-max-tokens N (NLL-only bounded window)");
            println!("                                         deterministic QA probes (40 x 2 formulations) + perplexity scorecard");
            println!("  microkimi routestats \"prompt\" [--model X.bin] [--max-new N] [--out routecms.bin]");
            println!("                                         one turn with the routing sketch armed, sketch saved on exit");
            println!("  microkimi cmsinfo sketch.bin           top-50 (layer, expert, count) + coverage curve of a sketch");
            println!("  microkimi metaltest | metaltest-packed | gputest | dstest | gpubench   Metal GPU checks (macOS only)");
        }
    }
    let _ = t0;
    // Exit explicitly: with an active Metal context + GBs of cached weight
    // buffers, graceful ObjC teardown at process exit can hang on macOS.
    // The OS reclaims everything anyway.
    if model::gpu_on() {
        std::process::exit(0);
    }
}

/// Hidden bench: matvec kernel timing on the two shapes that dominate the
/// engine (a mid projection and the full-vocab lm_head), 100 iterations.
fn dotbench_cmd() {
    // deterministic filler (splitmix64), no rand crate
    let mut state = 0x9E3779B97F4A7C15u64;
    let mut next_f32 = || {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        ((z ^ (z >> 31)) as f64 / u64::MAX as f64 - 0.5) as f32
    };
    for (rows, cols) in [(1024usize, 512usize), (163840, 1024)] {
        let w: Vec<f32> = (0..rows * cols).map(|_| next_f32()).collect();
        let x: Vec<f32> = (0..cols).map(|_| next_f32()).collect();
        let mut out = vec![0f32; rows];
        model::matvec(&w, rows, cols, &x, &mut out); // warmup
        let iters = 100;
        let t = Instant::now();
        for _ in 0..iters {
            model::matvec(&w, rows, cols, &x, &mut out);
        }
        let dt = t.elapsed().as_secs_f64() / iters as f64;
        println!(
            "matvec {}x{}: {:.4} ms/call ({:.2} GFLOP/s)  [checksum {:e}]",
            rows,
            cols,
            dt * 1000.0,
            2.0 * rows as f64 * cols as f64 / dt / 1e9,
            out.iter().map(|&v| v as f64).sum::<f64>()
        );
    }
    // MLA decode attention at real K3 dims (96 heads, latent 128+64/head, v
    // 128/head): per-head loops (cache re-streamed per head) vs MQA-style
    // all-heads kernels (cache streamed once), q8_0 cache vs f32 cache, at
    // 4k/16k/64k positions. The two q8 kernels share the exact integer dots
    // and per-head op sequence: their maxdiff must be 0 (bit-identical).
    // f32 timings are skipped past 16k (the f32 cache would not fit in RAM).
    {
        let mut cfg = crate::config::Config::microkimi();
        cfg.mla_heads = 96; // real K3 head count (micro dims otherwise)
        let (nh, hd, vd) = (cfg.mla_heads, cfg.mla_qh(), cfg.mla_v);
        let (nope, rope) = (cfg.mla_nope, cfg.mla_rope);
        let scale = (hd as f32).powf(-0.5);
        let q: Vec<f32> = (0..nh * hd).map(|_| next_f32()).collect();
        for &len in &[4096usize, 16384, 65536] {
            let pos = len - 1;
            let store_f32 = len <= 16384;
            let mut c = crate::model::MlaCache {
                k: Vec::new(),
                v: Vec::new(),
                kq: Vec::new(),
                ks: Vec::new(),
                kr: Vec::new(),
                vq: Vec::new(),
                vs: Vec::new(),
                q8: true,
                had: false,
            };
            let mut kf: Vec<f32> = Vec::new();
            let mut vf: Vec<f32> = Vec::new();
            let mut kr_row = vec![0f32; nh * hd];
            let mut vr_row = vec![0f32; nh * vd];
            let mut rope_tmp = vec![0f32; 64];
            for _j in 0..len {
                for i in 0..nh * hd {
                    kr_row[i] = next_f32();
                }
                for i in 0..nh * vd {
                    vr_row[i] = next_f32();
                }
                // engine invariant: the rope part of K is shared across heads
                rope_tmp[..rope].copy_from_slice(&kr_row[nope..nope + rope]);
                for h in 1..nh {
                    kr_row[h * hd + nope..h * hd + hd].copy_from_slice(&rope_tmp[..rope]);
                }
                c.push(&cfg, &kr_row, &vr_row);
                if store_f32 {
                    kf.extend_from_slice(&kr_row);
                    vf.extend_from_slice(&vr_row);
                }
            }
            let q8_mb = ((c.kq.len() + c.ks.len() * 4 + c.kr.len() * 4 + c.vq.len() + c.vs.len() * 4) as f64) / 1e6;
            // q8 per-head loop
            let mut attn = vec![0f32; nh * vd];
            let t = Instant::now();
            for h in 0..nh {
                let (qh, oh) = (&q[h * hd..(h + 1) * hd], &mut attn[h * vd..(h + 1) * vd]);
                crate::model::mla_attn_flash_q8(&cfg, &c, qh, h, pos, scale, oh);
            }
            let dt_q8_head = t.elapsed();
            // q8 all-heads (MQA-style, integer dot)
            let mut attn2 = vec![0f32; nh * vd];
            let t = Instant::now();
            crate::model::mla_attn_flash_q8_mqa(&cfg, &c, &q, pos, scale, &mut attn2);
            let dt_q8_mqa = t.elapsed();
            let maxdiff = attn.iter().zip(&attn2).map(|(a, b)| (*a as f64 - *b as f64).abs()).fold(0f64, f64::max);
            print!(
                "mla decode H={} pos={} (q8 cache {:.0} MB): q8 per-head {:.1} ms  q8 mqa {:.1} ms ({:.1}x)  [maxdiff {:e}]",
                nh,
                len,
                q8_mb,
                dt_q8_head.as_secs_f64() * 1000.0,
                dt_q8_mqa.as_secs_f64() * 1000.0,
                dt_q8_head.as_secs_f64() / dt_q8_mqa.as_secs_f64(),
                maxdiff
            );
            if store_f32 {
                let f32_mb = (len * nh * (hd + vd) * 4) as f64 / 1e6;
                let mut attn3 = vec![0f32; nh * vd];
                let t = Instant::now();
                crate::model::mla_attn_flash_mqa(&cfg, &kf, &vf, &q, pos, scale, &mut attn3);
                let dt_f32_mqa = t.elapsed();
                print!(
                    "  f32 mqa ({:.0} MB) {:.1} ms  q8-mqa gain {:.0}%",
                    f32_mb,
                    dt_f32_mqa.as_secs_f64() * 1000.0,
                    100.0 * (1.0 - dt_q8_mqa.as_secs_f64() / dt_f32_mqa.as_secs_f64())
                );
            }
            println!();
        }
    }
    // gemm_batch (batched prefill projections): position-major GEMM
    for (rows, cols, n) in [(512usize, 512usize, 600usize), (896, 512, 600)] {        let w: Vec<f32> = (0..rows * cols).map(|_| next_f32()).collect();
        let x: Vec<f32> = (0..n * cols).map(|_| next_f32()).collect();
        let mut out = vec![0f32; n * rows];
        model::gemm_batch(&w, rows, cols, &x, n, &mut out); // warmup
        let iters = 20;
        let t = Instant::now();
        for _ in 0..iters {
            model::gemm_batch(&w, rows, cols, &x, n, &mut out);
        }
        let dt = t.elapsed().as_secs_f64() / iters as f64;
        println!(
            "gemm {}x{}x{}: {:.4} ms/call ({:.2} GFLOP/s)  [checksum {:e}]",
            rows,
            cols,
            n,
            dt * 1000.0,
            2.0 * rows as f64 * cols as f64 * n as f64 / dt / 1e9,
            out.iter().map(|&v| v as f64).sum::<f64>()
        );
    }
    // mxfp4 quantized matvec: f32-dequant path vs integer q8 path
    for (rows, cols, nt) in [(64usize, 128usize, 1usize), (3072, 3584, 1), (163840, 1024, 10)] {
        let w: Vec<f32> = (0..rows * cols).map(|_| next_f32()).collect();
        let (p, s) = crate::quant::mxfp4::quantize(&w, rows, cols);
        drop(w);
        let x: Vec<f32> = (0..cols).map(|_| next_f32()).collect();
        let mut out = vec![0f32; rows];
        let mut line = format!("mxfp4 matvec {}x{} (nt={}):", rows, cols, nt);
        for (label, force) in [("f32", 0), ("q8", 1)] {
            crate::quant::q8::force_q8(force);
            crate::quant::mxfp4::matvec_packed(&p, &s, rows, cols, &x, &mut out, nt); // warmup
            let iters = 100;
            let t = Instant::now();
            for _ in 0..iters {
                crate::quant::mxfp4::matvec_packed(&p, &s, rows, cols, &x, &mut out, nt);
            }
            let dt = t.elapsed().as_secs_f64() / iters as f64;
            line.push_str(&format!("  {} {:.4} ms/call ({:.2} GFLOP/s)", label, dt * 1000.0, 2.0 * rows as f64 * cols as f64 / dt / 1e9));
        }
        crate::quant::q8::force_q8(-1);
        println!("{}  [checksum {:e}]", line, out.iter().map(|&v| v as f64).sum::<f64>());
    }
}

#[cfg(target_os = "macos")]
fn metaltest_cmd() {
    model::metal::metaltest();
}

#[cfg(not(target_os = "macos"))]
fn metaltest_cmd() {
    println!("metaltest is only available on macOS (Metal GPU support step 1)");
}

#[cfg(target_os = "macos")]
fn metaltest_packed_cmd() {
    model::metal::metaltest_packed();
}

#[cfg(not(target_os = "macos"))]
fn metaltest_packed_cmd() {
    println!("metaltest-packed is only available on macOS (packed mxfp4 Metal kernel)");
}

#[cfg(target_os = "macos")]
fn gputest_cmd() {
    model::metal::gputest();
}

#[cfg(not(target_os = "macos"))]
fn gputest_cmd() {
    println!("gputest is only available on macOS (Metal GPU support)");
}

#[cfg(target_os = "macos")]
fn dstest_cmd() {
    model::metal::dstest();
}

#[cfg(not(target_os = "macos"))]
fn dstest_cmd() {
    println!("dstest is only available on macOS (Metal GPU support, DeepSeek fp4 kernel)");
}

#[cfg(target_os = "macos")]
fn gpubench_cmd(args: &[String]) {
    let tl = Instant::now();
    let mp = model_flag(args).unwrap_or_else(bin_path);
    let tok = load_any_tokenizer(&mp, vocab_flag(args), crate::quant::weights::read_config(&mp).vocab);
    let adapters = adapter_flags(args);
    let mut model = if adapters.is_empty() {
        model::Model::load(&mp)
    } else {
        model::Model::load_with_adapters(&mp, &adapters)
    };
    println!("loading tokenizer + weights: {:.1?}", tl.elapsed());
    let question = "Once upon a time";
    let ids = tok.encode_chat_user(question);
    let n = 8;
    model::set_gpu(false);
    let (p_cpu, t_cpu, _) = bench_tokens(&ids, n, &tok, &mut model);
    model::set_gpu(true);
    let (p_gpu, t_gpu, ans) = bench_tokens(&ids, n, &tok, &mut model);
    println!();
    println!("gpubench - {} decode tokens on {}", n, mp);
    println!("  CPU : prefill {:.2} s, decode {:.0} ms/token ({:.1} tok/s)", p_cpu, t_cpu, 1000.0 / t_cpu);
    println!("  GPU : prefill {:.2} s, decode {:.0} ms/token ({:.1} tok/s)", p_gpu, t_gpu, 1000.0 / t_gpu);
    if t_cpu > 0.0 && t_gpu > 0.0 {
        println!("  decode speedup: {:.2}x", t_cpu / t_gpu);
    }
    println!("  answer (gpu): {}", ans);
}

/// Times prefill and per-token decode for n greedy steps. Returns
/// (prefill_s, avg_ms_per_token, decoded_answer).
#[cfg(target_os = "macos")]
fn bench_tokens(ids: &[u32], n: usize, tok: &tokenizer::AnyTokenizer, model: &mut model::Model) -> (f64, f64, String) {
    model.reset_cache();
    let t0 = Instant::now();
    let mut logits = Vec::new();
    let mut pos = 0usize;
    for &id in ids {
        logits = model.forward(id, pos);
        pos += 1;
    }
    let prefill = t0.elapsed().as_secs_f64();
    let mut times = Vec::new();
    let mut generated = Vec::new();
    for _ in 0..n {
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as u32;
        if next == tok.end_of_msg() {
            break;
        }
        let ta = Instant::now();
        logits = model.forward(next, pos);
        pos += 1;
        times.push(ta.elapsed().as_secs_f64());
        generated.push(next);
    }
    let avg = if times.is_empty() {
        0.0
    } else {
        times.iter().sum::<f64>() / times.len() as f64 * 1000.0
    };
    (prefill, avg, tok.decode(&generated))
}

#[cfg(not(target_os = "macos"))]
fn gpubench_cmd(_args: &[String]) {
    println!("gpubench is only available on macOS (Metal GPU support)");
}

#[cfg(target_os = "macos")]
fn gpu_status_line() {
    if model::gpu_on() {
        println!("GPU: Metal (matvecs >= {} elems)", model::GPU_MIN_ELEMS);
    } else {
        println!("GPU: off (CPU)");
    }
}

#[cfg(not(target_os = "macos"))]
fn gpu_status_line() {
    if model::gpu_on() {
        println!("GPU: requested via --gpu but Metal is macOS-only - using CPU");
    } else {
        println!("GPU: off (CPU)");
    }
}

#[cfg(target_os = "macos")]
fn gpu_prof_maybe_print() {
    if model::gpu_on() {
        model::metal::gpu_prof_print();
    }
}

#[cfg(not(target_os = "macos"))]
fn gpu_prof_maybe_print() {}

/// Positions of the values taken by --flag style options, so positional
/// extraction can skip them (e.g. the X in `--out X`).
fn flag_value_positions(args: &[String], names: &[&str]) -> Vec<usize> {
    let mut out = Vec::new();
    for (i, a) in args.iter().enumerate() {
        if names.contains(&a.as_str()) && i + 1 < args.len() {
            out.push(i + 1);
        }
    }
    out
}

/// Hidden debug tool behind `mkmem-div`: loads each .mkmem in turn, prefills
/// the same raw prompt on top of it, greedily decodes N tokens and prints the
/// token ids; then reports the per-position top-1 agreement of every state
/// against the first one (the reference). Quantifies how much a merged state
/// diverges from each of its parents.
fn mkmem_div_cmd(args: &[String]) {
    if !adapter_flags(args).is_empty() {
        eprintln!("error: mkmem-div cannot yet be combined with --adapter");
        std::process::exit(1);
    }
    let skip = flag_value_positions(args, &["--prompt", "--max-new", "--model", "--vocab"]);
    let paths: Vec<String> = args
        .iter()
        .enumerate()
        .skip(2)
        .filter(|(i, a)| !a.starts_with("--") && !skip.contains(i))
        .map(|(_, a)| a.clone())
        .collect();
    if paths.len() < 2 {
        eprintln!("error: mkmem-div needs at least 2 .mkmem files (reference first)");
        std::process::exit(1);
    }
    let prompt = value_flag(args, "--prompt").unwrap_or_else(|| "Once upon a time".to_string());
    let max_new: usize = value_flag(args, "--max-new").and_then(|s| s.parse().ok()).unwrap_or(20);
    let mp = model_flag(args).unwrap_or_else(bin_path);
    if crate::quant::weights::read_config(&mp).ds.is_some() {
        eprintln!("error: mkmem-div is only supported for K3 models (not DeepSeek-V4)");
        std::process::exit(1);
    }
    let tok = load_any_tokenizer(&mp, vocab_flag(args), crate::quant::weights::read_config(&mp).vocab);
    let mut model = model::Model::load(&mp);
    check_tok_compat(&tok, &model);
    let mut seqs: Vec<Vec<u32>> = Vec::new();
    for p in &paths {
        model.reset_cache();
        let init = match crate::memory::memory_pack::load(&mut model, p) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(1);
            }
        };
        let mut ids = tok.encode_raw(&prompt);
        strip_bos(&mut ids, &tok);
        let mut pos = model.cached_tokens();
        let mut logits = init;
        if !ids.is_empty() {
            logits = model.prefill(&ids, pos);
            pos += ids.len();
        }
        let mut seq = Vec::new();
        for _ in 0..max_new {
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0 as u32;
            seq.push(next);
            logits = model.forward(next, pos);
            pos += 1;
        }
        println!("{}: {:?}", p, seq);
        println!("  text: {}", tok.decode(&seq));
        seqs.push(seq);
    }
    let reference = &seqs[0];
    for (i, s) in seqs.iter().enumerate().skip(1) {
        let agree = reference.iter().zip(s.iter()).filter(|(a, b)| a == b).count();
        println!("agreement vs {}: {}/{} top-1 ({:.0}%)", paths[i], agree, reference.len(), agree as f64 / reference.len() as f64 * 100.0);
    }
}

fn model_flag(args: &[String]) -> Option<String> {    args.iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// Repeated external model adapter packs (`--adapter skill.mkap`). The packs
/// are verified and folded into private pages when the K3 model loads.
fn adapter_flags(args: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for (index, arg) in args.iter().enumerate() {
        if arg == "--adapter" {
            let value = args.get(index + 1).filter(|value| !value.starts_with("--"));
            match value {
                Some(path) => out.push(path.clone()),
                None => {
                    eprintln!("error: --adapter requires a .mkap path");
                    std::process::exit(1);
                }
            }
        }
    }
    out
}

fn vocab_flag(args: &[String]) -> Option<String> {
    args.iter()
        .position(|a| a == "--vocab")
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// Generic --flag value extractor (--memory, --save).
fn value_flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// --exit-layer N (0-based, inclusive): early exit after decoder layer N.
/// Validated against the model's layer count later (exit_layer_apply), once
/// the config is read.
fn exit_layer_flag(args: &[String]) -> Option<usize> {
    value_flag(args, "--exit-layer").map(|s| {
        s.parse().unwrap_or_else(|_| {
            eprintln!("error: --exit-layer expects a non-negative integer, got {:?}", s);
            std::process::exit(1);
        })
    })
}

/// --lens-probe "TOKEN" (repeatable): the raw strings, resolved against the
/// vocab later (lens_probes_resolve), once the tokenizer is loaded.
fn lens_probe_strings(args: &[String]) -> Vec<String> {
    args.iter()
        .enumerate()
        .filter(|(_, a)| a.as_str() == "--lens-probe")
        .filter_map(|(i, _)| args.get(i + 1).cloned())
        .collect()
}

/// Validates --exit-layer against the model config and arms the load-time
/// preset, so from_bin never allocates the KDA/MLA states of layers past the
/// exit (and a --stream run never fetches their experts).
fn exit_layer_apply(exit: Option<usize>, n_layers: usize, memory: bool, save: bool) {
    let Some(n) = exit else { return };
    if let Err(e) = model::check_exit_layer(n, n_layers) {
        eprintln!("error: {}", e);
        std::process::exit(1);
    }
    if memory || save {
        eprintln!("error: --exit-layer cannot be combined with --memory/--save (a .mkmem pack spans all layers)");
        std::process::exit(1);
    }
    model::preset_exit_layer(Some(n));
    println!("exit-layer: running layers 0..={} of {}", n, n_layers);
}

/// Resolves every --lens-probe string to a token id (exact vocab match) and
/// arms the probe columns of the logit-lens report.
fn lens_probes_resolve(probes: &[String], tok: &tokenizer::AnyTokenizer) {
    if probes.is_empty() {
        return;
    }
    let mut ids = Vec::with_capacity(probes.len());
    for p in probes {
        match model::resolve_lens_probe(tok, p) {
            Ok(id) => ids.push(id),
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(1);
            }
        }
    }
    model::set_lens_probes(ids);
}

/// --stream / --stream-ram N: MoE expert streaming with a RAM LRU budget of
/// N MB (default 512). Some(mb) when streaming is requested (--stream-ram
/// implies --stream), None for the historical full load.
fn stream_ram_flag(args: &[String]) -> Option<usize> {
    let mb = value_flag(args, "--stream-ram").and_then(|s| s.parse().ok());
    if args.iter().any(|a| a == "--stream") || mb.is_some() {
        Some(mb.unwrap_or(512))
    } else {
        None
    }
}

/// Loads a K3 model, streaming or full, and prints the fetch report at exit
/// when streaming was active.
fn load_k3_model(mp: &str, stream_mb: Option<usize>) -> model::Model {
    let pargs: Vec<String> = std::env::args().collect();
    let adapters = adapter_flags(&pargs);
    match stream_mb {
        Some(mb) => {
            // --stream-predict N: Markov expert prefetch (0 = off, default).
            // Parsed here from the process args so every streaming entry
            // point (run / chat / prefill) picks it up. top_k comes from the
            // model config (the expert batch size of one MoE layer).
            let n: usize = value_flag(&pargs, "--stream-predict").and_then(|s| s.parse().ok()).unwrap_or(0);
            if n > 0 {
                let top_k = crate::quant::weights::read_config(mp).top_k;
                crate::stream::set_predict(n, top_k);
                println!("stream: predictive prefetch enabled ({} experts/layer, top-k {})", n, top_k);
            }
            println!("stream: expert streaming enabled (RAM LRU budget {} MB)", mb);
            // --stream-fallback (or MICROKIMI_STREAM_FALLBACK=1): VQ1 shadow
            // fallback on expert cache misses. DEGRADED latency mode, NOT
            // bit-identical; needs the <model>.shadows sidecar (microkimi
            // shadow). Parsed from the process args like --stream-predict.
            let fallback = pargs.iter().any(|a| a == "--stream-fallback")
                || std::env::var("MICROKIMI_STREAM_FALLBACK").map(|v| v == "1").unwrap_or(false);
            crate::stream::set_fallback(fallback);
            if adapters.is_empty() {
                model::Model::load_streaming(mp, mb, fallback)
            } else {
                model::Model::load_streaming_with_adapters(mp, mb, fallback, &adapters)
            }
        }
        None => {
            if adapters.is_empty() {
                model::Model::load(mp)
            } else {
                model::Model::load_with_adapters(mp, &adapters)
            }
        }
    }
}

fn stream_report_maybe(stream_mb: Option<usize>) {
    if stream_mb.is_some() {
        println!("{}", crate::stream::report_line());
    }
}

/// Builds the decoding policy from --temp / --top-p / --seed / --spec /
/// --spec-rosa / --dry.
/// temp absent or 0 -> greedy (the exact historical path). With temp > 0 and
/// no --seed, the RNG is seeded from the wall clock (non reproducible).
fn sampler_flag(args: &[String]) -> model::Sampler {
    let temp: f32 = value_flag(args, "--temp").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let top_p: f32 = value_flag(args, "--top-p").and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let seed: u64 = value_flag(args, "--seed").and_then(|s| s.parse().ok()).unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E37_79B9_7F4A_7C15)
    });
    let mut s = model::Sampler::new(temp, top_p, seed);
    s.spec = value_flag(args, "--spec").and_then(|v| v.parse().ok()).unwrap_or(0);
    s.spec_rosa = value_flag(args, "--spec-rosa").and_then(|v| v.parse().ok()).unwrap_or(0);
    s.mtp = args.iter().any(|a| a == "--mtp");
    s.mtp_depth = value_flag(args, "--mtp-depth").and_then(|v| v.parse().ok()).unwrap_or(4);
    s.dry = value_flag(args, "--dry").and_then(|v| v.parse().ok()).unwrap_or(0.0);
    s
}

/// Drops the leading BOS of a freshly encoded prompt when resuming from a
/// .mkmem snapshot: BOS belongs to the stream start, already ingested when
/// the snapshot was taken. For the nano tokenizer encode_raw("") is just
/// [BOS], so an empty prompt becomes a pure continuation.
fn strip_bos(ids: &mut Vec<u32>, tok: &tokenizer::AnyTokenizer) {
    let bos = match tok {
        tokenizer::AnyTokenizer::Full(_) => Some(tokenizer::BOS),
        tokenizer::AnyTokenizer::Nano(n) => Some(n.bos),
        _ => None,
    };
    if ids.first() == bos.as_ref() {
        ids.remove(0);
    }
}

/// Loads the tokenizer matching the model: explicit --vocab, otherwise vocab_nano.json
/// next to the .bin — but ONLY when its vocab size matches the model's (a stray
/// vocab_nano.json next to microkimi-debug.bin must NOT hijack the full tokenizer),
/// otherwise the full Kimi vocabulary.
fn load_any_tokenizer(model_path: &str, vocab: Option<String>, model_vocab: usize) -> tokenizer::AnyTokenizer {
    let full = tokenizer::Tokenizer::load(&tokenizer_path());
    let nano_path = vocab.or_else(|| {
        let dir = std::path::Path::new(model_path).parent().unwrap_or(std::path::Path::new("."));
        let cand = dir.join("vocab_nano.json");
        if cand.exists() {
            Some(cand.to_string_lossy().into_owned())
        } else {
            None
        }
    });
    match nano_path {
        Some(p) => {
            // read the nano vocab size before committing to the nano tokenizer
            let nano_vs = std::fs::read(&p).ok().and_then(|b| {
                crate::json::parse(&b).get("vocab_size").and_then(|x| x.as_num()).map(|n| n as usize)
            });
            if nano_vs == Some(model_vocab) {
                println!("nano vocabulary (remap): {}", p);
                tokenizer::AnyTokenizer::Nano(tokenizer::NanoTokenizer::load(&p, full))
            } else {
                println!(
                    "warning: ignoring {} (nano vocab {} != model vocab {}) - using the full Kimi tokenizer",
                    p,
                    nano_vs.map(|v| v.to_string()).unwrap_or_else(|| "?".to_string()),
                    model_vocab
                );
                tokenizer::AnyTokenizer::Full(full)
            }
        }
        None => tokenizer::AnyTokenizer::Full(full),
    }
}

/// Fails cleanly when the tokenizer vocab does not match the model vocab
/// (either direction: nano on microkimi = [UNK] everywhere; full tokenizer on
/// nanokimi = out-of-range ids).
fn check_tok_compat(tok: &tokenizer::AnyTokenizer, model: &model::Model) {
    if tok.vocab_size() != model.cfg.vocab {
        eprintln!("error: tokenizer/model mismatch - model vocab is {}, tokenizer vocab is {}.", model.cfg.vocab, tok.vocab_size());
        eprintln!("hint: place vocab_nano.json next to the model (it ships in the GitHub release), or pass --vocab vocab_nano.json");
        std::process::exit(1);
    }
}

/// Loads tokenizer + weights, runs one inference turn with detailed output.
fn run_inference(question: &str, max_new: usize, debug: bool, model_path: &Option<String>, vocab: Option<String>, debug_routing: bool, raw: bool, memory: &Option<String>, save: &Option<String>, sampler: &mut model::Sampler, stream_mb: Option<usize>, exit: Option<usize>, probes: &[String]) -> String {
    let tl = Instant::now();
    let mp = model_path.clone().unwrap_or_else(bin_path);
    let mp_cfg = crate::quant::weights::read_config(&mp);
    // Qwen3.5-MoE model -> native tokenizer + checkpoint-backed engine.
    if mp_cfg.qwen.is_some() {
        if exit.is_some() {
            eprintln!("error: --exit-layer is only supported for K3 models (not Qwen)");
            std::process::exit(1);
        }
        if stream_mb.is_some() {
            eprintln!("error: --stream is not yet supported for Qwen models (the default mmap load is demand-paged)");
            std::process::exit(1);
        }
        if sampler.spec > 0 || sampler.spec_rosa > 0 {
            eprintln!("warning: --spec/--spec-rosa are only supported for K3 models, ignoring them (Qwen)");
            sampler.spec = 0;
            sampler.spec_rosa = 0;
        }
        if memory.is_some() && sampler.mtp {
            eprintln!("error: --mtp cannot resume from --memory (the draft cache is not part of the pairing prefix)");
            std::process::exit(1);
        }
        let tok = load_qwen_any_tokenizer(&mp, vocab, mp_cfg.vocab);
        let packs = adapter_flags(&std::env::args().collect::<Vec<_>>());
        let mut qwen = if packs.is_empty() {
            model::qwen::QwenModel::load(&mp)
        } else {
            model::qwen::QwenModel::load_with_adapters(&mp, &packs)
        };
        if qwen.has_adapter_packs() {
            println!(
                "Qwen adapter set: {}...",
                &qwen.adapter_set_sha256().unwrap()[..12]
            );
        }
        println!("loading tokenizer + weights: {:.1?}", tl.elapsed());
        println!("cores used for matvecs: {}", model::n_threads());
        gpu_status_line();
        let mut init_logits = None;
        if let Some(m) = memory {
            match crate::memory::qwen_state::load(&mut qwen, m) {
                Ok(l) => {
                    println!("memory loaded: {}", m);
                    init_logits = Some(l);
                }
                Err(e) => {
                    eprintln!("error: {}", e);
                    std::process::exit(1);
                }
            }
        }
        let (ids, stop) = if raw {
            (tok.encode_raw(question), tok.raw_stop())
        } else {
            (tok.encode_chat_user(question), tok.end_of_msg())
        };
        let answer = model::qwen::qwen_run_turn_resume(
            &ids,
            max_new,
            &tok,
            &mut qwen,
            debug,
            debug_routing,
            stop,
            init_logits,
            sampler,
        );
        if let Some(s) = save {
            match crate::memory::qwen_state::save(&qwen, s) {
                Ok(()) => println!("memory saved: {}", s),
                Err(e) => {
                    eprintln!("error: {}", e);
                    std::process::exit(1);
                }
            }
        }
        return answer;
    }
    // DeepSeek-V4 model -> dedicated tokenizer + DsModel engine.
    if mp_cfg.ds.is_some() {
        if !adapter_flags(&std::env::args().collect::<Vec<_>>()).is_empty() {
            eprintln!("error: external adapter packs are currently supported only for K3 models");
            std::process::exit(1);
        }
        if exit.is_some() {
            eprintln!("error: --exit-layer is only supported for K3 models (not DeepSeek-V4)");
            std::process::exit(1);
        }
        if stream_mb.is_some() {
            eprintln!("error: --stream is only supported for K3 models (not DeepSeek-V4)");
            std::process::exit(1);
        }
        if sampler.spec > 0 || sampler.spec_rosa > 0 {
            eprintln!("warning: --spec/--spec-rosa are only supported for K3 models, ignoring them (DeepSeek-V4)");
        }
        let tok = load_ds_any_tokenizer(&mp, vocab, mp_cfg.vocab);
        let mut model = model::deepseek::DsModel::load(&mp);
        println!("loading tokenizer + weights: {:.1?}", tl.elapsed());
        println!("cores used for matvecs: {}", model::n_threads());
        let (ids, stop) = if raw {
            (tok.encode_raw(question), tok.raw_stop())
        } else {
            (tok.encode_chat_user(question), tok.end_of_msg())
        };
        return model::deepseek::ds_run_turn(&ids, max_new, &tok, &mut model, debug, debug_routing, stop);
    }
    if !adapter_flags(&std::env::args().collect::<Vec<_>>()).is_empty()
        && (memory.is_some() || save.is_some())
    {
        eprintln!("error: --adapter cannot yet be combined with --memory or --save");
        std::process::exit(1);
    }
    exit_layer_apply(exit, mp_cfg.n_layers, memory.is_some(), save.is_some());
    let tok = load_any_tokenizer(&mp, vocab, mp_cfg.vocab);
    lens_probes_resolve(probes, &tok);
    let mut model = load_k3_model(&mp, stream_mb);
    check_tok_compat(&tok, &model);
    println!("loading tokenizer + weights: {:.1?}", tl.elapsed());
    println!("cores used for matvecs: {}", model::n_threads());
    gpu_status_line();

    let mut init_logits = None;
    if let Some(m) = memory {
        match crate::memory::memory_pack::load(&mut model, m) {
            Ok(l) => {
                println!("memory loaded: {}", m);
                init_logits = Some(l);
            }
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(1);
            }
        }
    }
    let (mut ids, stop) = if raw {
        (tok.encode_raw(question), tok.raw_stop())
    } else {
        (tok.encode_chat_user(question), tok.end_of_msg())
    };
    let answer = if init_logits.is_some() {
        strip_bos(&mut ids, &tok);
        model::run_turn_resume(&ids, max_new, &tok, &mut model, debug, debug_routing, stop, init_logits, sampler)
    } else {
        model::run_turn(&ids, max_new, &tok, &mut model, debug, debug_routing, stop, sampler)
    };
    if let Some(s) = save {
        save_memory(&model, s);
    }
    crate::stream::route_sketch::finish();
    gpu_prof_maybe_print();
    stream_report_maybe(stream_mb);
    answer
}

/// Snapshots the current state (caches + last logits) to a .mkmem file.
fn save_memory(model: &model::Model, path: &str) {
    match crate::memory::memory_pack::save(model, &model.last_logits, path) {
        Ok(()) => println!("memory saved: {}", path),
        Err(e) => {
            eprintln!("error: cannot write {}: {}", path, e);
            std::process::exit(1);
        }
    }
}

/// `microkimi prefill "text" --save mem.mkmem`: ingests the text (raw
/// completion encoding by default, chat template with --chat) and snapshots
/// the resulting state, without generating anything.
fn prefill_cmd(text: &str, save: &str, model_path: &Option<String>, vocab: Option<String>, chat: bool, stream_mb: Option<usize>) {
    let tl = Instant::now();
    let mp = model_path.clone().unwrap_or_else(bin_path);
    if crate::quant::weights::read_config(&mp).ds.is_some() {
        eprintln!("error: prefill is only supported for K3 models (not DeepSeek-V4)");
        std::process::exit(1);
    }
    if !adapter_flags(&std::env::args().collect::<Vec<_>>()).is_empty() {
        eprintln!("error: prefill/absorb state snapshots cannot yet be combined with --adapter");
        std::process::exit(1);
    }
    let tok = load_any_tokenizer(&mp, vocab, crate::quant::weights::read_config(&mp).vocab);
    let mut model = load_k3_model(&mp, stream_mb);
    check_tok_compat(&tok, &model);
    println!("loading tokenizer + weights: {:.1?}", tl.elapsed());
    let ids = if chat {
        tok.encode_chat_user(text)
    } else {
        tok.encode_raw(text)
    };
    let tp = Instant::now();
    if !ids.is_empty() {
        model.prefill(&ids, 0);
    }
    save_memory(&model, save);
    crate::stream::route_sketch::finish();
    let size = std::fs::metadata(save).map(|m| m.len()).unwrap_or(0);
    println!("prefill: {} tokens ingested in {:.1?} - state saved to {} ({:.1} KB)", ids.len(), tp.elapsed(), save, size as f64 / 1024.0);
    stream_report_maybe(stream_mb);
}

/// `microkimi routestats "prompt" [--out routecms.bin]`: runs one generation
/// turn with the count-min routing sketch armed and saves the sketch at the
/// end (same result as MICROKIMI_ROUTECMS on `run`, packaged as a command).
fn routestats_cmd(args: &[String]) {
    let skip = flag_value_positions(args, &["--out", "--model", "--vocab", "--max-new", "--adapter"]);
    let positional: Vec<&String> = args
        .iter()
        .enumerate()
        .skip(2)
        .filter(|(i, a)| !a.starts_with("--") && !skip.contains(i))
        .map(|(_, a)| a)
        .collect();
    let prompt = positional.first().map(|s| s.to_string()).unwrap_or_else(|| "Hello".to_string());
    let max_new = value_flag(args, "--max-new").and_then(|s| s.parse().ok()).unwrap_or(20);
    let out = value_flag(args, "--out").unwrap_or_else(|| "routecms.bin".to_string());
    let mp = model_flag(args);
    let mp_path = mp.clone().unwrap_or_else(bin_path);
    if crate::quant::weights::read_config(&mp_path).ds.is_some() {
        eprintln!("error: routestats is only supported for K3 models (not DeepSeek-V4)");
        std::process::exit(1);
    }
    stream::route_sketch::start(&out);
    run_inference(&prompt, max_new, false, &mp, vocab_flag(args), false, args.iter().any(|a| a == "--raw"), &None, &None, &mut sampler_flag(args), stream_ram_flag(args), exit_layer_flag(args), &[]);
    stream::route_sketch::finish();
}

/// `microkimi absorb file.txt --out pack.mkmem`: reads a document from disk
/// and ingests it exactly like `prefill` (raw completion encoding by default,
/// chat template with --chat), then snapshots the resulting KDA/MLA state as
/// a portable .mkmem pack. Resume it later with `run --memory pack.mkmem`.
fn absorb_cmd(args: &[String]) {
    let skip = flag_value_positions(args, &["--out", "--model", "--vocab", "--stream-ram", "--adapter"]);
    let positional: Vec<&String> = args
        .iter()
        .enumerate()
        .skip(2)
        .filter(|(i, a)| !a.starts_with("--") && !skip.contains(i))
        .map(|(_, a)| a)
        .collect();
    let Some(file) = positional.first() else {
        eprintln!("error: absorb requires a text file (microkimi absorb file.txt --out pack.mkmem)");
        std::process::exit(1);
    };
    let Some(out) = value_flag(args, "--out") else {
        eprintln!("error: absorb requires --out pack.mkmem");
        std::process::exit(1);
    };
    let text = match std::fs::read_to_string(file) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: cannot read {}: {}", file, e);
            std::process::exit(1);
        }
    };
    model::set_gpu(args.iter().any(|a| a == "--gpu"));
    println!("absorb: {} ({} bytes)", file, text.len());
    prefill_cmd(&text, &out, &model_flag(args), vocab_flag(args), args.iter().any(|a| a == "--chat"), stream_ram_flag(args));
}

fn chat_loop(model_path: &Option<String>, vocab: Option<String>, debug_routing: bool, raw: bool, memory: Option<String>, save: Option<String>, sampler: &mut model::Sampler, stream_mb: Option<usize>, exit: Option<usize>, probes: &[String]) {
    use std::io::Write;
    let tl = Instant::now();
    let mp = model_path.clone().unwrap_or_else(bin_path);
    let mp_cfg = crate::quant::weights::read_config(&mp);
    if mp_cfg.qwen.is_some() {
        if exit.is_some() {
            eprintln!("error: --exit-layer is only supported for K3 models (not Qwen)");
            std::process::exit(1);
        }
        if stream_mb.is_some() {
            eprintln!("error: --stream is not yet supported for Qwen models (the default mmap load is demand-paged)");
            std::process::exit(1);
        }
        if memory.is_some() || save.is_some() {
            eprintln!("error: --memory/--save state snapshots are only supported for K3 models (not Qwen)");
            std::process::exit(1);
        }
        if sampler.spec > 0 || sampler.spec_rosa > 0 {
            eprintln!("warning: --spec/--spec-rosa are only supported for K3 models, ignoring them (Qwen)");
            sampler.spec = 0;
            sampler.spec_rosa = 0;
        }
        return chat_loop_qwen(&mp, vocab, debug_routing, raw, sampler);
    }
    if mp_cfg.ds.is_some() {
        if !adapter_flags(&std::env::args().collect::<Vec<_>>()).is_empty() {
            eprintln!("error: external adapter packs are currently supported only for K3 models");
            std::process::exit(1);
        }
        if exit.is_some() {
            eprintln!("error: --exit-layer is only supported for K3 models (not DeepSeek-V4)");
            std::process::exit(1);
        }
        if stream_mb.is_some() {
            eprintln!("error: --stream is only supported for K3 models (not DeepSeek-V4)");
            std::process::exit(1);
        }
        return chat_loop_ds(&mp, vocab, debug_routing, raw);
    }
    if !adapter_flags(&std::env::args().collect::<Vec<_>>()).is_empty()
        && (memory.is_some() || save.is_some())
    {
        eprintln!("error: --adapter cannot yet be combined with --memory or --save");
        std::process::exit(1);
    }
    exit_layer_apply(exit, crate::quant::weights::read_config(&mp).n_layers, memory.is_some(), save.is_some());
    let tok = load_any_tokenizer(&mp, vocab, crate::quant::weights::read_config(&mp).vocab);
    lens_probes_resolve(probes, &tok);
    let mut model = load_k3_model(&mp, stream_mb);
    check_tok_compat(&tok, &model);
    println!("loading tokenizer + weights: {:.1?}", tl.elapsed());
    gpu_status_line();
    // --memory: resume a .mkmem snapshot; turns are then fed incrementally on
    // top of the loaded state (the history lives in the caches, no re-prefill).
    let mut init_logits = None;
    if let Some(m) = &memory {
        match crate::memory::memory_pack::load(&mut model, m) {
            Ok(l) => {
                println!("memory loaded: {}", m);
                init_logits = Some(l);
            }
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(1);
            }
        }
    }
    let resumed = memory.is_some();
    // prefix cache (default on for K3 chat; MICROKIMI_NO_PCK=1 disables):
    // after each turn the state after the full prompt is snapshotted into
    // <model>.pck/, and a turn whose prompt extends a cached prefix resumes
    // from the snapshot instead of re-prefilling. Bit-identity with a full
    // prefill requires the sequential KDA loop (the chunked form reassociates
    // per 64-token chunk, so a moved chunk boundary would break it). Off in
    // raw mode, with --memory (state comes from elsewhere), with an active
    // speculative proposer, and with --exit-layer (the mkmem payload spans
    // all layers, the truncated model holds only N+1 caches).
    let pck = if raw
        || resumed
        || sampler.spec > 0
        || sampler.spec_rosa > 0
        || exit.is_some()
        || model.has_adapter_packs()
    {
        if let Some(set) = model.adapter_set_sha256().filter(|_| !raw && !resumed) {
            println!("pck: disabled with external adapter set {}...", &set[..12]);
        }
        None
    } else {
        let p = memory::prefix_cache::open(&mp);
        if p.is_some() {
            crate::model::kda_chunk::force_sequential();
        }
        p
    };
    if raw {
        println!("\nRAW interactive mode - each line is a story beginning to continue (type 'quit' to exit)");
    } else {
        println!("\nInteractive mode - history kept (type 'quit' to exit)");
    }
    let stdin = std::io::stdin();
    let mut history: Vec<(String, String)> = Vec::new(); // (question, answer)
    loop {
        print!("\nYou > ");
        std::io::stdout().flush().unwrap();
        let mut line = String::new();
        if stdin.read_line(&mut line).unwrap() == 0 {
            break;
        }
        let q = line.trim();
        if q.eq_ignore_ascii_case("quit") || q.eq_ignore_ascii_case("exit") {
            break;
        }
        if q.is_empty() {
            continue;
        }
        if raw {
            // raw completion: BOS + text, stop on EOS, each turn independent
            // (with --memory: one continuous stream, BOS stripped after the snapshot)
            let mut ids = tok.encode_raw(q);
            let stop = tok.raw_stop();
            if resumed {
                strip_bos(&mut ids, &tok);
                model::run_turn_resume(&ids, 200, &tok, &mut model, false, debug_routing, stop, init_logits.take(), sampler);
            } else {
                model::run_turn(&ids, 200, &tok, &mut model, false, debug_routing, stop, sampler);
            }
        } else if resumed {
            let ids = tok.encode_chat(&[], q);
            let answer = model::run_turn_resume(&ids, 200, &tok, &mut model, false, debug_routing, tok.end_of_msg(), init_logits.take(), sampler);
            history.push((q.to_string(), answer));
        } else {
            let ids = tok.encode_chat(&history, q);
            let answer = memory::prefix_cache::run_turn_chat(pck.as_ref(), &ids, 200, &tok, &mut model, debug_routing, tok.end_of_msg(), sampler);
            history.push((q.to_string(), answer));
        }
    }
    if let Some(s) = &save {
        save_memory(&model, s);
    }
    crate::stream::route_sketch::finish();
    stream_report_maybe(stream_mb);
}

/// Interactive loop for Qwen3.5-MoE text models.
fn chat_loop_qwen(
    mp: &str,
    vocab: Option<String>,
    debug_routing: bool,
    raw: bool,
    sampler: &mut model::Sampler,
) {
    use std::io::Write;
    let tl = Instant::now();
    let cfg = crate::quant::weights::read_config(mp);
    let tok = load_qwen_any_tokenizer(mp, vocab, cfg.vocab);
    let packs = adapter_flags(&std::env::args().collect::<Vec<_>>());
    let mut qwen = if packs.is_empty() {
        model::qwen::QwenModel::load(mp)
    } else {
        model::qwen::QwenModel::load_with_adapters(mp, &packs)
    };
    if qwen.has_adapter_packs() {
        println!(
            "Qwen adapter set: {}...",
            &qwen.adapter_set_sha256().unwrap()[..12]
        );
    }
    println!("loading tokenizer + weights: {:.1?}", tl.elapsed());
    println!("cores used for matvecs: {}", model::n_threads());
    gpu_status_line();
    if raw {
        println!("\nRAW interactive mode - each line is an independent completion (type 'quit' to exit)");
    } else {
        println!("\nInteractive mode - history kept (type 'quit' to exit)");
    }
    // chat prefix cache: MKMEMQW1 images keyed by token prefix, so a turn
    // whose prompt extends a previous turn (or a past session) resumes
    // from the snapshot instead of re-ingesting the whole history
    let pck = memory::prefix_cache::open(mp);
    let stdin = std::io::stdin();
    let mut history: Vec<(String, String)> = Vec::new();
    loop {
        print!("\nYou > ");
        std::io::stdout().flush().unwrap();
        let mut line = String::new();
        if stdin.read_line(&mut line).unwrap() == 0 {
            break;
        }
        let question = line.trim();
        if question.eq_ignore_ascii_case("quit") || question.eq_ignore_ascii_case("exit") {
            break;
        }
        if question.is_empty() {
            continue;
        }
        let (ids, stop) = if raw {
            (tok.encode_raw(question), tok.raw_stop())
        } else {
            (tok.encode_chat(&history, question), tok.end_of_msg())
        };
        let answer = memory::prefix_cache::qwen_run_turn_chat(
            pck.as_ref(),
            &ids,
            200,
            &tok,
            &mut qwen,
            debug_routing,
            stop,
            sampler,
        );
        if !raw {
            history.push((question.to_string(), answer));
        }
    }
}

/// Loads the Qwen tokenizer copied beside the converted model, or an
/// explicitly supplied tokenizer.json.
fn load_qwen_any_tokenizer(
    model_path: &str,
    vocab: Option<String>,
    model_vocab: usize,
) -> tokenizer::AnyTokenizer {
    let path = if let Some(path) = vocab {
        path
    } else {
        let dir = std::path::Path::new(model_path)
            .parent()
            .unwrap_or(std::path::Path::new("."));
        ["qwen.tokenizer.json", "tokenizer.json"]
            .iter()
            .map(|name| dir.join(name))
            .find(|path| path.exists())
            .map(|path| path.to_string_lossy().into_owned())
            .unwrap_or_else(|| {
                eprintln!("error: no Qwen tokenizer found beside {}", model_path);
                eprintln!("hint: pass --vocab tokenizer.json or keep qwen.tokenizer.json beside the converted model");
                std::process::exit(1);
            })
    };
    println!("Qwen tokenizer: {}", path);
    // A vocabulary-sliced model keeps qwen.vocabmap.json beside it: the
    // tokenizer then loads the full vocabulary and remaps its output.
    let map_path = std::path::Path::new(model_path)
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join("qwen.vocabmap.json");
    if map_path.exists() {
        let bytes = std::fs::read(&map_path).unwrap();
        let j = json::parse_complete(&bytes);
        let sliced = j.get("vocab_size").and_then(|x| x.as_num()).map(|n| n as usize);
        let full = j.get("full_vocab_size").and_then(|x| x.as_num()).map(|n| n as usize);
        let new_to_old: Vec<u32> = j
            .get("new_to_old")
            .and_then(|x| x.as_arr())
            .map(|a| a.iter().filter_map(|v| v.as_num().map(|n| n as u32)).collect())
            .unwrap_or_default();
        if sliced != Some(model_vocab) {
            eprintln!(
                "error: {} was built for vocab {:?}, the model has {}",
                map_path.display(),
                sliced,
                model_vocab
            );
            std::process::exit(1);
        }
        let mut tok = model::qwentok::QwenTokenizer::load(&path, full.unwrap_or(248_320));
        if let Err(e) = tok.attach_remap(new_to_old, model_vocab) {
            eprintln!("error: {}: {}", map_path.display(), e);
            std::process::exit(1);
        }
        println!("Qwen vocab map: {} ({} kept rows)", map_path.display(), model_vocab);
        return tokenizer::AnyTokenizer::Qwen(tok);
    }
    tokenizer::AnyTokenizer::Qwen(model::qwentok::QwenTokenizer::load(
        &path,
        model_vocab,
    ))
}

/// Interactive loop for DeepSeek-V4 models (DsTokenizer + DsModel).
fn chat_loop_ds(mp: &str, vocab: Option<String>, debug_routing: bool, raw: bool) {
    use std::io::Write;
    let tl = Instant::now();
    let tok = load_ds_any_tokenizer(mp, vocab, crate::quant::weights::read_config(mp).vocab);
    let mut model = model::deepseek::DsModel::load(mp);
    println!("loading tokenizer + weights: {:.1?}", tl.elapsed());
    if raw {
        println!("\nRAW interactive mode - each line is a story beginning to continue (type 'quit' to exit)");
    } else {
        println!("\nInteractive mode - history kept (type 'quit' to exit)");
    }
    let stdin = std::io::stdin();
    let mut history: Vec<(String, String)> = Vec::new();
    loop {
        print!("\nYou > ");
        std::io::stdout().flush().unwrap();
        let mut line = String::new();
        if stdin.read_line(&mut line).unwrap() == 0 {
            break;
        }
        let q = line.trim();
        if q.eq_ignore_ascii_case("quit") || q.eq_ignore_ascii_case("exit") {
            break;
        }
        if q.is_empty() {
            continue;
        }
        if raw {
            let ids = tok.encode_raw(q);
            let stop = tok.raw_stop();
            model::deepseek::ds_run_turn(&ids, 200, &tok, &mut model, false, debug_routing, stop);
        } else {
            let ids = tok.encode_chat(&history, q);
            let answer = model::deepseek::ds_run_turn(&ids, 200, &tok, &mut model, false, debug_routing, tok.end_of_msg());
            history.push((q.to_string(), answer));
        }
    }
}

/// Loads the tokenizer for a DeepSeek-V4 model: explicit --vocab (a full
/// tokenizer.json OR a vocab_ds_nano.json remap), then vocab_ds_nano.json next
/// to the bin when its vocab size matches the model's, otherwise the full V4
/// tokenizer (tokenizer.json next to the bin / cache / HF download).
fn load_ds_any_tokenizer(mp: &str, vocab: Option<String>, model_vocab: usize) -> tokenizer::AnyTokenizer {
    let full = || model::dstok::DsTokenizer::load(&ds_tokenizer_path(mp, None));
    let try_nano = |p: &str| -> Option<tokenizer::AnyTokenizer> {
        let bytes = std::fs::read(p).ok()?;
        let j = crate::json::parse(&bytes);
        j.get("nano_to_ds")?;
        let vs = j.get("vocab_size").and_then(|x| x.as_num()).map(|n| n as usize);
        if vs != Some(model_vocab) {
            eprintln!(
                "warning: ignoring {} (nano vocab {} != model vocab {}) - using the full V4 tokenizer",
                p,
                vs.map(|v| v.to_string()).unwrap_or_else(|| "?".to_string()),
                model_vocab
            );
            return None;
        }
        println!("DS nano vocabulary (remap): {}", p);
        Some(tokenizer::AnyTokenizer::DsNano(tokenizer::DsNanoTokenizer::load(p, full())))
    };
    if let Some(p) = &vocab {
        // explicit --vocab: nano remap or a plain tokenizer.json
        if let Some(t) = try_nano(p) {
            return t;
        }
        return tokenizer::AnyTokenizer::Ds(model::dstok::DsTokenizer::load(p));
    }
    let dir = std::path::Path::new(mp).parent().unwrap_or(std::path::Path::new("."));
    let cand = dir.join("vocab_ds_nano.json");
    if cand.exists() {
        if let Some(t) = try_nano(&cand.to_string_lossy()) {
            return t;
        }
    }
    tokenizer::AnyTokenizer::Ds(full())
}

/// Locates the DeepSeek-V4 tokenizer.json: explicit --vocab, then next to the
/// model (written by `build-ds`), then the local cache, then downloaded.
pub fn ds_tokenizer_path(model_path: &str, vocab: Option<String>) -> String {
    if let Some(v) = vocab {
        return v;
    }
    let dir = std::path::Path::new(model_path).parent().unwrap_or(std::path::Path::new("."));
    let cand = dir.join("microdeepseek.tokenizer.json");
    if cand.exists() {
        return cand.to_string_lossy().into_owned();
    }
    let cache = format!("{}/.cache/microkimi", std::env::var("HOME").unwrap_or_default());
    let dst = format!("{}/microdeepseek.tokenizer.json", cache);
    if std::path::Path::new(&dst).exists() {
        return dst;
    }
    std::fs::create_dir_all(&cache).ok();
    println!("downloading tokenizer.json from huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731 …");
    let data = crate::stream::http::fetch(
        "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/resolve/main/tokenizer.json",
    )
    .expect("failed to download the V4 tokenizer.json (no local file found)");
    std::fs::write(&dst, data).unwrap();
    dst
}

pub fn bin_path() -> String {    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_default();
    // Default model: nanokimi-0.2b (the pretrained demo shipped in the GitHub
    // release), then microkimi-debug (the 93-layer architecture demo from
    // `build`). Legacy file names are kept as fallbacks.
    for name in ["nanokimi-0.2b.bin", "nanokimi.bin", "microkimi-debug.bin", "microkimi.bin"] {
        let candidates = [
            std::path::PathBuf::from(name),
            std::path::PathBuf::from(format!("models/{}", name)),
            exe_dir.join(name),
            exe_dir.join(format!("../../{}", name)),
        ];
        for c in &candidates {
            if c.exists() {
                return c.to_string_lossy().into_owned();
            }
        }
    }
    eprintln!("error: no model found.");
    eprintln!("  download nanokimi-0.2b.bin + vocab_nano.json from the GitHub Releases page into the repo root,");
    eprintln!("  or run 'microkimi build' to assemble microkimi-debug.bin (93-layer architecture demo).");
    std::process::exit(1);
}

pub fn tokenizer_path() -> String {
    for c in ["ref/tiktoken.model", "ref/tiktoken.model"] {
        if std::path::Path::new(c).exists() {
            return c.to_string();
        }
    }
    // tiktoken.model is not vendored (Moonshot license): local cache first, then
    // download from huggingface.co/moonshotai/Kimi-K3 if missing.
    let cache = format!("{}/.cache/microkimi", std::env::var("HOME").unwrap_or_default());
    let dst = format!("{}/tiktoken.model", cache);
    if std::path::Path::new(&dst).exists() {
        return dst;
    }
    std::fs::create_dir_all(&cache).ok();
    println!("downloading tiktoken.model from huggingface.co/moonshotai/Kimi-K3 …");
    let data = crate::stream::http::fetch(
        "https://huggingface.co/moonshotai/Kimi-K3/resolve/main/tiktoken.model",
    )
    .expect("failed to download tiktoken.model (no local file found)");
    std::fs::write(&dst, data).unwrap();
    dst
}
