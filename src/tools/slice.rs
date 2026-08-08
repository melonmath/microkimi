// `microkimi slice`: structural pruning of a K3 .bin model (MKIM0001/0002 in,
// MKIM0002 out). The output loads through the unmodified Model::load: every
// pruned dim is recorded in the MKIM0002 JSON config (hidden, n_experts,
// top_k, n_layers, mla_layers, dense_layers).
//
//   microkimi slice --model X.bin --out Y.bin [--hidden N] [--experts N] [--layers "spec"] [--cold-vq N]
//                                                [--vocab-top N <freqfile> [--vocab-base <remap.json>]]
//                                                [--imatrix imatrix.bin [--imatrix-score-only]]
//                                                [--expert-order=frequency --route-cms sketch.bin]
//
// Expert reordering (--expert-order=frequency --route-cms SKETCH): no
// pruning, ALL experts stay; per MoE layer the expert blobs are physically
// rewritten in descending routing-frequency order (the count-min sketch
// recorded with MICROKIMI_ROUTECMS, stream/route_sketch.rs), hottest first, densely packed
// (64-byte alignment). The router gate rows and bias are permuted with the
// same order, so expert ids are simply relabeled and the model is
// mathematically unchanged: any engine reads the reordered .bin, and old
// .bins keep working (the permutation rides in the MKIM0002 config as an
// "expert_order" index table, new_id -> old_id). The point is physical: hot
// experts become file-adjacent, so the stream engine's contiguous-run
// fusion (stream.rs warm_batch) serves a layer's top-k batch in far fewer
// physical reads on latency-bound disks. Combined with --experts N, the
// Frobenius keep-set membership is unchanged and only the file order
// follows the frequency.
//
// Vocabulary pruning (--vocab-top N freqfile): keeps the N most frequent
// token rows of embed_tokens.weight / lm_head.weight plus ALL special tokens
// (they have near-zero corpus frequency but are structural: <|open|>, <|sep|>,
// <|close|>, <|end_of_msg|>, UNK, PAD...). Detection is conservative: every id
// of the source config "specials" block is kept, and on a full Kimi vocab
// (vocab > 163584) the whole reserved block [163584, vocab) is kept. The
// freqfile ids index the model's CURRENT vocabulary: text format is
// "<token_id> <count>" per line ('#' comments, blank lines ok); a JSON object
// {"<id>": <count>, ...} is also accepted. nano/count_freq.py builds one from
// a tokenized corpus (u32/u16 binary + .meta.json sidecar). The output config
// carries the new (smaller) vocab size, and a runtime remap compatible with
// the engine's --vocab mechanism is written next to the .bin as
// <stem>.vocab.json (new_id -> kimi id via "nano_to_kimi"; dropped tokens
// encode as UNK). When the source model is itself remapped (e.g. nano vocab),
// the remap is composed through the base table: --vocab-base, else
// vocab_nano.json next to the source with a matching vocab_size, else (full
// Kimi vocab only) the identity.
//
// Precision tiering (--cold-vq N): no structural pruning, ALL experts stay
// (router untouched); per MoE layer the top-N experts by Frobenius score
// stay mxfp4 (hot) and the rest are requantized to VQ1 (cold): vectors of 16
// consecutive values mapped to the nearest of 256 entries of ONE global
// codebook (tensor "vq_codebook" [256,16] f32, 16 KB), 1 byte per vector =
// 0.5 bit/weight vs 4.25 for mxfp4. The codebook is Lloyd k-means (seeded,
// deterministic) over a reservoir sample of all cold-expert dequantized
// values, raw (unnormalized) vectors. With --experts M (M >= N): the tail
// below M is still pruned, top-N hot, ranks N..M cold VQ1.
// With --imatrix FILE (from `microkimi calibrate`): activation second
// moments weight both the k-means (distance + per-dimension means) and the
// nearest-centroid assignment, so weight columns feeding large activations
// keep more fidelity; the written file format is unchanged.
// --imatrix-score-only loads the stats only to REPORT the activation-weighted
// error of the blind codebook (A/B measurement).
//
// Ranking (v1, weight magnitude only, no activation calibration):
//   - channels (--hidden): score[c] = sum of |w| over every tensor touching
//     hidden channel c (column sums of input projections, embeddings and
//     lm_head; row sums of output projections; |w[c]| of the [d] norm
//     vectors). The top-N channels are kept, the SAME indices everywhere.
//   - experts (--experts): per MoE layer, score[e] = squared Frobenius norm
//     of the expert's w1+w2+w3 (dequantized). Top-N kept per layer, the 2
//     shared experts always stay, the router is re-indexed (rows sliced) and
//     top_k becomes min(top_k, N).
//   - layers (--layers): "0-11", "0,10,20" or mixes; tensors are renumbered
//     layers.* in keep order and the config records which kept layers are
//     MLA / dense. AttnRes choice: the block structure is NOT carried over
//     (blocks are an inference-time grouping, not weights); attn_res_block
//     keeps its value and is re-applied on the RENUMBERED layers, exactly as
//     if the pruned model had been built with that block size. The per-layer
//     res norm/proj vectors simply follow their layer.
//
// MXFP4 decision: kept experts are copied byte-for-byte in mxfp4 (their dims
// [moe_inter, routed_hidden] / [routed_hidden, moe_inter] never change: the
// latent MoE sits behind routed_hidden, so channel pruning does not touch
// them). Zero requantization loss. All sliced tensors are f32 and stay f32.

mod ckpt;
mod plan;
mod score;
mod source;
mod vocab;
mod vq;

pub(crate) use source::{split_layer, DirEntry};
use ckpt::{fnv1a, join_csv, SliceCkpt};
use plan::{expert_plan_key, slice_f32, slice_f32_rows, slice_vocab_rows, Plan};
use score::{channel_scores, expert_keep_sets, expert_score_map, top_n, ScoreCache};
use source::{n_rows, role_of, row_chunks, row_width, Role, Source};
use vq::{vq_quantize_tensor, vq_reservoir};
use vocab::{build_vocab_plan, VocabPlan};

use crate::quant::weights::{blob_size, f32_to_bytes, BinWriter, DTYPE_F32, DTYPE_VQ1};

fn value_flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

/// Parses a layer spec: "0-11", "0,10,20", "0-3,7,9-11" (ranges inclusive).
fn parse_layer_spec(spec: &str, n_layers: usize) -> Vec<usize> {
    let mut keep = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        assert!(!part.is_empty(), "bad --layers spec: '{}'", spec);
        if let Some((a, b)) = part.split_once('-') {
            let a: usize = a.trim().parse().unwrap_or_else(|_| panic!("bad --layers spec: '{}'", spec));
            let b: usize = b.trim().parse().unwrap_or_else(|_| panic!("bad --layers spec: '{}'", spec));
            assert!(a <= b, "bad --layers range '{}'", part);
            keep.extend(a..=b);
        } else {
            keep.push(part.parse().unwrap_or_else(|_| panic!("bad --layers spec: '{}'", spec)));
        }
    }
    keep.sort_unstable();
    keep.dedup();
    assert!(!keep.is_empty(), "--layers keeps nothing");
    assert!(keep.last().unwrap() < &n_layers, "--layers index out of range (model has {} layers)", n_layers);
    keep
}

pub fn run(args: &[String]) {
    let t0 = std::time::Instant::now();
    let Some(model) = value_flag(args, "--model") else {
        eprintln!("error: slice requires --model X.bin | model.safetensors | dir/ | https://huggingface.co/org/repo");
        std::process::exit(1);
    };
    let Some(out) = value_flag(args, "--out") else {
        eprintln!("error: slice requires --out Y.bin");
        std::process::exit(1);
    };
    let hidden: Option<usize> = value_flag(args, "--hidden").map(|s| s.parse().expect("bad --hidden"));
    let experts: Option<usize> = value_flag(args, "--experts").map(|s| s.parse().expect("bad --experts"));
    let layers_spec = value_flag(args, "--layers");
    // --cold-vq N: precision-tiered expert storage. ALL experts stay in the
    // file (the router is untouched); per MoE layer the top-N experts by
    // Frobenius score stay mxfp4 (hot), the rest are requantized to VQ1
    // (cold, ~0.5 bit/weight, one global 256x16 codebook). Combined with
    // --experts M (M >= N): only the top-M experts are kept, top-N hot.
    let cold_vq: Option<usize> = value_flag(args, "--cold-vq").map(|s| s.parse().expect("bad --cold-vq"));
    // --vocab-top N <freqfile>: vocabulary pruning (see the header comment).
    // N rows by frequency + every special token; the remap rides next to the
    // .bin as <stem>.vocab.json (engine --vocab compatible).
    let vocab_top: Option<(usize, String)> = args.iter().position(|a| a == "--vocab-top").map(|i| {
        let n: usize = args.get(i + 1).and_then(|s| s.parse().ok()).expect("--vocab-top needs N (rows to keep) and a freqfile path");
        let f = args.get(i + 2).cloned().expect("--vocab-top needs a freqfile path after N");
        (n, f)
    });
    let vocab_base = value_flag(args, "--vocab-base");
    // --expert-order=frequency --route-cms SKETCH: physical reorder of the
    // expert blobs of every MoE layer by descending routing frequency (the
    // count-min sketch recorded with MICROKIMI_ROUTECMS, stream/route_sketch.rs), hottest
    // expert first. The router gate rows and bias are permuted with the same
    // order, so expert ids are simply relabeled: the model is mathematically
    // unchanged, any engine reads the reordered .bin (old engines included)
    // and old .bins keep working. The point is physical: hot experts become
    // file-adjacent, so the stream engine's contiguous-run fusion (stream.rs
    // warm_batch) serves the top-k batch of a layer in far fewer physical
    // reads. The permutation rides in the MKIM0002 config as an index table
    // ("expert_order"). Combined with --experts N the keep-set membership
    // still comes from the Frobenius scores, only the file order changes.
    let kv_flag = |name: &str| {
        value_flag(args, name).or_else(|| args.iter().find_map(|a| a.strip_prefix(&format!("{}=", name)).map(|s| s.to_string())))
    };
    let expert_order_flag = kv_flag("--expert-order");
    let route_cms_path = kv_flag("--route-cms");
    if let Some(o) = &expert_order_flag {
        if o != "frequency" {
            eprintln!("error: --expert-order supports only 'frequency' (got '{}')", o);
            std::process::exit(1);
        }
        if route_cms_path.is_none() {
            eprintln!("error: --expert-order=frequency requires --route-cms SKETCH (record one with MICROKIMI_ROUTECMS=SKETCH)");
            std::process::exit(1);
        }
    }
    if hidden.is_none() && experts.is_none() && layers_spec.is_none() && cold_vq.is_none() && vocab_top.is_none() && expert_order_flag.is_none() {
        eprintln!("error: slice needs at least one of --hidden / --experts / --layers / --cold-vq / --vocab-top / --expert-order");
        std::process::exit(1);
    }
    if let (Some(m), Some(n)) = (experts, cold_vq) {
        assert!(n <= m, "--cold-vq N must be <= --experts M (hot experts are a subset of the kept ones)");
    }
    if cold_vq.is_some() {
        assert!(
            !model.starts_with("http://") && !model.starts_with("https://"),
            "--cold-vq requires a local .bin source (mxfp4 expert blobs)"
        );
    }

    let mut source = Source::open(&model, &out);
    if cold_vq.is_some() {
        assert!(matches!(source, Source::Bin(_)), "--cold-vq requires a .bin source (mxfp4 expert blobs)");
    }
    // --imatrix FILE: activation importance stats (microkimi calibrate) used
    // to weight the VQ codebook training + assignment of the cold experts.
    // --imatrix-score-only: load the same stats but only to REPORT the
    // activation-weighted error of the blind codebook (A/B measurement).
    let imatrix_score_only = args.iter().any(|a| a == "--imatrix-score-only");
    let imatrix: Option<crate::quant::imatrix::Imatrix> = match value_flag(args, "--imatrix") {
        Some(p) => {
            assert!(cold_vq.is_some(), "--imatrix only applies to --cold-vq");
            let im = crate::quant::imatrix::load(&p).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                std::process::exit(1);
            });
            let cfg0 = source.config();
            assert_eq!(im.routed_hidden, cfg0.routed_hidden, "imatrix routed_hidden {} != model {}", im.routed_hidden, cfg0.routed_hidden);
            assert_eq!(im.moe_inter, cfg0.moe_inter, "imatrix moe_inter {} != model {}", im.moe_inter, cfg0.moe_inter);
            println!(
                "imatrix: {} ({} tokens, {} MoE layers){}",
                p,
                im.tokens,
                im.layers.len(),
                if imatrix_score_only { " [score only: blind codebook, weighted error report]" } else { "" }
            );
            Some(im)
        }
        None => None,
    };

    // ── 1. layer selection (then resolve tensor shapes/byte sources) ──
    let kept_layers = match &layers_spec {
        Some(s) => parse_layer_spec(s, source.config().n_layers),
        None => (0..source.config().n_layers).collect(),
    };
    let new_layer_of = |old: usize| kept_layers.iter().position(|&l| l == old);
    source.resolve(&kept_layers);
    // with --hidden the non-expert tensors are read twice (scoring, writing):
    // cache the converted bytes on disk so remote bytes are fetched once
    if hidden.is_some() {
        source.enable_caching();
    }
    let arch = source.arch();
    let cfg = source.config();
    let d = cfg.d;
    println!(
        "slice: {} ({} layers, hidden {}, {} experts top-{} + {} shared, vocab {})",
        model, cfg.n_layers, d, cfg.n_experts, cfg.top_k, cfg.n_shared, cfg.vocab
    );
    println!("layers: keeping {}/{} {:?}", kept_layers.len(), cfg.n_layers, kept_layers);

    // physical expert order (--expert-order=frequency): per kept MoE layer
    // the old expert ids in write order, hottest first by count-min estimate
    // (ties and never-recorded experts id-ascending, deterministic). Computed
    // before the checkpoint key: the sketch identity is part of the key.
    let mut eorder_key = String::new();
    let expert_order: Option<std::collections::HashMap<usize, Vec<usize>>> = expert_order_flag.as_ref().map(|_| {
        let path = route_cms_path.as_deref().unwrap();
        let sketch = crate::stream::route_sketch::Cms::load(path).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });
        eorder_key = format!("frequency:{}:{}", path, sketch.total());
        let mut m = std::collections::HashMap::new();
        for &l in &kept_layers {
            if !cfg.is_moe(l) {
                continue;
            }
            let mut ids: Vec<usize> = (0..cfg.n_experts).collect();
            ids.sort_by_key(|&e| (std::cmp::Reverse(sketch.estimate(l as u32, e as u32)), e));
            m.insert(l, ids);
        }
        println!(
            "expert-order: frequency from {} ({} routing decisions, {} MoE layers reordered, hottest first)",
            path,
            sketch.total(),
            m.len()
        );
        m
    });

    // crash-safe resume checkpoint: model + kept layers + pruning params
    // (vocab-top is part of the key: N and the freqfile content hash)
    let vocabtop_key = vocab_top
        .as_ref()
        .map(|(n, f)| {
            let text = std::fs::read_to_string(f).unwrap_or_else(|e| panic!("freqfile {} unreadable: {}", f, e));
            format!("{}:{:016x}", n, fnv1a(&text))
        })
        .unwrap_or_default();
    let ckpt_key = format!(
        "model={}|layers={}|hidden={}|experts={}|vocabtop={}|eorder={}",
        model,
        join_csv(&kept_layers),
        hidden.map(|h| h.to_string()).unwrap_or_default(),
        experts.map(|e| e.to_string()).unwrap_or_default(),
        vocabtop_key,
        eorder_key
    );
    let ckpt = SliceCkpt::open(&out, &ckpt_key);

    // ── 2. channel selection (scored on the kept layers only) ──
    let channels: Option<Vec<usize>> = hidden.map(|h| {
        assert!(h > 0 && h <= d, "--hidden must be in 1..={}", d);
        if let Some(ch) = &ckpt.channels {
            println!("hidden: {}/{} channels restored from checkpoint", ch.len(), d);
            return ch.clone();
        }
        let n_scored = source
            .entries()
            .iter()
            .filter(|e| {
                split_layer(&e.name).map(|(l, _)| kept_layers.contains(&l)).unwrap_or(true)
                    && !matches!(role_of(&e.name, cfg, arch), Role::Copy | Role::RouterB | Role::Expert)
            })
            .count();
        println!("hidden: scoring channels over {} tensors...", n_scored);
        let scores = channel_scores(&source, &kept_layers, d);
        let keep = top_n(&scores, h);
        println!("hidden: keeping {}/{} channels (top-|w|), score range {:.3} .. {:.3}", h, d,
            keep.iter().map(|&i| scores[i]).fold(f64::INFINITY, f64::min),
            keep.iter().map(|&i| scores[i]).fold(f64::NEG_INFINITY, f64::max));
        ckpt.record_channels(&keep);
        keep
    });

    // ── 3. expert selection (per kept MoE layer) ──
    let expert_sets = experts.map(|n| {
        assert!(n > 0, "--experts must be >= 1");
        let t = std::time::Instant::now();
        // persistent full-score cache (config-independent, one level below
        // the per-run .sliceckpt): saves the whole scoring on reruns
        let score_cache = ScoreCache::open(&out, &model, cfg.n_layers, cfg.n_experts);
        let sets = expert_keep_sets(&source, &kept_layers, n, &ckpt, &score_cache);
        let how = if matches!(source, Source::Bin(_)) {
            "Frobenius of dequantized w1+w2+w3"
        } else {
            "scale-energy of w1+w2+w3 (weight_scale tensors only, 1/17 of the bytes)"
        };
        println!("experts: keeping {}/{} per MoE layer ({}), scored in {:.1?}", n, cfg.n_experts, how, t.elapsed());
        sets
    });

    // fold --expert-order into the per-layer expert lists the plan builder
    // uses: the list order IS the new expert id order (Expert rename and
    // RouterW/RouterB row gather both follow it). With --experts N the
    // Frobenius keep-set membership is preserved, only the order changes.
    let expert_sets = match &expert_order {
        None => expert_sets,
        Some(order) => Some(
            order
                .iter()
                .map(|(&l, ids)| {
                    let ids: Vec<usize> = match &expert_sets {
                        Some(sets) => {
                            let keep: std::collections::HashSet<usize> = sets[&l].iter().copied().collect();
                            ids.iter().copied().filter(|e| keep.contains(e)).collect()
                        }
                        None => ids.clone(),
                    };
                    (l, ids)
                })
                .collect(),
        ),
    };

    // ── 3b. precision tiering (--cold-vq): hot/cold split + global codebook ──
    // vq_hot: per kept MoE layer the hot (mxfp4) expert indices, ascending.
    // The codebook is trained on a seeded reservoir sample of ALL cold-expert
    // dequantized values, raw 16-vectors (no per-vector normalization: the
    // mxfp4 source already keeps per-32-group magnitudes similar enough that
    // raw VQ works; measured in the microquant report).
    let (vq_hot, vq_codebook): (Option<std::collections::HashMap<usize, Vec<usize>>>, Option<Vec<f32>>) =
        match cold_vq {
            None => (None, None),
            Some(n_hot) => {
                let t = std::time::Instant::now();
                let scores = expert_score_map(&source, &kept_layers);
                let hot: std::collections::HashMap<usize, Vec<usize>> =
                    scores.iter().map(|(&l, s)| (l, top_n(s, n_hot.min(cfg.n_experts)))).collect();
                // cold = kept experts minus the hot ones (with --experts M
                // the pruned tail never reaches the file NOR the codebook)
                let cold: std::collections::HashMap<usize, Vec<usize>> = scores
                    .iter()
                    .map(|(&l, s)| {
                        let keep: std::collections::HashSet<usize> = hot[&l].iter().copied().collect();
                        let kept: Vec<usize> = match &expert_sets {
                            Some(sets) => sets[&l].clone(),
                            None => (0..s.len()).collect(),
                        };
                        (l, kept.into_iter().filter(|e| !keep.contains(e)).collect())
                    })
                    .collect();
                let n_cold: usize = cold.values().map(|v| v.len()).sum();
                if n_cold == 0 {
                    println!("cold-vq: hot set covers all experts, nothing to quantize (no VQ tensors written)");
                    (Some(hot), None)
                } else {
                    println!(
                        "cold-vq: {} hot mxfp4 + {} cold VQ1 experts per MoE layer (ranked in {:.1?})",
                        n_hot.min(cfg.n_experts),
                        n_cold / cold.len().max(1),
                        t.elapsed()
                    );
                    let t = std::time::Instant::now();
                    let seed = 0x5EED_C0DE_B00B_1E5u64;
                    let train_im = if imatrix_score_only { None } else { imatrix.as_ref() };
                    let (samples, sample_w) = vq_reservoir(&source, &kept_layers, &cold, 300_000, seed, train_im);
                    let cb = match &sample_w {
                        Some(sw) => crate::quant::quant::train_codebook_weighted(&samples, sw, seed),
                        None => crate::quant::quant::train_codebook(&samples, seed),
                    };
                    println!(
                        "vq: global codebook ({}x{}) trained in {:.1?}{}",
                        crate::quant::quant::VQ_K,
                        crate::quant::quant::VQ_DIM,
                        t.elapsed(),
                        if sample_w.is_some() { " (activation-weighted)" } else { "" }
                    );
                    (Some(hot), Some(cb))
                }
            }
        };

    // ── 3c. vocabulary selection (--vocab-top): cheap, not checkpointed ──
    let vocab_plan: Option<VocabPlan> = vocab_top.as_ref().map(|(n, freq)| {
        assert!(*n > 0, "--vocab-top must be >= 1");
        build_vocab_plan(&model, &out, &source.source_json(), cfg, *n, freq, vocab_base.clone())
    });

    // ── 4. plan: output tensors in input directory order ──
    let mut plans: Vec<Plan> = Vec::new();
    for e in source.entries() {
        let role = role_of(&e.name, cfg, arch);
        let (out_name, experts_for_tensor, dtype_override): (String, Option<Vec<usize>>, Option<u8>) = match split_layer(&e.name) {
            None => (e.name.clone(), None, None),
            Some((l, rest)) => {
                let Some(nl) = new_layer_of(l) else { continue }; // pruned layer
                let pfx = format!("layers.{}.", nl);
                if role == Role::Expert {
                    // block_sparse_moe.experts.{e}.{w}
                    let tail = rest.strip_prefix("block_sparse_moe.experts.").unwrap();
                    let dot = tail.find('.').unwrap();
                    let oe: usize = tail[..dot].parse().unwrap();
                    let keep = expert_sets.as_ref().map(|s| &s[&l]);
                    let idx = match keep {
                        Some(k) => match k.iter().position(|&x| x == oe) {
                            Some(i) => i,
                            None => continue, // pruned expert
                        },
                        None => oe,
                    };
                    // cold experts (below the hot top-N) become VQ1
                    let cold = vq_hot.as_ref().is_some_and(|h| !h[&l].contains(&oe));
                    let dt = if cold { Some(DTYPE_VQ1) } else { None };
                    (format!("{}block_sparse_moe.experts.{}.{}", pfx, idx, &tail[dot + 1..]), None, dt)
                } else if matches!(role, Role::RouterW | Role::RouterB) {
                    (format!("{}{}", pfx, rest), expert_sets.as_ref().map(|s| s[&l].clone()), None)
                } else {
                    (format!("{}{}", pfx, rest), None, None)
                }
            }
        };
        let ch: Vec<usize> = channels.clone().unwrap_or_else(|| (0..d).collect());
        // embed/lm_head rows are the vocab axis: pruned by --vocab-top
        let vrows: Option<Vec<usize>> = match (&vocab_plan, e.name.as_str()) {
            (Some(v), "embed_tokens.weight" | "lm_head.weight") => Some(v.keep.clone()),
            _ => None,
        };
        let dims = if matches!(role, Role::Copy | Role::Expert) {
            e.dims.clone()
        } else {
            // compute the sliced dims without materializing the data
            let r = e.dims[0] as usize;
            let out_rows = vrows.as_ref().map(|k| k.len()).unwrap_or(r) as u32;
            match role {
                Role::VecD => vec![ch.len() as u32],
                Role::ColsD => vec![out_rows, ch.len() as u32],
                Role::RowsD => vec![ch.len() as u32, e.dims[1]],
                Role::BothD => vec![ch.len() as u32, ch.len() as u32],
                Role::RouterW => vec![experts_for_tensor.as_ref().map(|k| k.len()).unwrap_or(r) as u32, ch.len() as u32],
                Role::RouterB => vec![experts_for_tensor.as_ref().map(|k| k.len()).unwrap_or(r) as u32],
                _ => unreachable!(),
            }
        };
        plans.push(Plan {
            out_name,
            dtype: dtype_override.unwrap_or(e.dtype),
            dims,
            src_name: e.name.clone(),
            role,
            channels: ch,
            experts: experts_for_tensor,
            vocab: vrows,
        });
    }
    // physical write order with --expert-order: the plan list follows the
    // SOURCE directory order (old expert ids), which would scatter the
    // relabeled blobs. Re-sort each layer's expert run by new expert id
    // (then w1/w2/w3) so file-adjacent ids are byte-adjacent - the
    // precondition of the stream engine's run fusion. Everything else keeps
    // the source order.
    if expert_order.is_some() {
        let mut i = 0;
        while i < plans.len() {
            if plans[i].role != Role::Expert {
                i += 1;
                continue;
            }
            let layer = split_layer(&plans[i].out_name).map(|(l, _)| l);
            let mut j = i + 1;
            while j < plans.len() && plans[j].role == Role::Expert && split_layer(&plans[j].out_name).map(|(l, _)| l) == layer {
                j += 1;
            }
            plans[i..j].sort_by_key(|p| expert_plan_key(&p.out_name));
            i = j;
        }
    }
    // the global VQ codebook rides as one extra f32 tensor (src_name "" marks it)
    if vq_codebook.is_some() {
        plans.push(Plan {
            out_name: "vq_codebook".to_string(),
            dtype: DTYPE_F32,
            dims: vec![crate::quant::quant::VQ_K as u32, crate::quant::quant::VQ_DIM as u32],
            src_name: String::new(),
            role: Role::Copy,
            channels: Vec::new(),
            experts: None,
            vocab: None,
        });
    }

    // ── 5. MKIM0002 config ──
    let new_n_layers = kept_layers.len();
    let new_d = channels.as_ref().map(|c| c.len()).unwrap_or(d);
    let new_n_experts = experts.unwrap_or(cfg.n_experts);
    let new_top_k = cfg.top_k.min(new_n_experts);
    let new_vocab = vocab_plan.as_ref().map(|v| v.keep.len()).unwrap_or(cfg.vocab);
    // specials: with --vocab-top every known special is recorded at its NEW
    // id (a re-slice can then find them all again); otherwise the historical
    // bos/end_of_msg pair, unchanged.
    let specials_kv = match &vocab_plan {
        Some(v) => v
            .specials_new
            .iter()
            .map(|(n, id)| format!("\"{}\": {}", n, id))
            .collect::<Vec<_>>()
            .join(", "),
        None => format!("\"bos\": {}, \"end_of_msg\": {}", cfg.bos_id, cfg.eos_id),
    };
    let mla_layers: Vec<usize> = kept_layers.iter().enumerate().filter(|&(_, &l)| cfg.is_mla(l)).map(|(i, _)| i).collect();
    let dense_layers: Vec<usize> = kept_layers.iter().enumerate().filter(|&(_, &l)| !cfg.is_moe(l)).map(|(i, _)| i).collect();
    let tokenizer_kv = source
        .source_json()
        .get("tokenizer")
        .and_then(|t| t.as_str().map(|s| s.to_string()))
        .map(|s| format!(", \"tokenizer\": \"{}\"", s))
        .unwrap_or_default();
    let arch_kv = source.arch_config_key();
    let list = |v: &[usize]| v.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", ");
    // --expert-order audit table: new_id -> old_id per (renumbered) MoE layer
    let expert_order_kv = expert_order.as_ref().map(|m| {
        let layers: Vec<String> = kept_layers
            .iter()
            .enumerate()
            .filter_map(|(nl, &l)| m.get(&l).map(|ids| format!("\"{}\": [{}]", nl, list(ids))))
            .collect();
        format!(
            ", \"expert_order\": {{\"method\": \"frequency\", \"source\": \"{}\", \"new_to_old\": {{{}}}}}",
            route_cms_path.as_deref().unwrap(),
            layers.join(", ")
        )
    });
    let config_json = format!(
        "{{\"format\": 2{}, \"n_layers\": {}, \"hidden\": {}, \"vocab\": {}, \"n_experts\": {}, \"top_k\": {}, \"n_shared\": {}, \
\"kda_heads\": {}, \"kda_dim\": {}, \"kda_conv\": {}, \"kda_fa_rank\": {}, \"gate_lower_bound\": {}, \
\"mla_heads\": {}, \"mla_q_lora\": {}, \"mla_kv_lora\": {}, \"mla_nope\": {}, \"mla_rope\": {}, \"mla_v\": {}, \
\"routed_hidden\": {}, \"moe_inter\": {}, \"shared_inter\": {}, \"dense_inter\": {}, \
\"attn_res_block\": {}, \"first_k_dense\": {}, \"rms_eps\": {}{}, \
\"mla_layers\": [{}], \"dense_layers\": [{}], \
\"specials\": {{{}}}, \
\"pruning\": {{\"method\": \"weight-magnitude-v1\", \"hidden\": {}, \"experts\": {}, \"layers\": \"{}\"{}{}}}}}",
        arch_kv,
        new_n_layers, new_d, new_vocab, new_n_experts, new_top_k, cfg.n_shared,
        cfg.kda_heads, cfg.kda_dim, cfg.kda_conv, cfg.kda_fa, cfg.gate_lb,
        cfg.mla_heads, cfg.mla_qa, cfg.mla_kva, cfg.mla_nope, cfg.mla_rope, cfg.mla_v,
        cfg.routed_hidden, cfg.moe_inter, cfg.shared_inter, cfg.dense_inter,
        cfg.attn_res_block, cfg.first_k_dense, cfg.rms_eps, tokenizer_kv,
        list(&mla_layers), list(&dense_layers),
        specials_kv,
        new_d, new_n_experts, kept_layers.iter().map(|l| l.to_string()).collect::<Vec<_>>().join(","),
        cold_vq.map(|n| format!(", \"cold_vq\": {}", n)).unwrap_or_default(),
        vocab_top.as_ref().map(|(n, _)| format!(", \"vocab_top\": {}", n)).unwrap_or_default(),
    );
    // the expert_order audit table goes in as a top-level key (inserted
    // post-hoc: one less placeholder in an already dense format string)
    let mut config_json = config_json;
    if let Some(kv) = expert_order_kv {
        config_json.insert_str(config_json.len() - 1, &kv);
    }

    // ── 6. write ──
    let mut w = BinWriter::new();
    if expert_order.is_some() {
        // dense expert packing: the reordered blobs are read in fused spans,
        // page-alignment padding would be read and discarded on every span
        w.set_expert_align(64);
    }
    for p in &plans {
        w.add(&p.out_name, p.dtype, p.dims.clone());
    }
    let mut f = std::fs::File::create(&out).unwrap();
    let offsets = w.write_header_v2(&mut f, &config_json);
    let mut done = 0usize;
    let mut last_fetch_report = 0u64;
    let mut cur_layer: Option<usize> = None;
    let mut vq_err_sum = 0f64;
    let mut vq_werr_sum = 0f64;
    let mut vq_err_n = 0u64;
    for (p, &off) in plans.iter().zip(&offsets) {
        // the codebook plan has no source tensor: its data is the trained codebook
        if p.src_name.is_empty() {
            let cb = vq_codebook.as_ref().expect("codebook plan without --cold-vq");
            w.write_blob_at(&mut f, off, &f32_to_bytes(cb));
            done += 1;
            continue;
        }
        let se = source.entry(&p.src_name);
        match p.role {
            Role::Expert if p.dtype == DTYPE_VQ1 => {
                // imatrix column weights for this expert matrix (original
                // layer numbering: src_name), w1/w3 -> hidden, w2 -> inter
                let wts = imatrix.as_ref().and_then(|im| {
                    let (l, rest) = split_layer(&p.src_name)?;
                    im.col_weights(l, rest.rsplit('.').next()?)
                });
                let quant_w = if imatrix_score_only { None } else { wts.as_deref() };
                let (idx, err, werr) = vq_quantize_tensor(&source, se, vq_codebook.as_ref().unwrap(), quant_w, wts.as_deref());
                assert_eq!(blob_size(p.dtype, &p.dims), idx.len() as u64, "{}: vq size mismatch", p.src_name);
                w.write_blob_at(&mut f, off, &idx);
                vq_err_sum += err;
                if let Some(we) = werr {
                    vq_werr_sum += we;
                }
                vq_err_n += 1;
                if vq_err_n % 500 == 0 {
                    println!(
                        "  vq: {} tensors quantized, mean rel Frobenius error {:.3}{}",
                        vq_err_n,
                        vq_err_sum / vq_err_n as f64,
                        if werr.is_some() {
                            format!(", mean activation-weighted error {:.3}", vq_werr_sum / vq_err_n as f64)
                        } else {
                            String::new()
                        }
                    );
                }
            }
            Role::Copy | Role::Expert => {
                let blob = source.raw_blob(se);
                assert_eq!(blob_size(p.dtype, &p.dims), blob.len() as u64, "{}: size mismatch on copy", p.src_name);
                w.write_blob_at(&mut f, off, &blob);
            }
            Role::ColsD | Role::RowsD => {
                let (rows, cols) = (n_rows(se), row_width(se));
                let mut written = 0u64;
                for (r0, r1) in row_chunks(rows, cols) {
                    let vals = source.f32_rows(se, r0, r1);
                    let sliced = match &p.vocab {
                        Some(keep) => slice_vocab_rows(&vals, r0, r1, cols, &p.channels, keep),
                        None => slice_f32_rows(p.role, &vals, r0, r1, cols, &p.channels),
                    };
                    let bytes = f32_to_bytes(&sliced);
                    w.write_blob_at(&mut f, off + written, &bytes);
                    written += bytes.len() as u64;
                }
                assert_eq!(written, blob_size(DTYPE_F32, &p.dims), "{}: planned dims mismatch", p.src_name);
            }
            _ => {
                let (vals, dims) = slice_f32(se, &source.f32_rows(se, 0, n_rows(se)), p.role, &p.channels, p.experts.as_ref());
                assert_eq!(dims, p.dims, "{}: planned dims mismatch", p.src_name);
                w.write_blob_at(&mut f, off, &f32_to_bytes(&vals));
            }
        }
        done += 1;
        // one progress line per layer written (plans follow the directory
        // order, so layers are contiguous)
        if let Some((nl, _)) = split_layer(&p.out_name) {
            if cur_layer != Some(nl) {
                cur_layer = Some(nl);
                println!("  write: layer {}/{} ({}% of tensors)", nl + 1, new_n_layers, 100 * done / plans.len());
            }
        }
        if done % 20000 == 0 {
            println!("  {}/{} tensors written ({:.0?})", done, plans.len(), t0.elapsed());
        }
        if source.is_remote() {
            let fb = crate::stream::http::fetched_bytes();
            if fb - last_fetch_report >= (1 << 30) {
                last_fetch_report = fb;
                println!("  fetched {:.2} GB so far ({}/{} tensors written)", fb as f64 / 1e9, done, plans.len());
            }
        }
    }
    let out_size = std::fs::metadata(&out).unwrap().len();
    println!();
    println!("══ {} : {} tensors ══", out, plans.len());
    match std::fs::metadata(&model).ok().map(|m| m.len()) {
        Some(in_size) if !source.is_remote() => println!(
            "  size: {:.2} GB -> {:.2} GB ({:.1}%)",
            in_size as f64 / 1e9,
            out_size as f64 / 1e9,
            out_size as f64 / in_size as f64 * 100.0
        ),
        _ => println!("  input: remote safetensors via range requests (no full shard downloaded) -> {:.2} GB", out_size as f64 / 1e9),
    }
    if source.is_remote() {
        println!(
            "  bandwidth: {:.3} GB fetched in {} HTTP range requests",
            crate::stream::http::fetched_bytes() as f64 / 1e9,
            crate::stream::http::fetched_requests()
        );
    }
    println!("  config: {} layers (MLA {:?}, dense {:?}), hidden {}, {} experts top-{}", 
        new_n_layers, mla_layers, dense_layers, new_d, new_n_experts, new_top_k);
    println!("  AttnRes: block={} re-applied on the renumbered layers", cfg.attn_res_block);
    println!("  experts: mxfp4 blobs copied verbatim (no requantization)");
    if expert_order.is_some() {
        println!("  expert-order: frequency (route-cms) - router rows and expert blobs relabeled, hot experts file-adjacent, dense packing");
    }
    if vq_err_n > 0 {
        println!(
            "  cold-vq: {} tensors requantized to VQ1 ({} B/expert-matrix + one 16 KB global codebook), mean rel Frobenius error {:.3}",
            vq_err_n,
            cfg.moe_inter * cfg.routed_hidden / crate::quant::quant::VQ_DIM,
            vq_err_sum / vq_err_n as f64
        );
        if imatrix.is_some() {
            println!(
                "  cold-vq: mean activation-weighted rel error {:.3} (imatrix{})",
                vq_werr_sum / vq_err_n as f64,
                if imatrix_score_only { ", score only" } else { "-weighted codebook" }
            );
        }
    }
    println!("  done in {:.0?}", t0.elapsed());
    if let Some(v) = &vocab_plan {
        std::fs::write(&v.remap_path, &v.remap_json).unwrap_or_else(|e| panic!("{} unwritable: {}", v.remap_path, e));
        println!("  vocab: {} -> {} rows kept; runtime remap: {}", cfg.vocab, v.keep.len(), v.remap_path);
        println!("         run with: microkimi run \"...\" --model {} --vocab {}", out, v.remap_path);
    }
    ckpt.finish();
}
