// `microkimi eval`: deterministic, reproducible evaluation harness.
//
//   microkimi eval --model X.bin [--vocab V.json] [--max-new N]
//       [--ppl-file F] [--ppl-max-tokens N] [--skip-qa] [--json out.json]
//
// Two measurements, one scorecard:
//   - QA probes: 40 general-knowledge questions (geography, literature,
//     science, history, math), each asked in TWO formulations (raw
//     completion "The capital of France is" and "Q: ... A:") because a
//     pruned base model is not a chat model. Greedy decoding (argmax,
//     zero sampling), fixed max-new: run-to-run reproducible.
//   - perplexity: mean next-token NLL over a held-out generic English
//     text (embedded below, or --ppl-file), sequential forward over the
//     whole window (the text is short; no striding tricks).
//
// Determinism: greedy argmax + fixed texts + the engine's fixed-chunk
// parallel matvecs give identical scores from run to run. Nano-vocab
// models (8200) will UNK most probe words: expected, it is part of the
// honest comparison against full-vocab models - nothing is filtered.

use std::time::Instant;

struct Probe {
    cat: &'static str,
    completion: &'static str, // raw-completion prompt
    qa: &'static str,         // "Q: ... A:" prompt
    answers: &'static [&'static str],
}

const CATS: [&str; 5] = ["geography", "literature", "science", "history", "math"];

#[rustfmt::skip]
const PROBES: &[Probe] = &[
    // geography
    Probe { cat: "geography", completion: "The capital of France is", qa: "Q: What is the capital of France? A:", answers: &["Paris"] },
    Probe { cat: "geography", completion: "The capital of Japan is", qa: "Q: What is the capital of Japan? A:", answers: &["Tokyo"] },
    Probe { cat: "geography", completion: "The capital of Canada is", qa: "Q: What is the capital of Canada? A:", answers: &["Ottawa"] },
    Probe { cat: "geography", completion: "The capital of Australia is", qa: "Q: What is the capital of Australia? A:", answers: &["Canberra"] },
    Probe { cat: "geography", completion: "The capital of Brazil is", qa: "Q: What is the capital of Brazil? A:", answers: &["Brasilia", "Brasília"] },
    Probe { cat: "geography", completion: "The capital of Germany is", qa: "Q: What is the capital of Germany? A:", answers: &["Berlin"] },
    Probe { cat: "geography", completion: "The capital of Egypt is", qa: "Q: What is the capital of Egypt? A:", answers: &["Cairo"] },
    Probe { cat: "geography", completion: "The largest ocean on Earth is the", qa: "Q: What is the largest ocean on Earth? A:", answers: &["Pacific"] },
    // literature
    Probe { cat: "literature", completion: "Romeo and Juliet was written by", qa: "Q: Who wrote Romeo and Juliet? A:", answers: &["Shakespeare"] },
    Probe { cat: "literature", completion: "Don Quixote was written by", qa: "Q: Who wrote Don Quixote? A:", answers: &["Cervantes"] },
    Probe { cat: "literature", completion: "The author of the novel 1984 is", qa: "Q: Who is the author of the novel 1984? A:", answers: &["Orwell"] },
    Probe { cat: "literature", completion: "War and Peace was written by", qa: "Q: Who wrote War and Peace? A:", answers: &["Tolstoy"] },
    Probe { cat: "literature", completion: "The Odyssey was written by", qa: "Q: Who wrote the Odyssey? A:", answers: &["Homer"] },
    Probe { cat: "literature", completion: "Pride and Prejudice was written by", qa: "Q: Who wrote Pride and Prejudice? A:", answers: &["Austen"] },
    Probe { cat: "literature", completion: "The Divine Comedy was written by", qa: "Q: Who wrote the Divine Comedy? A:", answers: &["Dante"] },
    Probe { cat: "literature", completion: "The novel Moby-Dick was written by", qa: "Q: Who wrote the novel Moby-Dick? A:", answers: &["Melville"] },
    // science
    Probe { cat: "science", completion: "The chemical formula for water is", qa: "Q: What is the chemical formula for water? A:", answers: &["H2O"] },
    Probe { cat: "science", completion: "The planet closest to the Sun is", qa: "Q: What is the planet closest to the Sun? A:", answers: &["Mercury"] },
    Probe { cat: "science", completion: "The speed of light is approximately", qa: "Q: What is approximately the speed of light? A:", answers: &["299", "300,000", "300 000"] },
    Probe { cat: "science", completion: "The powerhouse of the cell is the", qa: "Q: What is the powerhouse of the cell? A:", answers: &["mitochondria", "mitochondrion"] },
    Probe { cat: "science", completion: "The gas that plants absorb from the air is", qa: "Q: What gas do plants absorb from the air? A:", answers: &["carbon dioxide", "CO2"] },
    Probe { cat: "science", completion: "Water boils at a temperature of", qa: "Q: At what temperature does water boil? A:", answers: &["100"] },
    Probe { cat: "science", completion: "The force that pulls objects toward the Earth is", qa: "Q: What force pulls objects toward the Earth? A:", answers: &["gravity", "gravitation"] },
    Probe { cat: "science", completion: "The largest planet in the Solar System is", qa: "Q: What is the largest planet in the Solar System? A:", answers: &["Jupiter"] },
    // history
    Probe { cat: "history", completion: "The French Revolution began in the year", qa: "Q: In what year did the French Revolution begin? A:", answers: &["1789"] },
    Probe { cat: "history", completion: "World War II ended in the year", qa: "Q: In what year did World War II end? A:", answers: &["1945"] },
    Probe { cat: "history", completion: "The first person to walk on the Moon was", qa: "Q: Who was the first person to walk on the Moon? A:", answers: &["Armstrong"] },
    Probe { cat: "history", completion: "The Berlin Wall fell in the year", qa: "Q: In what year did the Berlin Wall fall? A:", answers: &["1989"] },
    Probe { cat: "history", completion: "Christopher Columbus first reached America in the year", qa: "Q: In what year did Christopher Columbus first reach America? A:", answers: &["1492"] },
    Probe { cat: "history", completion: "The first emperor of Rome was", qa: "Q: Who was the first emperor of Rome? A:", answers: &["Augustus", "Octavian"] },
    Probe { cat: "history", completion: "The American Declaration of Independence was adopted in", qa: "Q: In what year was the American Declaration of Independence adopted? A:", answers: &["1776"] },
    Probe { cat: "history", completion: "The Great Wall was built to protect", qa: "Q: What was the Great Wall built to protect? A:", answers: &["China"] },
    // math
    Probe { cat: "math", completion: "Two plus two equals", qa: "Q: What is two plus two? A:", answers: &["4", "four"] },
    Probe { cat: "math", completion: "The square root of 144 is", qa: "Q: What is the square root of 144? A:", answers: &["12"] },
    Probe { cat: "math", completion: "Ten times ten equals", qa: "Q: What is ten times ten? A:", answers: &["100", "one hundred"] },
    Probe { cat: "math", completion: "A triangle has", qa: "Q: How many sides does a triangle have? A:", answers: &["3", "three"] },
    Probe { cat: "math", completion: "The value of pi is approximately", qa: "Q: What is approximately the value of pi? A:", answers: &["3.14"] },
    Probe { cat: "math", completion: "Seven multiplied by eight equals", qa: "Q: What is seven multiplied by eight? A:", answers: &["56", "fifty-six"] },
    Probe { cat: "math", completion: "One hundred divided by four equals", qa: "Q: What is one hundred divided by four? A:", answers: &["25", "twenty-five"] },
    Probe { cat: "math", completion: "The smallest prime number is", qa: "Q: What is the smallest prime number? A:", answers: &["2", "two"] },
];

/// Held-out perplexity text: generic encyclopedia-style English, written
/// for this harness (no training-corpus overlap by construction).
const PPL_TEXT: &str = "\
The Amazon River flows through the largest tropical rainforest on Earth. \
Spanning nine countries in South America, the Amazon basin covers roughly \
six million square kilometers and contains about ten percent of all known \
species. The river discharges more water than the next seven largest rivers \
combined, and its floodplain rises and falls by more than ten meters over \
the course of a year.

A city is shaped by the way its inhabitants move. Before the invention of \
the bicycle and the tramway, most people lived within walking distance of \
their workplace, and urban districts mixed workshops, shops, and housing. \
The railway allowed the middle class to settle in suburbs, while the \
elevator made the skyscraper practical by sparing office workers a long \
climb. Each new transport technology redraws the map of daily life.

Photosynthesis is the process by which green plants convert sunlight into \
chemical energy. Inside the chloroplasts of leaf cells, molecules of \
chlorophyll absorb light, mostly in the blue and red parts of the spectrum, \
which is why leaves appear green. The energy drives a chain of reactions \
that splits water, releases oxygen, and builds sugars from carbon dioxide. \
Nearly every food chain on the planet begins with this quiet chemistry.

The printing press changed Europe faster than any machine before it. When \
Johannes Gutenberg set up his workshop in Mainz around 1450, a single \
scribal Bible could take three years to copy. Fifty years later, printers \
in more than two hundred cities had produced millions of books. Ideas that \
once traveled at the pace of a walking messenger now crossed the continent \
in a season, and the cost of a book fell to a fraction of its former price.";

// ── helpers ──

enum EvalModel {
    K3(crate::model::Model),
    Qwen(crate::model::qwen::QwenModel),
    DeepSeek(crate::model::deepseek::DsModel),
}

impl EvalModel {
    fn reset(&mut self) {
        match self {
            EvalModel::K3(model) => model.reset_cache(),
            EvalModel::Qwen(model) => model.reset(),
            EvalModel::DeepSeek(model) => model.reset(),
        }
    }

    fn forward(&mut self, token: u32, pos: usize) -> Vec<f32> {
        match self {
            EvalModel::K3(model) => model.forward(token, pos),
            EvalModel::Qwen(model) => model.forward(token),
            EvalModel::DeepSeek(model) => model.forward(token, pos),
        }
    }

    fn prefill(&mut self, ids: &[u32]) -> Vec<f32> {
        match self {
            EvalModel::K3(model) => model.prefill(ids, 0),
            EvalModel::Qwen(model) => {
                let mut logits = Vec::new();
                for &token in ids {
                    logits = model.forward(token);
                }
                logits
            }
            EvalModel::DeepSeek(model) => {
                let mut logits = Vec::new();
                for (pos, &token) in ids.iter().enumerate() {
                    logits = model.forward(token, pos);
                }
                logits
            }
        }
    }
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0 as u32
}

/// Case-insensitive match with alphanumeric boundaries: "Paris" hits in
/// " Paris, " but not in "Parisian", "4" does not hit inside "1945".
fn contains_answer(text: &str, ans: &str) -> bool {
    let t = text.to_lowercase();
    let a = ans.to_lowercase();
    let alnum = |b: u8| b.is_ascii_alphanumeric();
    let mut start = 0;
    while let Some(i) = t[start..].find(&a) {
        let i = start + i;
        let before = i == 0 || !alnum(t.as_bytes()[i - 1]);
        let after = i + a.len() >= t.len() || !alnum(t.as_bytes()[i + a.len()]);
        if before && after {
            return true;
        }
        start = i + 1;
    }
    false
}

/// Greedy raw completion of `prompt`, at most `max_new` tokens, stop at the
/// raw-mode stop token. Returns (decoded completion, tokens processed).
fn greedy_complete(
    model: &mut EvalModel,
    tok: &crate::tokenizer::AnyTokenizer,
    prompt: &str,
    max_new: usize,
) -> (String, usize) {
    model.reset();
    let ids = tok.encode_raw(prompt);
    let mut logits = model.prefill(&ids);
    let mut pos = ids.len();
    let mut out = Vec::new();
    let stop = tok.raw_stop();
    for _ in 0..max_new {
        let next = argmax(&logits);
        if next == stop {
            break;
        }
        out.push(next);
        logits = model.forward(next, pos);
        pos += 1;
    }
    (tok.decode(&out), ids.len() + out.len())
}

/// Mean next-token NLL over the whole text (sequential forward, one window).
/// Returns (ppl, n_tokens_scored, tokens_processed).
fn perplexity(
    model: &mut EvalModel,
    tok: &crate::tokenizer::AnyTokenizer,
    text: &str,
    max_tokens: Option<usize>,
) -> (f64, usize, usize) {
    model.reset();
    let mut ids = tok.encode_raw(text);
    if let Some(limit) = max_tokens {
        ids.truncate(limit);
    }
    let mut nll = 0f64;
    let mut n = 0usize;
    let mut logits = Vec::new();
    for (i, &id) in ids.iter().enumerate() {
        logits = model.forward(id, i);
        if i + 1 < ids.len() {
            let target = ids[i + 1] as usize;
            let max = logits.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x)) as f64;
            let lse = logits
                .iter()
                .map(|&x| ((x as f64) - max).exp())
                .sum::<f64>()
                .ln()
                + max;
            nll += lse - logits[target] as f64;
            n += 1;
        }
    }
    let _ = logits;
    ((nll / n.max(1) as f64).exp(), n, ids.len())
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

struct Hit {
    cat: &'static str,
    fmt: &'static str,
    prompt: String,
    answer: String,
    completion: String,
}

pub fn run(args: &[String]) {
    let t0 = Instant::now();
    let mp = crate::model_flag(args).unwrap_or_else(crate::bin_path);
    let max_new: usize = crate::value_flag(args, "--max-new")
        .and_then(|s| s.parse().ok())
        .unwrap_or(15);
    let ppl_file = crate::value_flag(args, "--ppl-file");
    let json_out = crate::value_flag(args, "--json");
    let skip_qa = args.iter().any(|arg| arg == "--skip-qa");
    let ppl_max_tokens = crate::value_flag(args, "--ppl-max-tokens").map(|value| {
        value
            .parse::<usize>()
            .ok()
            .filter(|&count| count >= 2)
            .expect("--ppl-max-tokens must be an integer of at least two")
    });

    // --stream / --stream-ram / --stream-fallback are honored (streaming
    // load): bit-identical scores without the fallback, the degraded-mode
    // quality cost with it (MICROKIMI_FORCE_FALLBACK=1 forces 100% shadows)
    let stream_mb = crate::stream_ram_flag(args);
    let config = crate::quant::weights::read_config(&mp);
    let packs = crate::adapter_flags(args);
    let (tok, mut model) = if config.qwen.is_some() {
        if stream_mb.is_some() {
            eprintln!("error: --stream is not supported by Qwen eval");
            std::process::exit(1);
        }
        let tok = crate::load_qwen_any_tokenizer(&mp, crate::vocab_flag(args), config.vocab);
        let model = if packs.is_empty() {
            crate::model::qwen::QwenModel::load(&mp)
        } else {
            crate::model::qwen::QwenModel::load_with_adapters(&mp, &packs)
        };
        (tok, EvalModel::Qwen(model))
    } else if config.ds.is_some() {
        if !packs.is_empty() {
            eprintln!("error: external adapter packs are not supported by DeepSeek eval");
            std::process::exit(1);
        }
        if stream_mb.is_some() {
            eprintln!("error: --stream is not supported by DeepSeek eval");
            std::process::exit(1);
        }
        let tok = crate::load_ds_any_tokenizer(&mp, crate::vocab_flag(args), config.vocab);
        (
            tok,
            EvalModel::DeepSeek(crate::model::deepseek::DsModel::load(&mp)),
        )
    } else {
        let tok = crate::load_any_tokenizer(&mp, crate::vocab_flag(args), config.vocab);
        let model = crate::load_k3_model(&mp, stream_mb);
        crate::check_tok_compat(&tok, &model);
        (tok, EvalModel::K3(model))
    };
    if tok.vocab_size() != config.vocab {
        eprintln!(
            "error: tokenizer/model mismatch - model vocab is {}, tokenizer vocab is {}.",
            config.vocab,
            tok.vocab_size()
        );
        std::process::exit(1);
    }
    let model_size = std::fs::metadata(&mp).map(|m| m.len()).unwrap_or(0);

    // ── QA probes ──
    let mut processed = 0usize; // forward tokens, for the tok/s meta
    let mut cat_hits = [0usize; 5];
    let mut cat_tot = [0usize; 5];
    let mut fmt_hits = [0usize; 2]; // [completion, qa]
    let mut hits: Vec<Hit> = Vec::new();
    let mut total = 0usize;
    let mut total_hits = 0usize;
    if !skip_qa {
        for p in PROBES {
            let ci = CATS.iter().position(|&c| c == p.cat).unwrap();
            for (fi, prompt) in [p.completion, p.qa].iter().enumerate() {
                let (text, n) = greedy_complete(&mut model, &tok, prompt, max_new);
                processed += n;
                let hit = p.answers.iter().find(|a| contains_answer(&text, a));
                total += 1;
                cat_tot[ci] += 1;
                if let Some(a) = hit {
                    total_hits += 1;
                    cat_hits[ci] += 1;
                    fmt_hits[fi] += 1;
                    hits.push(Hit {
                        cat: p.cat,
                        fmt: if fi == 0 { "completion" } else { "qa" },
                        prompt: prompt.to_string(),
                        answer: a.to_string(),
                        completion: text.replace('\n', " "),
                    });
                }
            }
        }
    }

    // ── perplexity ──
    let (ppl_text, ppl_src) = match &ppl_file {
        Some(f) => (
            std::fs::read_to_string(f).unwrap_or_else(|e| panic!("{} unreadable: {}", f, e)),
            f.clone(),
        ),
        None => (PPL_TEXT.to_string(), "embedded eval text".to_string()),
    };
    let (ppl, ppl_tokens, ppl_processed) = perplexity(&mut model, &tok, &ppl_text, ppl_max_tokens);
    processed += ppl_processed;

    let dt = t0.elapsed().as_secs_f64();
    let tok_s = processed as f64 / dt;
    let tok_name = match &tok {
        crate::tokenizer::AnyTokenizer::Nano(_) => {
            format!("nano remap (vocab {})", tok.vocab_size())
        }
        crate::tokenizer::AnyTokenizer::Qwen(_) => format!("Qwen BPE (vocab {})", tok.vocab_size()),
        crate::tokenizer::AnyTokenizer::Ds(_) | crate::tokenizer::AnyTokenizer::DsNano(_) => {
            format!("DeepSeek BPE (vocab {})", tok.vocab_size())
        }
        _ => format!("full Kimi (vocab {})", tok.vocab_size()),
    };

    // ── scorecard ──
    let mut card = String::new();
    card.push_str("══ microkimi eval ══\n");
    card.push_str(&format!(
        "model: {} ({:.1} MB)\n",
        mp,
        model_size as f64 / 1e6
    ));
    card.push_str(&format!("tokenizer: {}\n", tok_name));
    card.push('\n');
    if skip_qa {
        card.push_str("── QA probes skipped (--skip-qa) ──\n");
    } else {
        card.push_str(&format!(
            "── QA probes (greedy, max-new {}, {} questions x 2 formulations) ──\n",
            max_new,
            PROBES.len()
        ));
        for (i, c) in CATS.iter().enumerate() {
            card.push_str(&format!("  {:<12} {}/{}\n", c, cat_hits[i], cat_tot[i]));
        }
        card.push_str(&format!(
            "  {:<12} {}/{} (completion {}/{}, qa {}/{})\n",
            "TOTAL",
            total_hits,
            total,
            fmt_hits[0],
            PROBES.len(),
            fmt_hits[1],
            PROBES.len()
        ));
        card.push_str("hits:\n");
        if hits.is_empty() {
            card.push_str("  (none)\n");
        }
        for h in &hits {
            card.push_str(&format!(
                "  [{}/{}] \"{}\" -> {} (\"{}\")\n",
                h.cat, h.fmt, h.prompt, h.answer, h.completion
            ));
        }
    }
    card.push('\n');
    card.push_str("── perplexity ──\n");
    card.push_str(&format!(
        "  text: {} ({} tokens scored)\n",
        ppl_src, ppl_tokens
    ));
    card.push_str(&format!(
        "  NLL/token: {:.4}   PPL: {:.2}\n",
        (ppl.ln()),
        ppl
    ));
    card.push('\n');
    card.push_str("── meta ──\n");
    card.push_str(&format!(
        "  {} tokens processed in {:.1?} ({:.0} tok/s)\n",
        processed,
        t0.elapsed(),
        tok_s
    ));
    print!("{}", card);

    // ── JSON archive ──
    if let Some(jp) = json_out {
        let mut j = String::from("{\n");
        j.push_str(&format!("  \"model\": \"{}\",\n", json_escape(&mp)));
        j.push_str(&format!("  \"model_bytes\": {},\n", model_size));
        j.push_str(&format!(
            "  \"tokenizer\": \"{}\",\n",
            json_escape(&tok_name)
        ));
        j.push_str(&format!("  \"max_new\": {},\n", max_new));
        j.push_str(&format!("  \"skip_qa\": {},\n", skip_qa));
        match ppl_max_tokens {
            Some(limit) => j.push_str(&format!("  \"ppl_max_tokens\": {},\n", limit)),
            None => j.push_str("  \"ppl_max_tokens\": null,\n"),
        }
        j.push_str("  \"qa\": {\n");
        for (i, c) in CATS.iter().enumerate() {
            j.push_str(&format!(
                "    \"{}\": [{}, {}],\n",
                c, cat_hits[i], cat_tot[i]
            ));
        }
        j.push_str(&format!("    \"total\": [{}, {}],\n", total_hits, total));
        j.push_str(&format!(
            "    \"completion\": [{}, {}],\n",
            fmt_hits[0],
            PROBES.len()
        ));
        j.push_str(&format!(
            "    \"qa_format\": [{}, {}],\n",
            fmt_hits[1],
            PROBES.len()
        ));
        j.push_str("    \"hits\": [\n");
        for (i, h) in hits.iter().enumerate() {
            j.push_str(&format!(
                "      {{\"cat\": \"{}\", \"fmt\": \"{}\", \"prompt\": \"{}\", \"answer\": \"{}\", \"completion\": \"{}\"}}{}\n",
                h.cat,
                h.fmt,
                json_escape(&h.prompt),
                json_escape(&h.answer),
                json_escape(&h.completion),
                if i + 1 < hits.len() { "," } else { "" }
            ));
        }
        j.push_str("    ]\n  },\n");
        j.push_str(&format!(
            "  \"ppl\": {{\"source\": \"{}\", \"tokens\": {}, \"nll\": {:.6}, \"ppl\": {:.4}}},\n",
            json_escape(&ppl_src),
            ppl_tokens,
            ppl.ln(),
            ppl
        ));
        j.push_str(&format!(
            "  \"meta\": {{\"tokens_processed\": {}, \"secs\": {:.1}, \"tok_s\": {:.1}}}\n",
            processed, dt, tok_s
        ));
        j.push_str("}\n");
        std::fs::write(&jp, j).unwrap_or_else(|e| panic!("cannot write {}: {}", jp, e));
        println!("json: {}", jp);
    }
    crate::stream_report_maybe(stream_mb);
}
