// MoE and dense FFN: router (noaux_tc), top-k expert dispatch with optional
// ExpertCache streaming (shadow fallback, lookahead/draft prefetch), shared
// experts, dense MLP for the non-MoE layers. moe_prefill batches the expert
// GEMMs over a chunk; the decode path computes the same values per token.

use super::*;

/// Router-lookahead prefetch (--stream-predict N; MICROKIMI_LOOKAHEAD=0
/// reverts to the Markov predictor): runs the NEXT MoE layer's router (one
/// small GEMV, n_experts x d) on the CURRENT MoE input x - the closest state
/// available to what that router will actually see (its true input is the
/// post-attention layernorm of the next layer, not yet computed) - and
/// background-prefetches its top-N predicted experts through the stream
/// cache while the current layer's experts compute. The selection replicates
/// the noaux_tc rule (sigmoid + correction bias, ranked by key) without
/// touching moe_forward's own selection; only WHEN bytes are fetched
/// changes, never WHICH experts run: the output is bit-identical.
pub(super) fn moe_lookahead(cfg: &Config, data: &[u8], w: &MoeW, layer2: usize, x: &[f32], n: usize, cache: &crate::stream::ExpertCache) {
    let gate_w = Model::t(data, &w.gate_w);
    let gate_b = Model::t(data, &w.gate_b);
    let mut logits = vec![0f32; cfg.n_experts];
    matvec(gate_w, cfg.n_experts, cfg.d, x, &mut logits);
    let mut ids: Vec<(u32, f32)> = logits.iter().enumerate().map(|(i, &l)| (i as u32, sigmoid(l) + gate_b[i])).collect();
    ids.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let expert_packed = cfg.routed_hidden * cfg.moe_inter / 2;
    let expert_blob = expert_packed + cfg.routed_hidden * cfg.moe_inter / 32;
    let expert_vq_blob = cfg.routed_hidden * cfg.moe_inter / crate::quant::VQ_DIM;
    let jobs: Vec<(u32, u32, [u64; 3], usize)> = ids
        .iter()
        .take(n)
        .map(|&(e, _)| {
            let eblob = if w.experts_vq[e as usize] { expert_vq_blob } else { expert_blob };
            (layer2 as u32, e, w.experts[e as usize], eblob)
        })
        .collect();
    cache.prefetch(jobs);
}

pub(super) fn moe_forward(cfg: &Config, data: &[u8], w: &MoeW, x: &[f32], prof: &mut Prof, layer: usize, pos: usize, stream: Option<&crate::stream::ExpertCache>) -> Vec<f32> {
    // noaux_tc router: sigmoid, +bias for selection, weights without bias
    let tm = Instant::now();
    let gate_w = Model::t(data, &w.gate_w);
    let gate_b = Model::t(data, &w.gate_b);
    let mut logits = vec![0f32; cfg.n_experts];
    matvec(gate_w, cfg.n_experts, cfg.d, x, &mut logits);
    let mut sel: Vec<(u32, f32, f32)> = Vec::with_capacity(cfg.top_k); // (expert, score, key)
    for (i, &l) in logits.iter().enumerate() {
        let sc = sigmoid(l);
        let key = sc + gate_b[i];
        let item = (i as u32, sc, key);
        if sel.len() < cfg.top_k {
            sel.push(item);
            if sel.len() == cfg.top_k {
                sel.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
            }
        } else if key > sel[cfg.top_k - 1].2 {
            sel[cfg.top_k - 1] = item;
            sel.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        }
    }
    let sumw: f32 = sel.iter().map(|s| s.1).sum::<f32>() + 1e-20;
    let weights: Vec<f32> = sel.iter().map(|s| s.1 / sumw).collect();
    if ROUTER_LAYERS.contains(&layer) {
        let mut ids: Vec<u32> = sel.iter().map(|s| s.0).collect();
        ids.sort();
        parity_rec(|d| {
            d.router.insert((pos, layer), ids);
        });
    }
    // --debug-routing: top-3 by renormalized weight + count of top-16 appearances
    ROUTING.with(|r| {
        if let Some(d) = r.borrow_mut().as_mut() {
            let mut top3: Vec<(u32, f32)> = sel.iter().map(|s| (s.0, s.1 / sumw)).collect();
            top3.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            top3.truncate(3);
            d.cur.push((layer, top3));
            for s in &sel {
                *d.counts.entry((layer, s.0)).or_insert(0) += 1;
            }
        }
    });
    // count-min routing statistics (no-op unless routestats/MICROKIMI_ROUTECMS)
    for s in &sel {
        crate::cms::record(layer, s.0);
    }
    // routing history for the draft-aware prefetch (streaming runs only;
    // no-op unless MICROKIMI_DRAFTPREFETCH is on)
    if stream.is_some() {
        crate::stream::route_record(pos as u32, layer as u32, sel.iter().map(|s| s.0).collect());
    }
    let mut h = vec![0f32; cfg.routed_hidden];
    matvec(Model::t(data, &w.routed_down), cfg.routed_hidden, cfg.d, x, &mut h);
    prof.t_router += tm.elapsed().as_secs_f64();
    // imatrix calibration hook (no-op unless `calibrate` is running)
    crate::imatrix::record_hidden(layer, &h);

    // MXFP4 experts (dequantized on the fly): SiTU(cat(w1 h, w3 h)) then w2.
    // The 16 experts are independent → one pool job per expert (offsets
    // precomputed at load time, zero lookup). Combination in fixed order after
    // the barrier → deterministic.
    let tm = Instant::now();
    let expert_packed = cfg.routed_hidden * cfg.moe_inter / 2;
    let expert_blob = expert_packed + cfg.routed_hidden * cfg.moe_inter / 32;
    let expert_vq_blob = cfg.routed_hidden * cfg.moe_inter / crate::quant::VQ_DIM;
    let (erh, emi) = (cfg.routed_hidden, cfg.moe_inter); // copies for the 'static closures
    let cbp = crate::pool::SPtr(w.vq_cb.as_ptr());
    let cblen = w.vq_cb.len();
    let mut outs = vec![0f32; cfg.top_k * cfg.routed_hidden];
    match stream {
        // historical full-load path: expert blobs read straight from the file image
        None => {
            let dp = crate::pool::SPtrU8(data.as_ptr());
            let dlen = data.len();
            let hp = crate::pool::SPtr(h.as_ptr());
            let op = crate::pool::MPtr(outs.as_mut_ptr());
            let mut jobs: Vec<crate::pool::Job> = Vec::with_capacity(cfg.top_k);
            for (ei, _) in weights.iter().enumerate() {
                let offs = w.experts[sel[ei].0 as usize];
                let vq = w.experts_vq[sel[ei].0 as usize];
                let eblob = if vq { expert_vq_blob } else { expert_blob };
                jobs.push(Box::new(move || {
                    let (dp, hp, op, cbp) = (dp, hp, op, cbp);
                    unsafe {
                        let data = std::slice::from_raw_parts(dp.0, dlen);
                        let h = std::slice::from_raw_parts(hp.0, erh);
                        let blob = |i: usize| &data[offs[i] as usize..offs[i] as usize + eblob];
                        let mut a = vec![0f32; emi];
                        let mut u = vec![0f32; emi];
                        if vq {
                            // cold expert: gather from the L1-resident codebook
                            let cb = std::slice::from_raw_parts(cbp.0, cblen);
                            crate::quant::matvec_vq(cb, blob(0), emi, erh, h, &mut a);
                            crate::quant::matvec_vq(cb, blob(2), emi, erh, h, &mut u);
                        } else {
                            crate::mxfp4::matvec_packed(&blob(0)[..expert_packed], &blob(0)[expert_packed..], emi, erh, h, &mut a, 1);
                            crate::mxfp4::matvec_packed(&blob(2)[..expert_packed], &blob(2)[expert_packed..], emi, erh, h, &mut u, 1);
                        }
                        let mut act = vec![0f32; emi];
                        for j in 0..emi {
                            act[j] = situ(a[j], u[j]);
                        }
                        crate::imatrix::record_inter(layer, &act);
                        let o = std::slice::from_raw_parts_mut(op.0.add(ei * erh), erh);
                        if vq {
                            let cb = std::slice::from_raw_parts(cbp.0, cblen);
                            crate::quant::matvec_vq(cb, blob(1), erh, emi, &act, o);
                        } else {
                            crate::mxfp4::matvec_packed(&blob(1)[..expert_packed], &blob(1)[expert_packed..], erh, emi, &act, o, 1);
                        }
                    }
                }));
            }
            crate::pool::pool().run(jobs);
        }
        // --stream: router-first prefetch. The router above already selected
        // the 16 experts; each pool job pulls its packed bytes through the
        // three-tier cache (RAM LRU -> disk -> HTTP) and computes as soon as
        // its bytes land. Same bytes, same matvec sequence: bit-identical.
        Some(cache) => {
            let cp = cache as *const crate::stream::ExpertCache as usize;
            let hp = crate::pool::SPtr(h.as_ptr());
            let op = crate::pool::MPtr(outs.as_mut_ptr());
            let layer32 = layer as u32;
            // Submit the reads sorted by file offset: the top-k experts are
            // not stored in id order, so offset order turns scattered seeks
            // into a near-sequential sweep. Each job writes its own output
            // slot, so the submission order cannot change the result.
            let mut order: Vec<usize> = (0..weights.len()).collect();
            if crate::stream::offset_sort() {
                order.sort_by_key(|&ei| w.experts[sel[ei].0 as usize][0]);
            }
            // fused run fetch: the missing experts of this layer are pulled
            // through the cache in as few physical reads as possible (one
            // span read per run of file-adjacent experts); the compute jobs
            // below then hit the RAM LRU. No-op when fusion is off or the
            // source is remote.
            let items: Vec<(u32, [u64; 3], usize)> = order
                .iter()
                .map(|&ei| {
                    let e = sel[ei].0;
                    (e, w.experts[e as usize], if w.experts_vq[e as usize] { expert_vq_blob } else { expert_blob })
                })
                .collect();
            cache.warm_batch(layer32, &items);
            let mut jobs: Vec<crate::pool::Job> = Vec::with_capacity(cfg.top_k);
            for &ei in &order {
                let e = sel[ei].0;
                let offs = w.experts[e as usize];
                let vq = w.experts_vq[e as usize];
                let eblob = if vq { expert_vq_blob } else { expert_blob };
                jobs.push(Box::new(move || {
                    let (hp, op, cbp) = (hp, op, cbp);
                    unsafe {
                        let cache = &*(cp as *const crate::stream::ExpertCache);
                        let served = cache.get(layer32, e, offs, eblob);
                        let h = std::slice::from_raw_parts(hp.0, erh);
                        // shadow fallback (--stream-fallback): a cache miss
                        // comes back as the resident VQ1 shadow (shadow
                        // codebook, 3 x expert_vq_blob bytes) - degraded,
                        // refilled in the background. Served::Full is the
                        // historical bit-identical path.
                        let (bytes, vq, eb, cb): (&[u8], bool, usize, &[f32]) = match &served {
                            crate::stream::Served::Full(b) => (&b[..], vq, eblob, std::slice::from_raw_parts(cbp.0, cblen)),
                            crate::stream::Served::Shadow(s, off) => {
                                (&s.data[*off..*off + 3 * expert_vq_blob], true, expert_vq_blob, &s.cb[..])
                            }
                        };
                        let blob = |i: usize| &bytes[i * eb..(i + 1) * eb];
                        let mut a = vec![0f32; emi];
                        let mut u = vec![0f32; emi];
                        if vq {
                            crate::quant::matvec_vq(cb, blob(0), emi, erh, h, &mut a);
                            crate::quant::matvec_vq(cb, blob(2), emi, erh, h, &mut u);
                        } else {
                            crate::mxfp4::matvec_packed(&blob(0)[..expert_packed], &blob(0)[expert_packed..], emi, erh, h, &mut a, 1);
                            crate::mxfp4::matvec_packed(&blob(2)[..expert_packed], &blob(2)[expert_packed..], emi, erh, h, &mut u, 1);
                        }
                        let mut act = vec![0f32; emi];
                        for j in 0..emi {
                            act[j] = situ(a[j], u[j]);
                        }
                        crate::imatrix::record_inter(layer, &act);
                        let o = std::slice::from_raw_parts_mut(op.0.add(ei * erh), erh);
                        if vq {
                            crate::quant::matvec_vq(cb, blob(1), erh, emi, &act, o);
                        } else {
                            crate::mxfp4::matvec_packed(&blob(1)[..expert_packed], &blob(1)[expert_packed..], erh, emi, &act, o, 1);
                        }
                    }
                }));
            }
            crate::pool::pool().run(jobs);
        }
    }
    let mut y = vec![0f32; cfg.routed_hidden];
    for (ei, &wi) in weights.iter().enumerate() {
        for j in 0..cfg.routed_hidden {
            y[j] += wi * outs[ei * cfg.routed_hidden + j];
        }
    }
    // norm BEFORE up-proj
    let mut yn = vec![0f32; cfg.routed_hidden];
    rmsnorm(cfg, &y, Model::t(data, &w.routed_norm), &mut yn);
    let mut out = vec![0f32; cfg.d];
    matvec(Model::t(data, &w.routed_up), cfg.d, cfg.routed_hidden, &yn, &mut out);
    // shared experts (2): SiTU MLP on the pre-down input
    let mut sa = vec![0f32; cfg.shared_inter];
    let mut su = vec![0f32; cfg.shared_inter];
    matvec(Model::t(data, &w.shared_gate), cfg.shared_inter, cfg.d, x, &mut sa);
    matvec(Model::t(data, &w.shared_up), cfg.shared_inter, cfg.d, x, &mut su);
    let mut sact = vec![0f32; cfg.shared_inter];
    for j in 0..cfg.shared_inter {
        sact[j] = situ(sa[j], su[j]);
    }
    let mut sout = vec![0f32; cfg.d];
    matvec(Model::t(data, &w.shared_down), cfg.d, cfg.shared_inter, &sact, &mut sout);
    if layer == 1 {
        let (routed, shared) = (out.clone(), sout.clone());
        parity_rec(|d| {
            d.l1_routed.insert(pos, routed);
            d.l1_shared.insert(pos, shared);
        });
    }
    for j in 0..cfg.d {
        out[j] += sout[j];
    }
    prof.t_experts += tm.elapsed().as_secs_f64();
    out
}

pub(super) fn dense_forward(cfg: &Config, data: &[u8], w: &DenseW, x: &[f32], prof: &mut Prof) -> Vec<f32> {
    let tm = Instant::now();
    let mut a = vec![0f32; cfg.dense_inter];
    let mut u = vec![0f32; cfg.dense_inter];
    matvec(Model::t(data, &w.gate), cfg.dense_inter, cfg.d, x, &mut a);
    matvec(Model::t(data, &w.up), cfg.dense_inter, cfg.d, x, &mut u);
    let mut act = vec![0f32; cfg.dense_inter];
    for j in 0..cfg.dense_inter {
        act[j] = situ(a[j], u[j]);
    }
    let mut out = vec![0f32; cfg.d];
    matvec(Model::t(data, &w.down), cfg.d, cfg.dense_inter, &act, &mut out);
    prof.t_experts += tm.elapsed().as_secs_f64();
    out
}

/// Batched MoE for prefill: `x` = n position rows [n * d], returns [n * d].
/// Router and shared/latent projections run as gemm_batch; the top-k
/// selection is per position (same code as the sequential path); the expert
/// work is grouped by expert id, one pool job per used expert over all its
/// assigned (position, slot) pairs, so a packed expert blob is read once per
/// prompt instead of once per token. Bit-identical to moe_forward per
/// position: each expert evaluation is the same matvec_packed sequence and
/// the combination keeps the slot order.
#[allow(clippy::too_many_arguments)]
pub(super) fn moe_prefill(
    cfg: &Config,
    data: &[u8],
    w: &MoeW,
    x: &[f32],
    n: usize,
    prof: &mut Prof,
    layer: usize,
    pos0: usize,
    stream: Option<&crate::stream::ExpertCache>,
) -> Vec<f32> {
    // noaux_tc router: sigmoid, +bias for selection, weights without bias
    let tm = Instant::now();
    let gate_w = Model::t(data, &w.gate_w);
    let gate_b = Model::t(data, &w.gate_b);
    let mut logits = vec![0f32; n * cfg.n_experts];
    gemm_batch(gate_w, cfg.n_experts, cfg.d, x, n, &mut logits);
    // top-k selection per position (identical to moe_forward)
    let mut sels: Vec<Vec<(u32, f32)>> = Vec::with_capacity(n); // (expert, renormalized weight) in slot order
    for t in 0..n {
        let logits_t = &logits[t * cfg.n_experts..(t + 1) * cfg.n_experts];
        let mut sel: Vec<(u32, f32, f32)> = Vec::with_capacity(cfg.top_k); // (expert, score, key)
        for (i, &l) in logits_t.iter().enumerate() {
            let sc = sigmoid(l);
            let key = sc + gate_b[i];
            let item = (i as u32, sc, key);
            if sel.len() < cfg.top_k {
                sel.push(item);
                if sel.len() == cfg.top_k {
                    sel.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
                }
            } else if key > sel[cfg.top_k - 1].2 {
                sel[cfg.top_k - 1] = item;
                sel.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
            }
        }
        let sumw: f32 = sel.iter().map(|s| s.1).sum::<f32>() + 1e-20;
        if ROUTER_LAYERS.contains(&layer) {
            let mut ids: Vec<u32> = sel.iter().map(|s| s.0).collect();
            ids.sort();
            parity_rec(|d| {
                d.router.insert((pos0 + t, layer), ids);
            });
        }
        // --debug-routing: top-3 by renormalized weight + count of top-16 appearances
        ROUTING.with(|r| {
            if let Some(d) = r.borrow_mut().as_mut() {
                let mut top3: Vec<(u32, f32)> = sel.iter().map(|s| (s.0, s.1 / sumw)).collect();
                top3.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                top3.truncate(3);
                d.cur.push((layer, top3));
                for s in &sel {
                    *d.counts.entry((layer, s.0)).or_insert(0) += 1;
                }
            }
        });
        sels.push(sel.iter().map(|s| (s.0, s.1 / sumw)).collect());
        // count-min routing statistics (no-op unless routestats/MICROKIMI_ROUTECMS)
        for s in &sel {
            crate::cms::record(layer, s.0);
        }
        // routing history for the draft-aware prefetch (streaming runs only)
        if stream.is_some() {
            crate::stream::route_record((pos0 + t) as u32, layer as u32, sel.iter().map(|s| s.0).collect());
        }
    }
    let mut h = vec![0f32; n * cfg.routed_hidden];
    gemm_batch(Model::t(data, &w.routed_down), cfg.routed_hidden, cfg.d, x, n, &mut h);
    prof.t_router += tm.elapsed().as_secs_f64();

    // MXFP4 experts grouped by expert id: one job per used expert. Inside a
    // job the expert's (position, slot) pairs are evaluated as a batch with
    // matvec_packed_nt (packed blob read and dequantized once per element
    // for up to 8 positions), then scattered to their output slots.
    let tm = Instant::now();
    let expert_packed = cfg.routed_hidden * cfg.moe_inter / 2;
    let expert_blob = expert_packed + cfg.routed_hidden * cfg.moe_inter / 32;
    let expert_vq_blob = cfg.routed_hidden * cfg.moe_inter / crate::quant::VQ_DIM;
    let (erh, emi, topk) = (cfg.routed_hidden, cfg.moe_inter, cfg.top_k); // copies for the 'static closures
    let cbp = crate::pool::SPtr(w.vq_cb.as_ptr());
    let cblen = w.vq_cb.len();
    let mut by_expert: std::collections::HashMap<u32, Vec<(usize, usize)>> = std::collections::HashMap::new();
    for (t, sel) in sels.iter().enumerate() {
        for (slot, &(e, _)) in sel.iter().enumerate() {
            by_expert.entry(e).or_default().push((t, slot));
        }
    }
    let mut outs = vec![0f32; n * cfg.top_k * cfg.routed_hidden];
    match stream {
        // historical full-load path: expert blobs read straight from the file image
        None => {
            let dp = crate::pool::SPtrU8(data.as_ptr());
            let dlen = data.len();
            let hp = crate::pool::SPtr(h.as_ptr());
            let op = crate::pool::MPtr(outs.as_mut_ptr());
            let mut jobs: Vec<crate::pool::Job> = Vec::with_capacity(by_expert.len());
            for (e, pairs) in by_expert {
                let offs = w.experts[e as usize];
                let vq = w.experts_vq[e as usize];
                let eblob = if vq { expert_vq_blob } else { expert_blob };
                jobs.push(Box::new(move || {
                    let (dp, hp, op, cbp) = (dp, hp, op, cbp);
                    unsafe {
                        let data = std::slice::from_raw_parts(dp.0, dlen);
                        let blob = |i: usize| &data[offs[i] as usize..offs[i] as usize + eblob];
                        let m = pairs.len();
                        let m8 = m.next_multiple_of(8); // zero-padded lanes (outputs ignored)
                        // gather the inputs of this expert's pairs, transposed [erh][m8]
                        let mut ht = vec![0f32; m8 * erh];
                        for (i, (t, _)) in pairs.iter().enumerate() {
                            let h = std::slice::from_raw_parts(hp.0.add(t * erh), erh);
                            for c in 0..erh {
                                ht[c * m8 + i] = h[c];
                            }
                        }
                        let mut a = vec![0f32; m8 * emi];
                        let mut u = vec![0f32; m8 * emi];
                        if vq {
                            let cb = std::slice::from_raw_parts(cbp.0, cblen);
                            crate::quant::matvec_vq_nt(cb, blob(0), emi, erh, &ht, m8, &mut a);
                            crate::quant::matvec_vq_nt(cb, blob(2), emi, erh, &ht, m8, &mut u);
                        } else {
                            matvec_packed_nt(&blob(0)[..expert_packed], &blob(0)[expert_packed..], emi, erh, &ht, m8, &mut a);
                            matvec_packed_nt(&blob(2)[..expert_packed], &blob(2)[expert_packed..], emi, erh, &ht, m8, &mut u);
                        }
                        // SiTU, transposed [emi][m8]
                        let mut act_t = vec![0f32; m8 * emi];
                        for i in 0..m {
                            for j in 0..emi {
                                act_t[j * m8 + i] = situ(a[i * emi + j], u[i * emi + j]);
                            }
                        }
                        let mut o = vec![0f32; m8 * erh];
                        if vq {
                            let cb = std::slice::from_raw_parts(cbp.0, cblen);
                            crate::quant::matvec_vq_nt(cb, blob(1), erh, emi, &act_t, m8, &mut o);
                        } else {
                            matvec_packed_nt(&blob(1)[..expert_packed], &blob(1)[expert_packed..], erh, emi, &act_t, m8, &mut o);
                        }
                        for (i, (t, slot)) in pairs.iter().enumerate() {
                            let dst = std::slice::from_raw_parts_mut(op.0.add((t * topk + slot) * erh), erh);
                            dst.copy_from_slice(&o[i * erh..(i + 1) * erh]);
                        }
                    }
                }));
            }
            crate::pool::pool().run(jobs);
        }
        // --stream: one job per used expert, bytes pulled through the
        // three-tier cache before the same batched matvec sequence runs.
        Some(cache) => {
            let cp = cache as *const crate::stream::ExpertCache as usize;
            let hp = crate::pool::SPtr(h.as_ptr());
            let op = crate::pool::MPtr(outs.as_mut_ptr());
            let layer32 = layer as u32;
            // same offset-sorted submission as moe_forward (each job scatters
            // to its own (position, slot) outputs: order cannot leak into the
            // result)
            let mut order: Vec<(u32, Vec<(usize, usize)>)> = by_expert.into_iter().collect();
            if crate::stream::offset_sort() {
                order.sort_by_key(|(e, _)| w.experts[*e as usize][0]);
            }
            // fused run fetch, same as moe_forward: missing experts of this
            // layer land in the RAM LRU with one span read per file-adjacent
            // run, the compute jobs below then hit the cache
            let items: Vec<(u32, [u64; 3], usize)> = order
                .iter()
                .map(|(e, _)| (*e, w.experts[*e as usize], if w.experts_vq[*e as usize] { expert_vq_blob } else { expert_blob }))
                .collect();
            cache.warm_batch(layer32, &items);
            let mut jobs: Vec<crate::pool::Job> = Vec::with_capacity(order.len());
            for (e, pairs) in order {
                let offs = w.experts[e as usize];
                let vq = w.experts_vq[e as usize];
                let eblob = if vq { expert_vq_blob } else { expert_blob };
                jobs.push(Box::new(move || {
                    let (hp, op, cbp) = (hp, op, cbp);
                    unsafe {
                        let cache = &*(cp as *const crate::stream::ExpertCache);
                        let served = cache.get(layer32, e, offs, eblob);
                        // shadow fallback, same contract as moe_forward:
                        // Served::Shadow = VQ1 shadow bytes + shadow codebook
                        // (degraded); Served::Full = historical path.
                        let (bytes, vq, eb, cb): (&[u8], bool, usize, &[f32]) = match &served {
                            crate::stream::Served::Full(b) => (&b[..], vq, eblob, std::slice::from_raw_parts(cbp.0, cblen)),
                            crate::stream::Served::Shadow(s, off) => {
                                (&s.data[*off..*off + 3 * expert_vq_blob], true, expert_vq_blob, &s.cb[..])
                            }
                        };
                        let blob = |i: usize| &bytes[i * eb..(i + 1) * eb];
                        let m = pairs.len();
                        let m8 = m.next_multiple_of(8); // zero-padded lanes (outputs ignored)
                        // gather the inputs of this expert's pairs, transposed [erh][m8]
                        let mut ht = vec![0f32; m8 * erh];
                        for (i, (t, _)) in pairs.iter().enumerate() {
                            let h = std::slice::from_raw_parts(hp.0.add(t * erh), erh);
                            for c in 0..erh {
                                ht[c * m8 + i] = h[c];
                            }
                        }
                        let mut a = vec![0f32; m8 * emi];
                        let mut u = vec![0f32; m8 * emi];
                        if vq {
                            crate::quant::matvec_vq_nt(cb, blob(0), emi, erh, &ht, m8, &mut a);
                            crate::quant::matvec_vq_nt(cb, blob(2), emi, erh, &ht, m8, &mut u);
                        } else {
                            matvec_packed_nt(&blob(0)[..expert_packed], &blob(0)[expert_packed..], emi, erh, &ht, m8, &mut a);
                            matvec_packed_nt(&blob(2)[..expert_packed], &blob(2)[expert_packed..], emi, erh, &ht, m8, &mut u);
                        }
                        // SiTU, transposed [emi][m8]
                        let mut act_t = vec![0f32; m8 * emi];
                        for i in 0..m {
                            for j in 0..emi {
                                act_t[j * m8 + i] = situ(a[i * emi + j], u[i * emi + j]);
                            }
                        }
                        let mut o = vec![0f32; m8 * erh];
                        if vq {
                            crate::quant::matvec_vq_nt(cb, blob(1), erh, emi, &act_t, m8, &mut o);
                        } else {
                            matvec_packed_nt(&blob(1)[..expert_packed], &blob(1)[expert_packed..], erh, emi, &act_t, m8, &mut o);
                        }
                        for (i, (t, slot)) in pairs.iter().enumerate() {
                            let dst = std::slice::from_raw_parts_mut(op.0.add((t * topk + slot) * erh), erh);
                            dst.copy_from_slice(&o[i * erh..(i + 1) * erh]);
                        }
                    }
                }));
            }
            crate::pool::pool().run(jobs);
        }
    }
    // combination per position in slot order, norm BEFORE up-proj
    let mut yn = vec![0f32; n * cfg.routed_hidden];
    for (t, sel) in sels.iter().enumerate() {
        let mut y = vec![0f32; cfg.routed_hidden];
        for (slot, &(_, wi)) in sel.iter().enumerate() {
            for j in 0..cfg.routed_hidden {
                y[j] += wi * outs[(t * cfg.top_k + slot) * cfg.routed_hidden + j];
            }
        }
        rmsnorm(cfg, &y, Model::t(data, &w.routed_norm), &mut yn[t * cfg.routed_hidden..(t + 1) * cfg.routed_hidden]);
    }
    let mut out = vec![0f32; n * cfg.d];
    gemm_batch(Model::t(data, &w.routed_up), cfg.d, cfg.routed_hidden, &yn, n, &mut out);
    // shared experts (2): SiTU MLP on the pre-down input
    let mut sa = vec![0f32; n * cfg.shared_inter];
    let mut su = vec![0f32; n * cfg.shared_inter];
    gemm_batch(Model::t(data, &w.shared_gate), cfg.shared_inter, cfg.d, x, n, &mut sa);
    gemm_batch(Model::t(data, &w.shared_up), cfg.shared_inter, cfg.d, x, n, &mut su);
    let mut sact = vec![0f32; n * cfg.shared_inter];
    for j in 0..n * cfg.shared_inter {
        sact[j] = situ(sa[j], su[j]);
    }
    let mut sout = vec![0f32; n * cfg.d];
    gemm_batch(Model::t(data, &w.shared_down), cfg.d, cfg.shared_inter, &sact, n, &mut sout);
    if layer == 1 {
        for t in 0..n {
            let routed = out[t * cfg.d..(t + 1) * cfg.d].to_vec();
            let shared = sout[t * cfg.d..(t + 1) * cfg.d].to_vec();
            parity_rec(|d| {
                d.l1_routed.insert(pos0 + t, routed);
                d.l1_shared.insert(pos0 + t, shared);
            });
        }
    }
    for j in 0..n * cfg.d {
        out[j] += sout[j];
    }
    prof.t_experts += tm.elapsed().as_secs_f64();
    out
}

/// Batched dense MLP for prefill: `x` = n position rows [n * d], returns
/// [n * d]. Bit-identical to dense_forward per position.
pub(super) fn dense_prefill(cfg: &Config, data: &[u8], w: &DenseW, x: &[f32], n: usize, prof: &mut Prof) -> Vec<f32> {
    let tm = Instant::now();
    let mut a = vec![0f32; n * cfg.dense_inter];
    let mut u = vec![0f32; n * cfg.dense_inter];
    gemm_batch(Model::t(data, &w.gate), cfg.dense_inter, cfg.d, x, n, &mut a);
    gemm_batch(Model::t(data, &w.up), cfg.dense_inter, cfg.d, x, n, &mut u);
    let mut act = vec![0f32; n * cfg.dense_inter];
    for j in 0..n * cfg.dense_inter {
        act[j] = situ(a[j], u[j]);
    }
    let mut out = vec![0f32; n * cfg.d];
    gemm_batch(Model::t(data, &w.down), cfg.d, cfg.dense_inter, &act, n, &mut out);
    prof.t_experts += tm.elapsed().as_secs_f64();
    out
}

// ── parity dumps (thread-local, inactive during normal inference) ──
