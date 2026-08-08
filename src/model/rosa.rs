// rosa.rs - online suffix-automaton proposer for --spec-rosa.
//
// An unbounded-context, exact n-gram: an incrementally built suffix
// automaton (Ukkonen's online construction, O(1) amortized per token, O(n)
// states over the committed token stream). The prediction for the current
// context is the most frequent continuation of the LONGEST suffix of the
// context observed before, chained token by token through the automaton's
// transitions. Unlike the plain --spec n-gram (longest 2..=8 suffix, most
// recent occurrence), the context length is not bounded and continuations
// are frequency-ordered rather than recency-picked.
//
// Structure per state: DAWG transitions (token -> state), a suffix link
// (the longest proper suffix class, the fallback when a state has no
// trusted continuation), and successor statistics: a small map
// token -> (decayed count, stamp) of the tokens that followed this state
// when it was the active tip. Counts decay exponentially (lazy per-entry
// decay, same pattern as stream.rs's Markov predictor), so proposals
// follow recent repetition and cold history fades instead of needing
// structural pruning; states stay O(fed tokens), fine for decode-time
// contexts.

use std::collections::HashMap;

/// Halflife of the successor-count decay, in fed tokens.
const DECAY_HALFLIFE: f32 = 256.0;
/// Successor entries weaker than this after decay are ignored: a
/// continuation must have been seen recently enough to be trusted.
const MIN_SCORE: f32 = 0.25;

struct State {
    next: HashMap<u32, usize>,      // DAWG transition token -> state
    link: usize,                    // suffix link (NONE for the root's)
    len: usize,                     // longest string in this equivalence class
    succ: HashMap<u32, (f32, u64)>, // continuation token -> (decayed count, stamp)
}

const ROOT: usize = 0;
const NONE: usize = usize::MAX;

pub struct SuffixAutomaton {
    st: Vec<State>,
    last: usize,
    epoch: u64, // tokens fed so far
}

impl SuffixAutomaton {
    pub fn new() -> SuffixAutomaton {
        SuffixAutomaton {
            st: vec![State { next: HashMap::new(), link: NONE, len: 0, succ: HashMap::new() }],
            last: ROOT,
            epoch: 0,
        }
    }

    /// Decay factor of a count stamped `stamp` read at the current epoch.
    fn decay(&self, stamp: u64) -> f32 {
        (-(self.epoch.saturating_sub(stamp) as f32) / DECAY_HALFLIFE).exp2()
    }

    /// Effective (decayed) score of a successor entry.
    fn score(&self, e: &(f32, u64)) -> f32 {
        e.0 * self.decay(e.1)
    }

    /// Adds `incr` to the successor entry (state s, token c), lazily decayed.
    fn bump(&mut self, s: usize, c: u32, incr: f32) {
        let epoch = self.epoch;
        let e = self.st[s].succ.entry(c).or_insert((0.0, epoch));
        let d = (-(epoch.saturating_sub(e.1) as f32) / DECAY_HALFLIFE).exp2();
        e.0 = e.0 * d + incr;
        e.1 = epoch;
    }

    /// Ukkonen's online extension with token c (O(1) amortized).
    pub fn feed(&mut self, c: u32) {
        let prev_last = self.last;
        self.epoch += 1;
        let cur = self.st.len();
        let len = self.st[prev_last].len + 1;
        self.st.push(State { next: HashMap::new(), link: ROOT, len, succ: HashMap::new() });
        let mut p = prev_last;
        while p != NONE && !self.st[p].next.contains_key(&c) {
            self.st[p].next.insert(c, cur);
            p = self.st[p].link;
        }
        if p == NONE {
            self.st[cur].link = ROOT;
        } else {
            let q = self.st[p].next[&c];
            if self.st[p].len + 1 == self.st[q].len {
                self.st[cur].link = q;
            } else {
                // split q: the clone takes the shorter end of its class
                let clone = self.st.len();
                let (qnext, qlink) = (self.st[q].next.clone(), self.st[q].link);
                let clen = self.st[p].len + 1;
                self.st.push(State { next: qnext, link: qlink, len: clen, succ: HashMap::new() });
                while p != NONE && self.st[p].next.get(&c) == Some(&q) {
                    self.st[p].next.insert(c, clone);
                    p = self.st[p].link;
                }
                self.st[q].link = clone;
                self.st[cur].link = clone;
            }
        }
        self.last = cur;
        // successor statistics: token c followed the previous tip state
        self.bump(prev_last, c, 1.0);
    }

    /// The best continuation chain of up to `n` tokens from the current
    /// context: from the longest suffix class with a trusted successor
    /// (suffix-link fallback toward shorter suffixes), repeatedly take the
    /// most frequent continuation (decay-weighted) and follow its DAWG
    /// transition to continue the chain.
    pub fn propose(&self, n: usize) -> Vec<u32> {
        let mut out = Vec::new();
        let mut s = self.last;
        for _ in 0..n {
            let mut pick: Option<(u32, usize)> = None;
            let mut cur = s;
            loop {
                // ROOT's succ only records the stream's first token (it is
                // the tip before anything is fed): a unigram, not a
                // continuation of the current suffix. Never trusted.
                if cur == ROOT {
                    break;
                }
                let best = self.st[cur]
                    .succ
                    .iter()
                    .map(|(&c, e)| (c, self.score(e)))
                    .filter(|&(_, sc)| sc >= MIN_SCORE)
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                if let Some((c, _)) = best {
                    // the DAWG transition for c always exists (the count was fed through it)
                    if let Some(&nx) = self.st[cur].next.get(&c) {
                        pick = Some((c, nx));
                    }
                    break;
                }
                cur = self.st[cur].link;
            }
            let Some((c, nx)) = pick else { break };
            out.push(c);
            s = nx;
        }
        out
    }
}

#[cfg(test)]
mod rosa_tests {
    use super::*;

    fn fed(s: &str) -> SuffixAutomaton {
        // map chars to token ids for readability
        let mut a = SuffixAutomaton::new();
        for c in s.bytes() {
            a.feed(c as u32);
        }
        a
    }

    fn as_string(v: &[u32]) -> String {
        v.iter().map(|&c| (c as u8) as char).collect()
    }

    #[test]
    fn periodic_text_continues_period() {
        let a = fed("ab cd ab cd ab cd ");
        // longest repeated suffix is "ab cd " -> next tokens of the period
        assert_eq!(as_string(&a.propose(6)), "ab cd ");
    }

    #[test]
    fn frequency_orders_proposals() {
        // "xa" appears 3 times, "xb" once: the continuation of "x" is "a"
        let a = fed("xaxaxaxb x");
        assert_eq!(as_string(&a.propose(1)), "a");
    }

    #[test]
    fn unseen_context_proposes_nothing() {
        let a = fed("abc");
        assert!(a.propose(4).is_empty());
    }

    #[test]
    fn chain_follows_transitions() {
        let a = fed("the quick brown fox the quick brown fox the q");
        // the chain should follow "uick brown fox" greedily
        assert_eq!(as_string(&a.propose(5)), "uick ");
    }
}
