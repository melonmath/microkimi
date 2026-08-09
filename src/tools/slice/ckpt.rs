// Crash-safe resume checkpoint (<out>.sliceckpt) (moved from slice.rs).

// ── crash-safe resume checkpoint (<out>.sliceckpt) ──
//
// The scoring phases (channel |w| sums, per-layer expert scale-energy) are
// the long part of a remote slice (tens of thousands of range requests, ~1h
// of silence) and used to be lost entirely on a spot-VM preemption. Every
// finished entry is appended to the sidecar and fsynced IMMEDIATELY (it
// survives kill -9); a rerun with the same parameters resumes from it.
//
// Format (text, one entry per line):
//   config <fnv1a-64 hex>     run parameters: model | kept layers | hidden | experts | merge
//   channels <c0,c1,...>      hidden keep-set (ascending)
//   experts <layer> <e0,e1,...>  keep-set of one scored MoE layer (ascending)
//   merge <layer> <c0,c1,...>   --merge-experts: cluster id of every old expert of one layer
//
// A `config` mismatch (different model/layers/hidden/experts/merge) discards the
// sidecar with a warning and starts fresh. Covered: channel keep-set and
// expert keep-sets (both the remote scale-energy branch and the local .bin
// Frobenius branch of expert_keep_sets), and the per-layer merge cluster
// assignments. NOT covered: the --cold-vq score
// map (expert_score_map restarts from scratch) and the write phase itself
// (it restarts from scratch too: it is much faster than the scoring phases
// and resumability would complicate the append-only .bin layout). The file
// is deleted on successful completion.

pub(super) struct SliceCkpt {
    path: std::path::PathBuf,
    file: std::sync::Mutex<std::fs::File>,
    pub(super) channels: Option<Vec<usize>>,
    pub(super) experts: std::collections::HashMap<usize, Vec<usize>>,
    pub(super) merges: std::collections::HashMap<usize, Vec<usize>>, // layer -> cluster id per old expert
}

/// FNV-1a 64 over the run-parameter string (no hash crates, std only).
pub(super) fn fnv1a(s: &str) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub(super) fn parse_csv(s: &str) -> Option<Vec<usize>> {
    s.split(',').map(|t| t.parse().ok()).collect()
}

pub(super) fn join_csv(v: &[usize]) -> String {
    v.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",")
}

impl SliceCkpt {
    /// Loads <out>.sliceckpt when present. A matching `config` line restores
    /// the recorded entries, anything else (missing or corrupt file, parameter
    /// mismatch) starts fresh with a warning.
    pub(super) fn open(out: &str, key: &str) -> SliceCkpt {
        use std::io::Write;
        let path = std::path::PathBuf::from(format!("{}.sliceckpt", out));
        let want = format!("{:016x}", fnv1a(key));
        let mut config_ok = false;
        let mut channels = None;
        let mut experts = std::collections::HashMap::new();
        let mut merges = std::collections::HashMap::new();
        let existed = path.is_file();
        if let Ok(bytes) = std::fs::read(&path) {
            // a kill -9 between the write and the fsync of the LAST entry can
            // leave a torn trailing line: only complete lines are parsed
            let text = String::from_utf8_lossy(&bytes);
            let body = match text.rsplit_once('\n') {
                Some((body, _)) => body,
                None => "",
            };
            for line in body.lines() {
                let mut it = line.split_whitespace();
                match it.next() {
                    Some("config") => config_ok = it.next() == Some(want.as_str()),
                    Some("channels") => channels = it.next().and_then(parse_csv),
                    Some("experts") => {
                        if let (Some(l), Some(set)) = (it.next().and_then(|s| s.parse().ok()), it.next().and_then(parse_csv)) {
                            experts.insert(l, set);
                        }
                    }
                    Some("merge") => {
                        if let (Some(l), Some(a)) = (it.next().and_then(|s| s.parse().ok()), it.next().and_then(parse_csv)) {
                            merges.insert(l, a);
                        }
                    }
                    _ => {}
                }
            }
        }
        if existed && !config_ok {
            println!("sliceckpt: ignoring {} (parameters differ), starting fresh", path.display());
            std::fs::remove_file(&path).ok();
            channels = None;
            experts = std::collections::HashMap::new();
            merges = std::collections::HashMap::new();
        }
        let n = experts.len() + merges.len() + channels.is_some() as usize;
        if config_ok && n > 0 {
            println!("sliceckpt: resumed {} ({} entries)", path.display(), n);
        }
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&path).unwrap();
        if f.metadata().unwrap().len() == 0 {
            f.write_all(format!("config {}\n", want).as_bytes()).unwrap();
            f.sync_data().unwrap();
        }
        SliceCkpt { path, file: std::sync::Mutex::new(f), channels, experts, merges }
    }

    /// Appends one entry and fsyncs it (every entry must survive kill -9).
    /// The whole line goes out in ONE write: a kill -9 lands either before
    /// or after it, never in the middle of a line.
    pub(super) fn record(&self, line: &str) {
        use std::io::Write;
        let mut f = self.file.lock().unwrap();
        f.write_all(format!("{}\n", line).as_bytes()).unwrap();
        f.sync_data().unwrap();
    }

    pub(super) fn record_channels(&self, ch: &[usize]) {
        self.record(&format!("channels {}", join_csv(ch)));
    }

    pub(super) fn record_experts(&self, layer: usize, set: &[usize]) {
        self.record(&format!("experts {} {}", layer, join_csv(set)));
    }

    /// --merge-experts: the cluster id of every old expert of one layer.
    pub(super) fn record_merge(&self, layer: usize, assign: &[usize]) {
        self.record(&format!("merge {} {}", layer, join_csv(assign)));
    }

    /// Successful completion: a finished .bin needs no checkpoint.
    pub(super) fn finish(self) {
        drop(self.file);
        if std::fs::remove_file(&self.path).is_ok() {
            println!("sliceckpt: {} removed (slice complete)", self.path.display());
        }
    }
}
