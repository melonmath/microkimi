// Standalone commands built on top of the engine: build/build_ds construct
// the .bin weight files, slice/slice_st shrink a model (local and remote
// safetensors), eval runs the eval harness, replay re-runs saved sessions.

pub mod build;
pub mod build_ds;
pub mod complete_batch;
pub mod convert_qwen;
pub mod eval;
pub mod parity;
pub mod replay;
pub mod selftest;
pub mod serve;
pub mod slice;
pub mod slice_qwen;
pub mod slice_st;
