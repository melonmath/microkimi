// Standalone commands built on top of the engine: build/build_ds construct
// the .bin weight files, slice/slice_st shrink a model (local and remote
// safetensors), eval runs the eval harness, replay re-runs saved sessions.

pub mod build;
pub mod build_ds;
pub mod eval;
pub mod parity;
pub mod replay;
pub mod selftest;
pub mod slice;
pub mod slice_st;
