//! # `dct4-mdct-kit`
//!
//! An **indirect** DCT-IV kit: it computes the length-`N` DCT-IV *through* an external
//! length-`2N -> N` MDCT backend, without implementing an MDCT core.
//!
//! The crate is the executable form of the design in *"Indirect Computation of the DCT-IV
//! via an External MDCT Backend: A Unified Mathematical Framework, GPU Adaptation, and
//! Compliance Verification"*. It is deliberately organised as a **specification object**:
//! every claim in the paper is represented by a type, a check, or a proof-obligation
//! encoded as a test.
//!
//! ```text
//!            FOLD (inverse folding)      MDCT IP           POST
//!   U  ──►  P · u  (permutation,  ──►  α · M_{2N->N}  ──►  σ/α, saturation,
//!           sign, zero-fill)                               optional permutation  ──► X
//! ```
//!
//! ## The three stages
//!
//! | Stage | Theory | Module |
//! |-------|--------|--------|
//! | `FOLD` — realises `F^{-1}`, permutation + sign inversion + zero filling, no rounding | Thm. 2, Cor. 1 | [`folding`] |
//! | `MDCT IP` — external backend, realises `M_ip = α·M_B` | Thm. 1, §4.2 | [`backend`] |
//! | `POST` — scaling, sign, saturation, optional permutation | (2) | [`pipeline`] |
//!
//! ## Why it is correct
//!
//! The folding equivalence theorem ([`folding::fold`]) states
//! `M_{2N->N}(x) = D_N(F(x))`. The inverse map of Thm. 2 gives, for `u = (u_L, u_U)` with
//! blocks of length `N/2`,
//!
//! ```text
//! x = F^{-1}(u) = (u_U, 0, 0, -u_L)
//! ```
//!
//! Substituting into `F` yields `F(x) = (-0 - (-u_L), u_U - 0) = u`, hence
//! `M_{2N->N}(F^{-1}(u)) = D_N(u)`. Only `N` of the `2N` backend input samples are non-zero:
//! the kit trades `2x` arithmetic redundancy for backend reuse and, critically, performs
//! **no arithmetic rounding in the folding stage** (`||P||_2 = 1`, verified by
//! [`folding::InverseFoldMatrix::is_isometry`]).
//!
//! ## Hard preconditions H1–H4
//!
//! A backend is only connected if it is window-free (H1), phase-matched (H2), uniformly
//! scaled (H3), and naturally ordered or permutable (H4). [`compliance::evaluate`] decides
//! these by probing the backend; [`compliance::ComplianceReport::gate`] refuses service
//! (`backend_reject`) otherwise. [`pipeline::DctIvKit::new`] wires the gate in front of the
//! pipeline, so an inadmissible backend can never be used by accident.
//!
//! ## GPU adaptation
//!
//! [`gpu`] carries the closed-form model of the paper: buffer pooling with `K` slots,
//! the stream partial order as a DAG, and throughput/latency projections that are
//! cross-validated against an event-driven simulation.
//!
//! ## Example
//!
//! ```
//! use dct4_mdct_kit::backend::EmulatedMdct;
//! use dct4_mdct_kit::config::KitConfig;
//! use dct4_mdct_kit::dct4::dct4;
//! use dct4_mdct_kit::pipeline::{DctIvKit, GatePolicy};
//!
//! let cfg = KitConfig::new(32);
//! let backend = EmulatedMdct::ideal("golden-mdct-ip");
//! let mut kit = DctIvKit::new(cfg, backend, GatePolicy::Enforce, None).unwrap();
//!
//! let u: Vec<f64> = (0..32).map(|i| ((i * 7 + 3) as f64).sin()).collect();
//! let mut x = vec![0.0; 32];
//! kit.transform_block(&u, &mut x).unwrap();
//!
//! let reference = dct4(&u);
//! let err = reference.iter().zip(&x).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
//! assert!(err < 1e-12, "indirect DCT-IV must match the direct definition, err = {err}");
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod backend;
pub mod compliance;
pub mod config;
pub mod dct4;
pub mod error;
pub mod fixed;
pub mod folding;
pub mod gpu;
pub mod mdct;
pub mod numerics;
pub mod pipeline;
pub mod util;

pub use backend::BackendError;
pub use error::KitError;
