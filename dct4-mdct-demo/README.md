# dct4-mdct-kit

**Indirect computation of the DCT-IV through an external MDCT backend.**

A Rust implementation of the design in *"Indirect Computation of the DCT-IV via an External MDCT
Backend: A Unified Mathematical Framework, GPU Adaptation, and Compliance Verification"*.

The kit computes a length-`N` DCT-IV **without implementing an MDCT core**. It drives an external
`2N → N` MDCT engine through a three-stage pipeline, and it refuses to run unless the engine
satisfies four hard preconditions.

```text
              FOLD                      MDCT IP                POST
 U  ──►  x = F⁻¹(u)  ──►  y = α·M₂ₙ→ₙ(x)  ──►  X = (σ/α)·y, saturation,
 (N)     permutation,        external          optional permutation   ──► X
         sign, zero-fill     backend                   (N)
         (2N)                 (N)
```

Only `N` of the `2N` samples handed to the backend are non-zero: the kit trades **2× arithmetic
redundancy for backend reuse**, and pays for it with one extra compliance gate and one extra
post-processing stage — not with a new transform core.

---

## 1. Why it is correct

### Theorem 1 — folding equivalence (§3.4)

For `x ∈ R²ᴺ` split into four blocks `(a, b, c, d)` of length `h = N/2`, with `vᴿ[q] = v[h-1-q]`:

```text
F(x) = ( -cᴿ - d ,  a - bᴿ )                              (length N)
M₂ₙ→ₙ(x) = D_N(F(x))
```

### Theorem 2 — inverse folding (§3.5)

For `u = (u_L, u_U)` of length `N`, each half of length `N/2`:

```text
x = F⁻¹(u) = ( u_U , 0 , 0 , -u_L )                       (length 2N)

F(F⁻¹(u)) = ( -0 - (-u_L) , u_U - 0 ) = (u_L, u_U) = u
```

Therefore

```text
M₂ₙ→ₙ(F⁻¹(u)) = D_N(F(F⁻¹(u))) = D_N(u)
```

### Corollary 1 — the map is a sparse isometry

`F⁻¹` is the matrix `P ∈ {0,±1}²ᴺˣᴺ` with `Pᵀ P = I` and `‖P‖₂ = 1`. It contains **no
multiplication and no division**, so it introduces **no arithmetic rounding** — on `f64` and,
more importantly, on raw fixed-point codes.

In this crate:

| Claim | Code | Verified by |
|---|---|---|
| Theorem 1 | [`folding::fold`], [`mdct::MdctKernel`] | `tests/folding_theorem.rs::theorem1_folding_equivalence` |
| Theorem 2 | [`folding::inverse_fold`] | `…::theorem2_inverse_folding` (exact: error `== 0.0`) |
| Corollary 1 | [`folding::InverseFoldMatrix`] | `…::corollary1_sparse_operator` (`PᵀP = I`, `nnz = N`) |
| Eq. 2 chain | [`pipeline::DctIvKit`] | `tests/pipeline_end_to_end.rs` |

---

## 2. The backend contract H1–H4 (§4.2, §8)

The kit interfaces only with backends that satisfy all four preconditions. Each one is decided
by a numeric probe, not by a vendor declaration:

| Id | Precondition | Decidable criterion | Method |
|----|--------------|---------------------|--------|
| **H1** | window-free impulse response `w_ip ≡ 1` | For every input index `m`, the gain of impulse column `m`, normalised by the median gain, deviates by at most `window_tol`. | `probe_impulse` × `2N` |
| **H2** | phase-matched | On a small `N`: `max_k |y[k] − α̂·ref[k]| / (1 + max|ref|) ≤ ε_phase`, over deterministic probe vectors. | `probe_vector` |
| **H3** | scalar scaling | `spread(α) = (max α_k − min α_k) / mean|α_k| ≤ τ_dtype` **and** measured, declared and configured `α` agree. | `probe_vector` |
| **H4** | natural ordering | Order is natural, or the declared permutation is a valid bijection (POST inverts it). | descriptor + all probes |

Any failure → `backend_reject`. [`compliance::evaluate`] returns a
[`compliance::ComplianceReport`] with one [`compliance::CheckOutcome`] per precondition, the
measured statistics, and one [`compliance::Violation`] per failure carrying a
[`compliance::Vulnerability`] classification and a concrete remedy.

**Ordering is decided first.** Without a decidable bin order every per-bin statistic is
meaningless, so H1–H3 *fail closed* instead of reporting noise as a defect — a deliberate
security posture, not an accident.

```text
$ cargo run --example compliance_sheet

### sine-windowed IP
    H1 window-free impulse response FAIL  metric = 8.607e-1
    H2 phase-matched              FAIL  metric = 3.623e-1
    H3 scalar scaling             FAIL  metric = 8.713e-1
    H4 natural ordering           pass  metric = 0.000e0
    => REJECT (backend_reject)
       - WindowContamination: implied window deviates by 8.607e-1 …
         remedy: insert an inverse-taper stage before FOLD, or regenerate the IP from a
                 window-free core; a per-sample window is not a scalar and cannot be undone in POST
```

### `τ_dtype`

The acceptability threshold of a backend format is

```text
τ_dtype = κ · √n · ε ,     κ = 8
```

where `ε` is the format's relative precision: `2⁻²⁴` (f32), `2⁻⁵³` (f64), `2⁻ᵍ` (fixed `Q` with
`q` fractional bits). A probe of `n` outputs accumulates `~√n·ε` of relative uncertainty, and `κ`
keeps a compliant backend on the safe side. The practical consequence: **the same 12-bit
truncation is acceptable for an IP that declares `Q1.12` and a contract violation for one that
declares `f64`** — `tests/compliance_gate.rs::precision_is_judged_against_the_declared_format`
pins this down.

See [`numerics`] for the derivation and [`compliance::Thresholds`] for the overridable knobs.

---

## 3. Configuration validity (eq. 3)

```text
valid ⟺ (N mod 2 = 0) ∧ (N ∈ N_ip) ∧ (Win ≥ Wu + 1) ∧ (Wout ≥ Wip)
```

The plus-one is the **guard bit of the fold**: `F` sums two words (`−cᴿ − d` and `a − bᴿ`), so a
`Wu`-wide folded word can overflow even when every input sample is representable.
`folding::fold_is_exact` demonstrates the failure and the fix:

```rust
let wu  = QFormat::new(16, 14);       // Q1.14
assert!(!fold_is_exact(16, wu, wu));                 // overflows
assert!( fold_is_exact(16, wu, QFormat::new(17, 14))); // guard bit
```

Beyond eq. 3 the validator also rejects: fractional-bit disagreement between `Wu` and `Win`
(the fold is exact only if the fractional point does not move), non-finite or non-positive `α`,
`σ ∉ {+1,−1}`, an empty batch, and a zero-slot pool. All violations are collected, not
fail-fast, so a bring-up sees the full list at once.

---

## 4. Two arithmetic paths

| Path | API | FOLD | POST |
|---|---|---|---|
| Analysis | `transform_block` / `transform_batch` | `f64`, exact | `f64` scalar + optional permutation |
| Production | `transform_block_fixed` / `transform_batch_fixed` | raw `i64` codes, **exact** | one `i128` rational multiply, one fractional-point move, saturation |

The fixed path keeps the paper's central promise: **the folding stage performs no rounding**.
The only rounding in the kit is the single post-stage requantization, with deterministic
round-half-away-from-zero, and every clamp is counted.

Three counters turn silent corruption into a test failure:

| Counter | Non-zero means | Corrective action |
|---|---|---|
| `fold_overflows` | a folded word left `Win` | the guard bit is missing — fix eq. 3 |
| `ip_clamps` | the backend word left `Wip` | `Wip` does not cover `α·X` — widen it (`numerics::required_ip_width`) |
| `saturations` | the post stage clamped | `Wout` is too narrow for `σ/α` — widen it or lower `α` |

> **Dynamic-range note.** The backend output is `y = α·X`, so `Wip` must cover the *scaled*
> range. A 24-bit `Q9.14` IP with `α = 2¹⁵` cannot hold 16-bit audio: that is 15 integer bits
> short. `numerics::required_ip_width(α, max|X|, frac)` gives the minimum width, and
> `DctIvKit::ip_clamps` reports violations at runtime instead of letting them pass silently.

---

## 5. GPU adaptation (§6)

On a GPU nothing about the mathematics changes; the *scheduling* does. The crate models:

```text
per block:  FOLD → MDCT → POST        (2 pool slots held for the whole traversal)
per stream: blocks run strictly in order
in flight:  c = min(M, ⌊K/2⌋, B)
```

**Closed form** (`gpu::project`) — conservative, see the module docs:

```text
L  = t_f + t_m + t_p                      per-block latency
P  = max(t_m, L/c)                        steady-state period
makespan = max( L ,  t_f + B·t_m + t_p ,  L + (B−c)⁺·P  ,  staircase )
throughput = B / makespan
bytes/block = 8N · sizeof(sample)         FOLD 3N + MDCT 3N + POST 2N
```

The four terms exist because there are four regimes: a single block, an unbuffered MDCT-bound
queue, a saturated buffer pipeline, and a wave-quantised pool. A model that used only one of
them would under-promise a deadline.

**Event-driven replay** (`gpu::simulate`) reproduces the same plan exactly: one shared MDCT
unit, unlimited-lane FOLD/POST, greedy in-order issue. The test suite asserts
`projection ≥ replay` over a grid of plans, and `gap ≤ 2·latency` — the closed form is a safe
design-time bound, and the replay is the reference.

**Stream partial order** (`gpu::DependencyGraph`) is materialised as a DAG so it can be reviewed
or handed to a scheduler:

```text
FOLD(b) → MDCT(b) → POST(b)              stage order
FOLD(b) → FOLD(b + M)                    same stream serialises
POST(b) → FOLD(b + c)                    buffer reuse
```

`critical_path` gives the resource-free lower bound on the makespan, and `to_dot()` exports the
graph for a design review.

```text
$ cargo run --example gpu_plan
  M    K     c  latency us   period us makespan us   blocks/s      util
  1    2     1        9.11        9.11     2331.66     109793    100.0%
  4    6     3        9.11        3.04      785.34     325975     99.0%
  8    8     4        9.11        3.04      785.34     325975     74.2%
```

---

## 6. Error model (§7, §9)

`numerics::forward_error_bound` decomposes the output error into three terms:

```text
|ΔX| ≤ N·2^-(q_in+1)                                    input word (worst-case L1)
     + |σ/α| · γ₂ₙ · √(2N) · ‖u‖₂                       backend arithmetic
     + 2^-(q_out+1)                                     post rounding
```

with `γₙ = n·ε / (1 − n·ε)`. `‖X‖₂ = √(N/2)·‖u‖₂` exactly (because `D_NᵀD_N = (N/2)I`), which is
what the relative bound divides by. `numerics::budget` compares the backend's contribution with
the output half-LSB and reports a `PrecisionInsufficiency` violation when the declared format
cannot resolve the configured output word.

The bound is pessimistic by construction, and the tests assert the *measured* error stays below
it (`tests/pipeline_end_to_end.rs::fixed_point_path_meets_the_forward_error_bound`).

---

## 7. Quick start

```toml
[dependencies]
dct4-mdct-kit = { path = "dct4-mdct-kit" }
```

```rust
use dct4_mdct_kit::backend::EmulatedMdct;
use dct4_mdct_kit::config::{KitConfig, LengthSet};
use dct4_mdct_kit::pipeline::{DctIvKit, GatePolicy};

// Eq. 3: N even and supported, one guard bit, Wout >= Wip.
let cfg = KitConfig::new(64)
    .with_alpha(1024.0)                       // the IP realises y = 1024·M(x)
    .with_lengths(LengthSet::of(&[32, 64, 128]));

let backend = EmulatedMdct::ideal("acme-mdct-ip")
    .with_scale(1024.0)
    .with_lengths(LengthSet::of(&[32, 64, 128]));

// Probes H1–H4; returns KitError::BackendRejected on any failure.
let mut kit = DctIvKit::new(cfg, backend, GatePolicy::Enforce, None)?;

let u = vec![0.25; 64];
let x = kit.transform(&u)?;                   // X = D_64(u)

println!("{}", kit.report().to_markdown());   // the verification record
```

Running the examples:

```bash
cargo run --release --example end_to_end       # both arithmetic paths + compliance record
cargo run --release --example compliance_sheet # accept/reject decision sheet with remedies
cargo run --release --example gpu_plan         # projection vs replay, DOT export
cargo test                                     # 134 checks: unit, integration, doc
```

### Build environment

The crate has **zero dependencies**, so any stable Rust `>= 1.70` builds it:

```bash
cargo build --release && cargo test
```

The toolchain used for this checkout was installed entirely inside the workspace
(`.toolchain/`), because the machine's `~/.cargo/bin` shims had no toolchain installed. To use
that one instead of a system toolchain:

```bash
export RUSTUP_HOME=/Users/kodlien/Documents/1/.toolchain/rustup
export CARGO_HOME=/Users/kodlien/Documents/1/.toolchain/cargo
cargo test
```

Both `rustfmt` and `clippy` components are installed in that toolchain; `cargo fmt` and
`cargo clippy --all-targets` are clean.

---

## 8. Project layout

| Path | Contents |
|---|---|
| `src/folding.rs` | `F`, `F⁻¹`, the sparse `P`, the guard-bit probe |
| `src/dct4.rs`, `src/mdct.rs` | the analytic definitions — verification oracles, not fast transforms |
| `src/backend/` | `MdctBackend` trait, descriptor, trusted reference backend, defect-injecting emulator |
| `src/compliance.rs` | H1–H4 decision procedure, thresholds, violations, remedies, Markdown record |
| `src/config.rs` | `KitConfig`, `N_ip`, eq. 3 |
| `src/fixed.rs` | `QFormat`, `Q`, saturating operators, deterministic rounding |
| `src/pipeline.rs` | `DctIvKit`: fold → backend → post, float and fixed paths, counters, budget check |
| `src/numerics.rs` | `τ_dtype`, forward error bound, `Wip` sizing, SNR |
| `src/gpu/` | buffer pool, stream partial order + replay, closed-form projection |
| `tests/` | the theorem, the pipeline, the gate, the configuration, the GPU model |
| `examples/` | runnable acceptance artefacts |
| `docs/DESIGN.md` | paper→code mapping, derivations, assumptions, deviations |

---

## 9. Verification summary

| Area | Checks |
|---|---|
| Theorems 1, 2, Corollary 1, eq. 2 | 8 integration tests over `N ∈ {4…128}`, plus exact integer round-trips |
| End-to-end DCT-IV | 10 integration tests: definition match, batch, α/σ absorption, permutation, fixed path vs bound, saturation, guard bit, length roster |
| Compliance gate | 14 integration tests: every H1–H4 rejection path, fail-closed ordering, `τ_dtype` scale argument, report contents |
| Configuration | 12 integration tests: each eq. 3 clause, truth table, structural rules |
| GPU model | 13 integration tests: capacity law, projection ≥ replay over a grid, DAG structure, critical path, bandwidth account |
| Unit tests | 76 across modules (fixed-point, kernels, probes, pool, schedule, model) |
| Doc tests | 1 (the README example, compiled) |

`cargo clippy --all-targets` is clean; `cargo fmt` is applied; the crate forbids `unsafe`.

---

## 10. Design decisions worth knowing

1. **The MDCT backend is modelled as a single shared unit** in the GPU schedule. The reusable IP
   is the scarce resource; FOLD and POST are elementwise and assumed to have enough lanes.
   This is stated in `gpu::stream` and is the assumption that makes the closed form exact in the
   MDCT-bound regime.
2. **H1–H3 fail closed** when the output ordering is undecidable, rather than reporting a
   meaningless pass.
3. **The guard-bit contract has one residual corner.** `c = d = −2^(W−1)` folds to `+2^W`, one
   code past the `(W+1)`-bit maximum. The kit counts such words as `fold_overflows` instead of
   letting them wrap, and the doc on `folding::fold_is_exact` explains when `Wu+2`
   is the right choice.
4. **`Wip` must cover `α·X`**, not `X`; `ip_clamps` exists because an under-sized backend word is
   otherwise invisible until the output is wrong.
5. **`GatePolicy::Bypass` still probes and still records the violations.** It exists for
   bring-up and marks the kit `is_enforced() == false`; its output must not ship.
6. **The analytic DCT-IV/MDCT kernels in `dct4`/`mdct` are oracles**, used by the compliance
   probe and the test suite. The production path never calls them.

---

## 11. Provenance

The technical content implemented here is §1–§4.3 of the source PDF: the folding theorems, the
three-stage pipeline, the backend contract H1–H4, and the validity constraint of eq. 3. The PDF's
remaining pages contain an unrelated short-story fragment; it is not part of the specification and
is not represented in the code.

## 12. License

MIT — see [LICENSE](LICENSE).
