# Design notes: from the paper to the code

This document maps each part of the source design onto a module, records the derivations that the
code relies on, and lists the assumptions and deliberate deviations. The README is the tour; this
file is the engineering record.

---

## 1. Symbol table

| Paper | Code |
|---|---|
| `N` | `KitConfig::n` |
| `B` | `KitConfig::batch` |
| `U ∈ R^{B×N}` | `transform_batch(u, x)` argument `u` |
| `X ∈ R^{B×N}` | `transform_batch(u, x)` argument `x` |
| `Wu, Win, Wip, Wout` | `KitConfig::{input, guard, ip, output}` as `QFormat` (width includes sign) |
| `Q` | `QFormat::frac` |
| `S, σ` | `KitConfig::post_scale`, `KitConfig::sign` |
| `α` | `KitConfig::alpha` |
| `K` | `KitConfig::pool_slots`, `StreamPlan::slots` |
| `N_ip` | `KitConfig::lengths` / `BackendDescriptor::supported_lengths` (`LengthSet`) |

`QFormat::width` counts the sign bit, so `Q1.14` is `QFormat::new(16, 14)` and its integer range is
`[-2, 2)`. This is why clause 3 of eq. 3 is `Win >= Wu + 1` rather than `Win >= Wu`.

---

## 2. Section-by-section mapping

| Paper section | Module | Note |
|---|---|---|
| §3.2 DCT-IV and MDCT definitions | `dct4`, `mdct` | `O(N²)` analytic kernels; verification oracles |
| §3.3 Folding operator `F` | `folding::fold` | generic over `Scalar`, so the same code runs on `f64` and `i64` |
| §3.4 Theorem 1 | `folding` + `mdct` | tested for `N ∈ {4…128}`, 8 random blocks each |
| §3.5 Theorem 2, `F⁻¹` | `folding::inverse_fold` | `(u_U, 0, 0, −u_L)`; exact on integers |
| §3.5 Corollary 1, `P` | `folding::InverseFoldMatrix` | `nnz = N`, `PᵀP = I`, `‖P‖₂ = 1` |
| §3.6 Eq. 2, complete chain | `pipeline::DctIvKit` | `POST ∘ MDCT ∘ FOLD` |
| §4.1 Three-stage pipeline | `pipeline` | FOLD / MDCT IP / POST |
| §4.2 Contract H1–H4 | `compliance` | probes, thresholds, violations, remedies |
| §4.3 Eq. 3 validity | `config::KitConfig::validate` | plus structural rules the paper leaves implicit |
| §6 Buffer pooling, stream order | `gpu::pool`, `gpu::stream` | logical-time pool, DAG, exact replay |
| §6 Throughput, latency | `gpu::model` | closed form, conservative, cross-checked |
| §7, §9 Error model, `τ_dtype` | `numerics` | forward bound, budget, SNR |
| §8 Compliance verification | `compliance`, `DctIvKit::new` | `backend_reject` at construction |
| §10 Vulnerabilities and remedies | `compliance::Vulnerability` | one variant per failure mode |

---

## 3. Derivations used by the code

### 3.1 Theorem 1 in the code's own terms

`F` and the MDCT share the same cosine kernel. With `θ(m,k) = (π/N)(m + 1/2 + N/2)(k + 1/2)`:

```text
M(x)[k] = Σ_{m=0}^{2N−1} x[m] cos θ(m,k)
```

Splitting at `N/2, N, 3N/2` and substituting `m' = 2N − 1 − m` in the last two segments uses

```text
cos θ(2N − 1 − m', k) = cos( π(k+1/2) − θ(m' − N/2, k) )
```

which reduces the `2N`-term sum to `N` terms of `D_N` acting on `(−cᴿ − d, a − bᴿ)`. The code does
not reproduce the proof symbolically; it checks the identity numerically for `N ∈ {4…128}` at
`1e-11` and relies on the closed-form `F`/`F⁻¹` pair for the pipeline.

### 3.2 Why `F⁻¹` is exact

`F⁻¹` writes `N` inputs into `2N` slots with signs and a zero block, and `F` reads each slot at
most once with a sign. There is no addition of two non-zero values, no multiplication and no
division, so on integers it is a bijection on `{0,±1}` coefficients. The only failure mode is
*overflow of the destination word*, which is exactly what the guard bit and the
`fold_overflows` counter address.

A useful consequence, used by the error model: `‖F⁻¹(u)‖₂ = ‖u‖₂`, which is Corollary 1 stated in
norms, and what lets the forward bound use `‖u‖₂` rather than `‖x‖₂`.

### 3.3 The forward error bound

`dct4::Dct4Kernel` implements `D` with `DᵀD = (N/2)I` (checked as an involution in the module
tests: `D(D(u)) = (N/2)u`). Hence `‖X‖₂ = √(N/2)·‖u‖₂` exactly, which gives a meaningful
denominator for a relative bound.

Three terms, all worst case:

```text
input word      |ΔX[k]| ≤ Σ_n |Δu[n]| ≤ N · 2^−(q_in+1)
backend         |ΔY[k]| ≤ γ_{2N} · Σ_m |x[m]| ≤ γ_{2N} · √(2N) · ‖x‖₂ = γ_{2N} · √(2N) · ‖u‖₂
post            |Δ| ≤ 2^−(q_out+1)
```

scaled by `|S| = 1/α` where the backend term enters the output domain. `γ_n = nε/(1 − nε)`.
The `√(2N)` factor is Cauchy–Schwarz on the kernel row; the `N` factor on the input term is the
`L1` bound. Both are intentionally loose: a bound that occasionally overestimates is usable, one
that occasionally underestimates is not.

### 3.4 `τ_dtype`

```
τ_dtype = κ √n ε ,  κ = 8
```

`ε` is `SampleFormat::eps`: `2⁻²⁴`, `2⁻⁵³`, `2⁻ᵍ` or `2⁻³¹` for `i32`. The `√n` reflects the
accumulation of round-off in an inner product of `n` terms; `κ` is a safety factor that keeps a
compliant backend strictly inside the comparison. The threshold is used for three things:

* H3's `spread(α) ≤ τ_dtype`;
* H2's `ε_phase` (`τ_dtype` at `2n`);
* H1's `window_tol` (`τ_dtype` at `2n`).

Consequences, deliberate:

* an `f64` IP cannot hide a 1 % per-bin gain error behind a "close enough" tolerance;
* a `Q1.12` IP is judged at `2⁻¹²`, so honest truncation is accepted;
* the same truncation on an IP that declares `f64` is rejected (test:
  `precision_is_judged_against_the_declared_format`).

---

## 4. Compliance decision procedure

```text
evaluate(backend, cfg):
    probe_n ← min(cfg.n, 8) rounded to even, unless a permutation is declared → cfg.n
    thresholds ← for_format(backend.format, probe_n)

    H4: ordering valid bijection over cfg.n?
        no  → H4 fails; H1–H3 fail closed; return

    normalize(y) ← inverse permutation if declared, identity otherwise

    H1: for m in 0..2·probe_n:
            y_m ← normalize(probe_impulse(m))
            g_m ← <y_m, a_m> / <a_m, a_m>
        w_m ← |g_m| / median(|g|)
        deviation ← max |w_m − 1|
        pass iff deviation ≤ window_tol

    H2/H3: for each of `probe_vectors` deterministic inputs x:
            ref ← M_{2n→n}(x)      (analytic)
            y   ← normalize(probe_vector(x))
            α̂   ← <y, ref>/<ref, ref>
            phase_error ← max |y − α̂·ref| / (1 + α̂·max|ref|)
            α_k ← y[k]/ref[k]  for |ref[k]| ≥ 0.25·max|ref|
        spread ← (max α_k − min α_k) / mean|α_k|
        H2 pass iff phase_error ≤ ε_phase
        H3 pass iff spread ≤ τ_dtype ∧ |α̂/α| ≈ 1 ∧ |α_declared/α| ≈ 1

    violations ← one per failed precondition, with class + remedy
```

Why each probe is shaped as it is:

* **H1 normalises by the median gain.** A window multiplies column `m` by `w[m]`: within a single
  column it is indistinguishable from `α`. It is only visible *across* columns, which is why all
  `2N` impulses are probed and the per-column gains are compared with each other.
* **H2 is pointwise.** An energy comparison can hide a sign flip or a phase offset that cancels
  over the band.
* **H3 is per bin.** POST can undo one scalar `α`; it cannot undo a per-bin gain.
* **H4 comes first.** With an unknown bin order, all three statistics are noise. Failing closed is
  the only defensible default.

The probe vectors are generated by `util::Rng` (xorshift64*) with fixed seeds, so a decision is
reproducible across machines — a requirement for an *auditable* gate. The report is
Markdown-exportable (`ComplianceReport::to_markdown`).

---

## 5. GPU model: assumptions

1. **One shared MDCT unit.** The backend is the resource being reused, so the schedule serialises
   its occupancy. This is what makes `P = max(t_m, Σ/c)` the correct steady-state period.
2. **FOLD and POST have enough lanes.** They are elementwise (one pass over `2N` and `N` samples)
   and memory-bound; on any GPU with spare SMs they never bind. If they did, add their per-stage
   service rate to the model.
3. **Two slots per in-flight block** — the folded `2N` word and the `N` output — held from FOLD
   start to POST end. This is conservative: in reality the folded word is dead after MDCT.
4. **Greedy in-order issue, round-robin over streams.** `simulate` replays exactly this policy;
   `project` bounds it from above.
5. **Logical time.** The pool tracks release instants, not CUDA events; nothing in this crate
   talks to a driver.

Under those assumptions `simulate` is exact for the greedy schedule, and `project` is a
conservative bound. The integration test `projection_bounds_the_replay_and_stays_useful` sweeps
5 stream counts × 5 pool sizes × 6 batch sizes × 4 cost models and asserts both properties.

### The staircase term

The pool releases slots in **waves of `c` blocks**, each held for `Σ`, staggered by `t_m`
(the MDCT queue's occupancy). That gives

```text
L + ⌊(B−1)/c⌋·Σ + ((B−1) mod c)·t_m
```

This regime appears whenever `t_f` or `t_p` dominates the MDCT unit — i.e. whenever a kernel
launch dominates a small block — and it is where the smooth `L + (B−c)⁺·P` law under-estimates.
Both bounds are taken, plus the unbuffered `t_f + B·t_m + t_p` and the bare `L`. The maximum of
the four is what `project` returns.

---

## 6. Assumptions and deviations (explicit)

| # | Item | What the code does | Why |
|---|---|---|---|
| 1 | Eq. 3's four clauses | Implemented verbatim, plus fractional-bit agreement, `α` domain, `σ` domain, non-empty batch/pool | The paper's clause list assumes a well-formed configuration; a real API cannot |
| 2 | Guard bit corner | `fold_overflows` counts the `c = d = −2^(W−1)` corner instead of wrapping | `+2^W` is one code past a `(W+1)`-bit maximum; no width contract can cover it with one bit |
| 3 | `Wip` dynamic range | `ip_clamps` counter + `numerics::required_ip_width` | `y = α·X`; the paper's `Wout ≥ Wip` rule says nothing about `α·max|X|` |
| 4 | Undecidable ordering | H1–H3 fail closed | A per-bin statistic computed on permuted bins is not evidence |
| 5 | Permutation correction | POST inverts the declared permutation | H4's "correctable by permutation" escape clause, made executable |
| 6 | Bypass mode | Still probes, still records, flags `is_enforced() == false` | Bring-up needs visibility; shipping needs the gate |
| 7 | GPU stage model | One shared MDCT unit, unlimited FOLD/POST lanes | The IP is the reusable resource; the other stages are memory-bound |
| 8 | Closed form | Conservative bound, not an equality | Under-promising a deadline is worse than over-estimating a wave |
| 9 | Reference kernels | `dct4`/`mdct` are `O(N²)` | They are oracles for the probe and the tests, never the production path |
| 10 | PDF scope | Only §1–§4.3 is implemented | The remaining pages are an unrelated story fragment, not a specification |

---

## 7. Test strategy

The suite is organised by *claim*, not by function, so a failure names the obligation it broke:

| File | Claim |
|---|---|
| `tests/folding_theorem.rs` | the mathematics: Thm 1, Thm 2, Cor 1, eq. 2, exactness on integers |
| `tests/pipeline_end_to_end.rs` | the pipeline reproduces the DCT-IV definition, on both paths, under every convention POST claims to absorb |
| `tests/compliance_gate.rs` | every H1–H4 rejection path, the fail-closed rule, `τ_dtype`'s scale argument |
| `tests/config_validity.rs` | eq. 3 clause by clause, plus a truth table over a small grid |
| `tests/gpu_adaptation.rs` | pooling law, DAG structure, projection ≥ replay, bandwidth account |

Property-style checks were preferred to golden vectors wherever the property is the point: exact
equality for integer folds, `PᵀP = I` for the isometry, `projection ≥ replay` over a grid, and
`measured ≤ bound` for the error model.

Two unit tests in `fixed.rs` deserve a note: `guard_bit_is_observable` shows the exact raw codes
that overflow a `Wu` word and fit a `Wu + 1` one, and `requantize_and_scale` pins the rounding and
saturation policy that the post stage depends on for determinism.
