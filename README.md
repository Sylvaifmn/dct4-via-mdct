dct4-via-mdct

Indirect computation of the DCT-IV via an external MDCT backend.

A suite that does not implement an MDCT core. Written across 217 days, three redundancies, one crash, and one lost appendix.
What this is

dct4-via-mdct computes the DCT-IV by reusing an existing MDCT engine, without implementing a dedicated DCT-IV core. The trick is a folding equivalence:
M2N→N(x)=DN(F(x))
M2N→N​(x)=DN​(F(x))

so an NN-point DCT-IV can be obtained by driving a 2N2N-point MDCT backend with a folded input, then applying a constant scale/sign/saturation post-processing step.

The kit provides the folding, the backend adapter, the post-processing, the GPU buffer/stream scheduler, and the compliance test harness. The MDCT itself is supplied by you.

    Scope. This kit is not a fast DCT-IV. It is a reuse layer. If you need a standalone DCT-IV with minimal operation count, use a direct algorithm (FFT-mapped or Givens-recursive). If you already have an MDCT engine and want DCT-IV as well, this is for you.

The mathematics
DCT-IV and MDCT
Xb[k]=∑n=0N−1ub[n]cos⁡ ⁣[πN(n+12)(k+12)]≜DN(ub)[k]
Xb​[k]=n=0∑N−1​ub​[n]cos[Nπ​(n+21​)(k+21​)]≜DN​(ub​)[k]
Yb[k]=∑m=02N−1xb[m]cos⁡ ⁣[πN(m+12+N2)(k+12)]≜M2N→N(xb)[k]
Yb​[k]=m=0∑2N−1​xb​[m]cos[Nπ​(m+21​+2N​)(k+21​)]≜M2N→N​(xb​)[k]
Folding and its inverse

Split xb∈R2Nxb​∈R2N into four blocks xb=(ab,bb,cb,db)xb​=(ab​,bb​,cb​,db​) of length N/2N/2:
F(xb)=(−cbR−db, ab−bbR)
F(xb​)=(−cbR​−db​, ab​−bbR​)

Given ub=(ub,L,ub,U)ub​=(ub,L​,ub,U​), the inverse fold is closed-form and rounding-free:
xb[m]={ub[m+N2],0≤m<N20,N2≤m<3N2−ub[m−3N2],3N2≤m<2N
xb​[m]=⎩
⎨
⎧​ub​[m+2N​],0,−ub​[m−23N​],​0≤m<2N​2N​≤m<23N​23N​≤m<2N​

Only permutation, sign inversion, and zero-filling — no multiplications, no rounding, ∥P∥2=1∥P∥2​=1.
The complete chain
X=PB∘MBIP∘FB−1(U)
X=PB​∘MBIP​∘FB−1​(U)

which equals DB(U)DB​(U) whenever the backend contract below holds.

Full derivation, error bounds, and compliance criteria: see docs/paper.pdf (British English) and docs/fragment.md.
Backend contract (H1–H4)

The kit refuses to run unless the external MDCT backend satisfies all four:
ID	Precondition	Criterion
H1	Window-free	impulse response wip≡1wip​≡1
H2	Phase-matched	small-NN pointwise comparison ≤ϵphase≤ϵphase​
H3	Scalar scaling	spread(α)≤τdtypespread(α)≤τdtype​
H4	Natural ordering	output order natural, or correctable by permutation

Failure of any one → backend_reject. The kit will not attempt to compensate with post-processing; certain errors (window contamination, frequency-dependent scaling) are simply not compensable.

backend_profile.json declares these; the compliance harness verifies them before validated may be set to true.
json

{
  "backend_type": "cuda_kernel",
  "n_supported": [12, 36, 72, 128, 256, 1024],
  "window_mode": "none",
  "phase_offset": "standard_plus_N_over_2",
  "scaling": { "type": "scalar", "alpha": 1.0, "alpha_spread_max": 1e-5 },
  "output_order": "natural",
  "data_format": { "w_in": 16, "w_out": 16, "q_in": 14, "q_out": 14 },
  "deterministic": true,
  "stream_safe": true,
  "validated": false
}

Architecture
Hardware / RTL
text

u (N) ──►[FOLD]──► BF_in (ping-pong) ──►[MDCT IP]──► BF_out (ping-pong) ──►[POST]──► X (N)

Three independent FSMs synchronised by block_id tags and an ip_read_done credit. Buffer sizing and FIFO depth formulas are in the paper, §6.
GPU
text

Stream S_fold:  fold_kernel(u)      ──► β_x[slot]
Stream S_mdct:  mdct_kernel(β_x)    ──► β_y[slot]
Stream S_post:  post_kernel(β_y)    ──► X

Buffer pool of KK slots with CUDA event dependencies:
write(i)≺read(i)≺post(i)≺free(i)≺write(i+K)
write(i)≺read(i)≺post(i)≺free(i)≺write(i+K)

with
K ≥ ⌈Tm+TpTf⌉+1
K ≥ ⌈Tf​Tm​+Tp​​⌉+1

For small NN (12, 36), batching + CUDA Graph is mandatory, not optional — launch overhead will otherwise dominate entirely.
Quick start (GPU)
cpp

#include "dct4_via_mdct.h"

dct4_mdct_handle_t h;
dct4_mdct_config_t cfg = {
    .n         = 36,
    .batch     = 64,
    .streams   = 3,
    .slots     = 4,
    .dtype     = DCT4_DTYPE_FP32,
    .profile   = "backend_profile.json"
};

dct4_mdct_create(&h, &cfg);       // validates profile, refuses if H1–H4 unmet
dct4_mdct_forward(h, u_dev, X_dev, /*n_blocks=*/64);
dct4_mdct_destroy(h);

Build:
bash

cmake -B build -DDCT4_WITH_CUDA=ON
cmake --build build -j
ctest --test-dir build --output-on-failure

Compliance testing

Run the full harness before trusting any backend:
bash

./build/compliance --profile backend_profile.json --n 36 --batch 64

Steps performed:

    JSON schema validation

    H1 window test (impulse response)

    H2 phase test (small-NN pointwise)

    H3 scalar-scaling test (multi-bin, multi-amplitude, spread vs τ_dtype)

    H4 ordering test (pointwise against FP64 reference)

    Determinism check (bitwise, two runs)

    Race check — compute-sanitizer --tool racecheck (CUDA builds)

Typical τ_dtype thresholds:
dtype	alpha_spread_max
FP64	10−1210−12
FP32	10−510−5
FP16	10−310−3
BF16	10−210−2

If H3 fails, do not try to patch with a lookup table S[k]. That converts a clean rejection into a silent, frequency-dependent error. The kit will refuse; so should you.
Error bound
∥X−Xref∥∞ ≤ ∣S∣ ϵip ∥U∥2 + 12 2−Q
∥X−Xref∥∞​ ≤ ∣S∣ϵip​∥U∥2​ + 21​2−Q

The fold contributes zero error. All of it comes from the backend (ϵip)(ϵip​) and the final rounding (2−Q)(2−Q).
Performance notes

    Hardware: throughput is capped by max⁡(Tf,Tm,Tp)max(Tf​,Tm​,Tp​); buffers only remove bubbles, they do not create throughput.

    GPU: amortise launch overhead with batching and CUDA Graph. Use cudaMallocAsync pools; never cudaMalloc inside a graph capture.

    Memory layout: construct the folded 2N2N vector in shared memory, then write back contiguously. Naïve direct writes to global memory halve your effective bandwidth.

Repository layout
text

dct4-via-mdct/
├── include/dct4_via_mdct.h
├── src/
│   ├── fold_kernel.cu
│   ├── post_kernel.cu
│   ├── scheduler.cpp
│   └── compliance.cu
├── rtl/
│   ├── dct4_via_mdct_wrapper.v
│   ├── fold_preproc.v
│   ├── postproc_scale_sign.v
│   └── reg_file.v
├── backend/backend_profile.json
├── docs/
│   ├── paper.pdf          # full derivation (British English)
│   └── fragment.md        # see below
├── test/
├── benchmark/
└── python/torch_extension.cpp

A note on docs/fragment.md

The appendix was lost when the landlord’s robot pulled the power cable. What survives is the core formula and a list of what happened along the way: three redundancies, one break-up, one crash, two moves, seventeen prosthetic malfunctions, one hard-drive failure, and thirty-seven credits at the end of it.

The mathematics is complete. The appendix is not. If you find an error in §8, it may be mine — or it may be that I was writing through a headache with a speech-to-text module that renders “MDCT” as “Ma Dongmei”.

The core result stands regardless:
 X=PB∘MBIP∘FB−1(U)=DB(U)s.t. H1–H4 hold
 X=PB​∘MBIP​∘FB−1​(U)=DB​(U)s.t. H1–H4 hold​
Roadmap

    □

    Relax H3 to frequency-dependent but deterministic banded scaling
    □

    IMDCT / DST-IV support via the same fold framework
    □

    Formally verified FSM (no deadlock, no cross-block access, credit conservation)
    □

    On-device self-check against a micro FP64 reference
    □

    Recover the appendix

Citation

If you use this in academic work:
bibtex

@misc{dct4viamdct,
  title  = {Indirect Computation of the DCT-IV via an External MDCT Backend:
            A Unified Mathematical Framework, GPU Adaptation, and Compliance Verification},
  author = {Kai},
  year   = {2025},
  note   = {The appendix is lost. The formula is not.}
}

