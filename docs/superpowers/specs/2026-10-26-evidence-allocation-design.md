# Evidence-driven frozen-codec allocation

Approved scope: implement the two priorities discussed with the user: more useful positive-tier evaluations and true finite-render correction of important decisions. Work in the current checkout, preserving unrelated changes and existing experiment entry points.

## Invariants

- Completed shared codec is frozen, including statistics; trained scenes only.
- Reliable 16bit XYZ, rates 0/8/24/48, 10 dB, source RGB MSE, fixed beta normalized RD objective.
- Single-draw learning, ten-draw deployment, no forced tier proportions, no joint finetuning.
- Every sampled q0 receives the original gate feedback; finite correction is supplemental, never a cap on that path.
- Train/calibration/test separation and exact deployment cost accounting remain unchanged.

## Approach

Keep the existing kernel and lifecycle as the compatibility path. An independent entry enables two independent switches: evidence scheduling/memory and finite-render feedback. All-candidate finite rendering is too expensive; additional uncorrected first-order evaluations alone can reinforce incorrect decisions.

Each block evaluates two distinct targets by default. One is selected by least coverage/oldest visit, the other by current image influence, uncertainty, past action-cost variation and staleness. Each counterfactual changes just one point per independent block. Candidates share actual decode, gradient and paired channel noise.

Maintain centered image-action means and second moments per point. Only fresh measurements produce updates. Exponentially age evidence over 1000 global steps; invalidate a block when mean total-variation drift of its probability distribution exceeds 0.1. Ordinary stochastic mask changes are samples of that distribution, not automatic invalidations. Rate costs are current analytic action differences and are never normalized by per-point magnitude. Separate tier LR defaults to 0.4; keep LR remains 0.1. These are exposed engineering defaults.

Finite feedback defaults to eight interventions per update (six positive, two keep, when available). Mix high impact, ambiguous RD decisions and rotating coverage. Baseline training-window images are cached during backward. Compare candidate images using mean(2 residual*delta + delta^2) with float64 accumulation. Selected finite costs replace, rather than merely audit, training estimates. Keep-set changes recompute semantic context and all affected block outputs. Supplemental renders preserve training RNG, codec state and base view statistics.

## Records / safeguards

Checkpoint scheduler, evidence moments/age/context snapshots and finite-selection coverage together with the existing table/RNG. Reject incompatible resume settings. Record candidate throughput, coverage, effective evidence, finite correction count/sign/error (zero changes separately), probability confidence, timing and deployment seed sensitivity. Preserve initial/final image policy, PSNR/SSIM/LPIPS and tier maps.

CPU tests use the real small codec with a differentiable toy rasterizer; server CUDA quality/speed remains an experiment, not a local success claim.
