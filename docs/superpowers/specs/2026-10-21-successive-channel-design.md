# Conditional successive channel codec

Approved scope: independent Truck channel experiment; frozen semantic teacher,
reliable quantized16 XYZ, 48-real latent, cumulative8/24/48 complex symbols,
10dB AWGN. Source channel weights and Adam are never copied. Existing runs and
packet formats retain their behavior. Work in the current repository as requested.

## Signal and receiver boundary

Three transmitter MLPs use hidden192 and three residual blocks. E1 reads y;
E2/E3 read y and y-minus-transmitter-local prior reconstruction. Local estimates
use the actual quantized scales and noiseless transmitted prefixes, with no
gradients through the local decoder. No actual receiver noise/target is fed back.
The receiver uses only received prefixes, q, quantized scale bytes and delivered
XYZ. Main stages read the complete available prefix, previous recovery, log2
scales and SNR. Stage2/3 produce cumulative residuals. No extra pointwise
denoisers; an independent final spatial refiner is enabled after main training.

Each segment is normalized over its actual donor points in each ORIGINAL block.
The positive RMS scale includes floor0.01 and is rounded UP to IEEE binary16;
the same quantized scale divides the transmitted signal. Values outside finite
binary16 range abort explicitly. Missing segments carry no scale; receiver uses
one as the unused placeholder. Each scale frame is magic SCL1, uint32 entry
count and little-endian binary16 entries, in block/segment order derived from q.
No upper-tier scale reaches a lower-tier stage. Scales are reliable side data,
not free information: rate reports include content/framing bytes and explicitly
assumed digital net bits/use. No FEC or packet errors are simulated.

## Training

Default stages: main8000, neighbor1000, cooperative1000 real Adam updates.
Semantic encoder/decoder and attribute statistics stay frozen. Main phase trains
all three prefix outputs on the same batch/noise, without neighbor repair.
Every step includes original-y MSE plus0.1 frozen-teacher local response.
Every fourth step also applies independent directional RGB; other steps compute
the same RGB scalar as a diagnostic only. Color weights are
based on EMA squared gradients w.r.t. recovered latent, updated only after250
main steps; color starts after500 steps, with requested relative gradient0.25,
capped by current gradient ratio and coefficient1. These are public engineering
settings, not optimal constants. There is never a pure-color replacement update.
Every fourth main/cooperative step adds0.25 noiseless main latent MSE.
Directions: half training camera directions, half isotropic, eight total.

Neighbor phase freezes sender/main stages and trains only the zero-output spatial
refiner (world-distance16 neighbors). q1/q2/q3/mixed rotate. Cooperative phase
uses main LR1e-5, neighbor LR1e-4 and averages common recovery losses before and
after spatial repair. Main/neighbor LR initially1e-4. Every phase starts new Adam;
exact resume within a phase restores Adam, RNG, adaptive-color state and cursor.
Neighbor output is disabled initially and the enabled flag is checkpointed.

## Experiment and validation

Independent CLI/script with source identity checks, source immutable, dry-run,
same-output exact resume. Checkpoints retain model factory support. General
legacy symbol-only encode/decode/allocation methods must reject this variant
instead of dropping the scale frame. Explicit packet API supports receiver-only
decode from actual analog payload and scale bytes.

Numerical validation every500 steps, phase boundaries and final; test32 views,
four paired noise trials, clean/noiseless/10dB, q1/q2/q3/mixed, source/photo
PSNR/SSIM/VGG-LPIPS/color bias. Images only initial/final representative views.
Per-update logs include prefix/main/final errors, task weights and gradients,
optimizer group updates, power, scale overhead, time and GPU peaks. Add small
fixed-block pre/post-neighbor diagnostics at validation checkpoints so future
dependency claims can be measured. Render forward is evaluation-only.

Targeted CPU tests only, including wire parity/power/padding, prefix isolation,
gradient boundaries, phase scopes, adaptive color, source-only initialization,
checkpoint loading and exact resume across phase boundaries. Synthetic rendering
tests validate plumbing, not Truck quality. No local full training/regression.
