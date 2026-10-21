# Successive channel implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Execute inline in the current repository; existing user instructions authorize implementation and GitHub push.

**Goal:** Implement the approved conditional enhancement, block-scale delivery and independently supervised channel main path.

**Architecture:** New opt-in channel variant and independent three-phase runner. Existing semantic modules, scene loading and image evaluation are reused.

**Tech Stack:** PyTorch, Python unittest, Bash launchers.

**Spec:** docs/superpowers/specs/2026-10-21-successive-channel-design.md

## Global constraints

- Preserve user work and legacy config hashes; no full local training/regression.
- Frozen semantics, random channel initialization,10dB,8/24/48, explicit scale cost.
- Work in current repository and push when targeted checks/review pass.

## Review focus

1. Missing/corrupt scale frame must fail before receiver inference.
2. Partial/padded/q0 blocks must not change donor power or leak upper prefixes.
3. Phase-boundary interruptions must restore correct enabled flag and Adam scope.
4. New factory variant must never silently enter legacy symbol-only transport.
5. No color step may drop common recovery, nor noise simulation reset effective SNR by signal energy.

## Task1: signal modules and receiver packet

- [x] Write failing tests in tests/test_successive_channel.py for scale byte roundtrip,
  donor power, prefix causality, real receiver parity and gradients.
- [x] Implement gaussian_jscc/successive_channel.py and successive_codec.py;
  extend modular_config/model_factory only for opt-in variant.
- [x] Run new module tests and legacy config/factory/transport tests.

Interfaces: SuccessiveChannelEncoder.forward(y,q,snr,decoder) -> symbols,scales;
SuccessiveChannelDecoder.forward(received,mask,q,snr,scales,delivered_xyz,return_stages);
scale frame encode/decode; SuccessiveGaussianCodec packet and batched APIs.

## Task2: objectives, scopes and resume-capable runner

- [x] Write failing objective/phase/runner tests in tests/test_successive_training.py.
- [x] Implement successive_objective.py and successive_experiment.py: common losses,
  adaptive color,8000/1000/1000 phases, source validation and exact checkpoint resume.
- [x] Add opt-in scale rate records to existing modular evaluation; fixed-block
  prefix/main/final diagnostics at every evaluation.
- [x] Verify real CPU short experiment vs interrupted/resumed tensor/Adam equality.

Interfaces: independent parser/main(max_updates), phase selection/optimizer,
objective-state metadata in progress; existing evaluation/checkpoint formats.

## Task3: launch, documentation and final review

- [x] Add scripts/test_successive_channel_truck.sh with explicit GPU and documented defaults.
- [x] Add user-facing docs, actual dry-run launcher test, syntax/compile checks.
- [x] Run targeted new+adjacent tests, request independent whole-change review,
  resolve important findings, commit only task files and push main.

## Execution ledger

- Setup: user selected current-repository development; unrelated dirty files preserved.
- Pre-flight: packet scales flow through codec/factory/evaluation; training scopes use
  the same new decoder modules. No source channel optimizer is loaded.
- Task1 complete: six signal tests pass; byte parity, donor power, prefix isolation,
  sender-proxy gradients, checkpoint reload and missing/nonfinite receiver data checked.
- Task2 complete: objective and phase tests plus real short synthetic-render integration
  pass; exact resume cuts3/4/6/9 match all model/Adam tensors and progress.
- Ruling: actual RMS is rounded UP to binary16 and the transmitter divides by that
  same quantized value, avoiding wire/training mismatch and quantization power overshoot.
  Out-of-range scales abort; the protocol does not silently substitute float32.
- Ruling: original semantics are copied into the new factory model; no source channel
  tensor or Adam is initialized from the source. Existing config serialization is unchanged.
- Final review: independent read-only review found no Critical/Important issues.
  Added separately named optimizer-group diagnostics requested by review; missing-key
  regression observed RED then GREEN. Color is computed on other steps for continuous
  diagnostics only; documented the small extra SH cost (no extra auxiliary backward).
- Verification:49 targeted tests across successive channel/training, modular codec,
  checkpoint/transport/evaluation, progressive latent, spatial denoising and power
  normalization passed. Bash syntax, compileall and real source dry-run passed.
  No full repository regression or full Truck/CUDA run was performed.
- Limits retained from review: real Truck quality, CUDA memory/throughput/determinism
  and FEC reliability require the server experiment. Synthetic renderer numbers are
  plumbing checks only. Unrelated dirty user files were left untouched.
