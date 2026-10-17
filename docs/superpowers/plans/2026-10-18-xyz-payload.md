# XYZ-in-payload JSCC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separate Truck experiment carrying three normalized XYZ values and a 48D attribute latent through one progressive channel codec, without reliable per-point XYZ.

**Architecture:** New config/model, training, evaluation, checkpoint and packet modules reuse existing attribute/context and progressive-channel components. Semantic encoder stays frozen in communication render by default, as explicitly confirmed by the user.

**Tech Stack:** Python, PyTorch, NumPy, plyfile, existing CUDA rasterizer on the server, unittest and CPU synthetic-render tests locally.

**Spec:** `docs/superpowers/specs/2026-10-18-xyz-in-payload-jscc-design.md`

## Global Constraints

- Separate architecture `xyz_payload_jscc_v1`; existing reliable-coordinate experiments retain default behavior and serialized identities.
- XYZ uses `u=2*(XYZ-c)/s`, `s=max(bbox_span)`; attribute latent 48, total latent 51; no learned position transform.
- q1/q2/q3 cumulative TOTAL budgets 8/24/48 complex symbols, fixed 10 dB; no allocator or channel spatial denoiser.
- Budgets semantic bootstrap/render 4000/1000, channel bootstrap/render 4000/2000. Directional color update replaces every fourth semantic bootstrap update.
- Group-balanced payload MSE, XYZ weight 1, auxiliary payload weight 1 in render, explicit settings; LR 1e-4; replay render backward.
- Preserve exact resume, self-contained receiver packets, source/photo PSNR/SSIM/optional LPIPS, XYZ and color diagnostics, compact endpoint images and plots.
- Main worktree contains unrelated user edits; never stage or overwrite them.

## Review Focus

- Padded or all-inactive blocks must not become source-coordinate donors or produce NaN diagnostics.
- A corrupted/mismatched packet or checkpoint must fail before decoding with the wrong geometry or model.
- Resume at the final update of a phase must materialize its endpoint without replaying an optimizer update.
- A failed/partial output must never be mistaken for complete; initialization and resume must reject incompatible objectives.
- Frozen semantic encoder and unused prefix stages must not receive Adam momentum updates during communication tuning.

### Task 1: Coordinate payload model and attribute objective

**Files:** new `gaussian_jscc/xyz_payload_config.py`, `xyz_payload_codec.py`; modify `modular_core.py`, `local_response.py`; test `tests/test_xyz_payload_codec.py`.

**Interfaces:** `XYZPayloadConfig.attribute_config() -> ModularCodecConfig`; `XYZPayloadCodec.encode_representation(features, xyz, active) -> y`; `decode_representation(y_hat,q,delivered_xyz=None) -> features`, rejecting side coordinates; existing transport-style encode/decode plus batched clean/channel methods.

- [ ] Write tests: 3+48 layout, unequal-axis global scaling, receiver uses only payload XYZ, padding and no external XYZ; prefix causality and continuous XYZ-to-attribute gradients.
- [ ] Run `python -m unittest discover -s tests -p test_xyz_payload_codec.py -v`; expect missing implementation failures.
- [ ] Implement new configuration and wrappers around existing attribute modules; allow optional differentiable geometry in existing SemanticDecoder while keeping its default detached. Extract isolated attribute response computation without weakening legacy entrypoint guard.
- [ ] Rerun focused tests, expect all pass; commit tested model/objective changes.

### Task 2: Training objectives, frozen scopes and checkpoint state

**Files:** new `xyz_payload_training.py`, `xyz_payload_checkpoint.py`; test `tests/test_xyz_payload_training.py`.

**Interfaces:** `payload_loss(y,y_hat,q,xyz_weight=1.) -> (loss,metrics)`; `bootstrap_step`, `render_step`, `configure_phase`; `save_checkpoint(path,model,optimizer,progress,scene_record)`, `load_checkpoint`, `restore_checkpoint`.

- [ ] Write tests for literal group-balanced loss values, padding, phase updates, frozen encoder, non-color head momentum, render-position gradients and exact Adam/RNG checkpoint restoration.
- [ ] Run focused tests RED, implement new objective paths using full_scene_step replay and existing color supervision, then run GREEN.
- [ ] Enforce unique checkpoint marker, finite settings, objective identity and phase ownership; commit.

### Task 3: Source-free transport and evidence-backed evaluation

**Files:** new `xyz_payload_transport.py`, `xyz_payload_evaluation.py`, `xyz_payload_plots.py`; test `tests/test_xyz_payload_transport.py`, `test_xyz_payload_evaluation.py`.

**Interfaces:** `export_packet(model,raw,q,snr,kind,seed,out,net_bits_per_use=2.)`; `receive_packet(model,folder)` uses no source PLY; `evaluate_paths(model,scene,step,phase,out,...) -> list[dict]`; `plot_run(out)`.

- [ ] Write tests for absent xyz.bin, exact packing budgets, CRC/model/config/geometry checks, no source access, hand-computable position metrics, clean/noiseless/AWGN distinctions and RNG preservation.
- [ ] Run RED; implement streaming decode, position/bias/color/latent evidence, matched noise, image protocols, full and teacher-replacement diagnostic renders and compact endpoint storage.
- [ ] Run GREEN; verify plot artifacts from real synthetic records rather than source-text assertions; commit.

### Task 4: Runnable experiment, exact continuation and documentation

**Files:** new `xyz_payload_experiment.py`, `scripts/test_xyz_payload_truck.sh`, `docs/xyz_payload_truck.md`, `tests/test_xyz_payload_experiment.py`.

**Interfaces:** `parser()`, `main(args,max_updates=None)`; shell `SOURCE_RUN NEW_OUT [options]`, explicit CUDA_VISIBLE_DEVICES.

- [ ] Write synthetic-render integration tests for all phases, midphase/boundary resume equivalence, mismatched objective rejection, invalid output/settings, metadata-only random initialization and optional LPIPS protocol.
- [ ] Run RED; implement dedicated scene loader without PositionCostMeter, training driver, compact validation/test export, phase checkpoint completion and explicit semantic initializer.
- [ ] Run focused GREEN, shell syntax/behavior tests, CLI help, CPU coordinate-channel learning smoke test and entire unittest suite with output captured to log.
- [ ] Document complete random-start and exact-resume server commands; mark approved spec implemented only after verification. Commit.
- [ ] Fresh whole-branch reviewer examines information leakage, loss weighting, replay/RNG/Adam, packet accounting and resume. Fix important findings with regression tests, run suite, merge fast-forward and push GitHub as already requested by user.
