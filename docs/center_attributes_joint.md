# Continue from a selected center model

`scripts/test_center_attributes_joint.sh CENTER_RUN NEW_OUTPUT` starts from the
source run's `best_steps.center` / `codec_selected_center_STEP.pt`, not the last
center model. It skips A explicitly; this does not assert A passed its old gate.
Use a completed center-only run containing `training_state.pt` and selected
checkpoints. The source directory is never changed.

Defaults (engineering starting values, environment-overridable):

| Phase | Updates | Objective | Trainable modules | LR |
| --- | ---: | --- | --- | --- |
| B | 5000 | Existing log-covariance and local RGB-response attribute objective | Attribute encoder/decoder only | 2e-4 |
| C | 1000 | Source-scene rendered image MSE | Both center and attribute paths | Center 1e-5, attributes 1e-4 |

B uses predicted frozen XYZ as receiver context, not teacher coordinates. C does
not detach XYZ and uses replay, default 64 blocks/batch. Each new phase starts a
fresh Adam; attaching the last-center optimizer to best-center weights would be
incorrect. This remains clean representation learning, not noisy JSCC training.

The wrapper explicitly selects `--later-phase-policy budget`: B/C run their full
budgets without plateau/gap gates. Nonfinite checks remain active. B chooses the
best heldout attribute objective for entry to C; C chooses best validation full
render MSE, including the initial B model as a candidate. PSNR may not improve.

All B/C checkpoints, validation metrics, render images, loss curves and center
drift snapshots are in NEW_OUTPUT. `phase_start.json` records the source and
selected center step; `codec_center_initial.pt` preserves the starting model.
Step numbering continues from the source's processed update count (10000 in the
current experiment), while B/C phase counters start at zero. Earlier A logs stay
in CENTER_RUN, not duplicated as if the selected weights were the final weights.

Example (choose a free GPU):

```bash
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_center_attributes_joint.sh \
  "$PREV" "$OUT" > "${OUT}.log" 2>&1 &
```

Overrides: `ATTRIBUTE_STEPS`, `JOINT_STEPS`, `ATTRIBUTE_LR`, `JOINT_CENTER_LR`,
`JOINT_ATTRIBUTE_LR`, `RENDER_BLOCKS`, `PYTHON_BIN`. Validate/render/save every 500
updates and at phase boundaries. B/C loss scales are different: use full-render
validation metrics to compare quality, not raw losses across phases.

After interruption, use `python -m gaussian_jscc train-center-attributes --resume
NEW_OUTPUT/training_state.pt` (with the same GPU visibility). Do not restart the
wrapper against the same output. Exact resume retains the active phase optimizer,
RNG, budgets and original learning rates.
