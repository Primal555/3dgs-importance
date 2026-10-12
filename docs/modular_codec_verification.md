# Modular experiment implementation status

The eight implementation tasks in the approved plan are complete. The old
experiments remain independent; the new Truck entry point is
`scripts/test_modular_codec_truck.sh`.

Before the final read-only review, the complete CPU suite reported 239 tests,
with four skips and no failures. A 60-update synthetic four-phase run exercised
training, packet receive, plots and exact CPU continuation. This is execution
evidence, not evidence of improved Truck quality or CUDA performance.

The final reviewer identified two resume-setting gaps. The final patch records
and compares `blocks_per_batch` and `views_per_step`, and preserves the skipped
semantic phases when resuming a semantic-only fork. These narrow final changes
were inspected but not retested, following the user's instruction to stop
additional regression testing. Checkpoints created before these settings were
recorded are intentionally rejected rather than silently treated as exact resumes.

Minor follow-ups remain: recovered-latent mean/std reporting, and explicit
override matching for the optional historical direct-control arm. The default
launch does not enable that arm. Do not assume that optional arm is a matched
SNR/seed comparison when overriding the modular defaults.

Real Truck quality, accelerator determinism, peak memory and runtime require
the server experiment. Unrelated workspace changes are excluded from this update.
