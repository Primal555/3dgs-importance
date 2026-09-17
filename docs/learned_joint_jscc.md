# Fully learned Gaussian JSCC mainline

This replaces the handcrafted geometric transport direction. It is an implemented
architecture, not a claim that the position-recovery problem has already been solved.
Use `train-learned` or its alias `train`. The historical training implementations
and `train-route2` entry point have been removed; joint training uses `--joint-steps`.

## Architecture and wire contract

```text
Frozen source PLY + independent per-Gaussian q in {0,1,2,3}
  -> Morton ordering / fixed source blocks (default 256; q0 does not move boundaries)
  -> learned XYZ / scale / rotation / opacity / SH embeddings
  -> shared feature fusion + FCGS-inspired sender grid aggregation
  -> learned shared symbol head
  -> per-row prefix selection k(q) + smooth power normalization
  -> packed complex symbols -> AWGN
  -> receiver unpack using q map (absent slots are masked)
  -> local received-feature attention (window 32; alternating shifted windows)
  -> parallel learned affine XYZ / scale / rotation / opacity / SH heads
  -> remove q0 rows -> physical parameter conversion -> render
```

The sender grid uses source XYZ; the receiver never receives that grid or its
bounds. Its attention keys/values come from noisy symbol features. Decoder
context is not constructed from guessed XYZ. Deterministic sequence encoding
restarts in every fixed block: no global Gaussian-ID embedding / scene lookup table.

q1/q2/q3 may coexist inside every block. A block is a context/computation unit,
not an allocation unit. q0 has zero payload, is masked out of aggregation and
attention, and is omitted from the rendered scene. q1/q2 mean lower resource
budgets, **not smaller opacity**.

All active symbols are learned from joint geometry/attribute features. There is
no position/appearance split in the rate table and no separate geometric encoder.
Separate output heads only specify parameter semantics; the receiver does not
need a header describing how many symbols were devoted to XYZ. `return_seed`
exists for old benchmark API compatibility and returns final XYZ, not another stage.

Default rates are `(0,8,16,32)` **complex** symbols per source Gaussian. They are
engineering comparison points, not established optimal budgets or guaranteed
quality levels. The tier conditions both networks; different tiers need not be
literal truncations of the very same encoded waveform. No monotonic-quality
guarantee is imposed.

For selected real/imaginary coordinates `z`, normalization is
`z / sqrt(mean_complex_energy(z) + power_floor)`. The default floor `.01`
bounds normalization gain by 10 and yields mean complex energy **at most** one.
There are no fixed-energy filler symbols or transmitted amplitude multipliers.
AWGN variance uses unit reference power; effective measured transmit energy is
reported in packet stats. This is a power constraint, not a guarantee against
large gradients elsewhere.

Metadata still contains global bbox normalization (six floats), source count,
the compressed two-bit-per-row q syntax, configuration/model identity and
channel configuration. No individual/block coordinate reference is sent.
Headers are assumed reliably received, with their size and modeled channel
uses counted. Header channel errors are not simulated; model weights are shared
in advance and excluded from payload cost. These assumptions remain explicit.

## Render-first training objective

The maintained trainer now optimizes multiview source-render RGB MSE. The old
six-term reconstruction/projection combination is no longer the main objective.
Optional normalized-feature SmoothL1 is initialization only; it is absent from
render and mask-joint training. The architecture and v4 wire identity are unchanged.

See [complete objective, stages, validation and server commands](render_first_training.md).
Use `scripts/test_render_first.sh` for a full-scene, reduced-view/step observation
run. `scripts/train_codec_learned.sh` runs the longer schedule. Both accept an
explicit learned_joint `INIT` only with `INITIALIZATION=checkpoint`. The default
is random weights with zero bootstrap steps, even if an old `INIT` remains in the shell.

Joint mask training uses image MSE plus an expected normalized payload penalty;
see [four-way allocation](route2_joint_jscc.md). Reliable header delivery remains
an explicit assumption. No claim of cross-scene generalization is made.

## Verification boundaries

CPU tests exercise real codec/channel gradients, replay/checkpoint equivalence,
streamed multiview gradients, fixed-noise validation, per-Gaussian tier behavior,
packet identities, and full stage orchestration with a synthetic renderer.
They do not establish real CUDA rasterizer fidelity, training convergence, or
acceptable recovered images. These require the server experiment.

The CPU-only `scripts/test_learned_codec_local.py` remains an optional normalized
feature bootstrap smoke test, not a substitute for rendered-quality evaluation.
