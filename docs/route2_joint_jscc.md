# Four-tier per-Gaussian allocation: local image feedback

The current training entry point is `train-learned --joint-steps N` with a
**progressive** codec and a matching trained codec checkpoint. The historical
scene-level REINFORCE allocator is no longer called by this entry point.
`scripts/test_local_mask_feedback.sh` starts one controlled run from a fixed
codec: first mask-only updates, then optional joint codec/mask updates. It
writes parity diagnostics, validation images, allocation snapshots and measured
rates into the same output directory.

## Forward semantics

Each original PLY row has four logits. Hard sampled tiers select q0 (drop) or
one of the 8/16/32-complex-symbol progressive prefixes. Only q>0 rows receive
16-bit compressed XYZ. Deployment uses `argmax` tiers, real packing and the
ordinary Gaussian rasterizer. There is no q0 payload, coordinate or splat.

During **training only**, up to `--mask-shadow-per-block` sampled q0 rows per
source block are decoded hypothetically as q1 and added to the MaskGaussian
renderer with an exactly zero existence mask. This cap limits counterfactual
context changes and makes q0 feedback sparse when many rows are dropped.
This does not change the rendered forward image; the custom rasterizer supplies
an image derivative with respect to that row's mask. Positive-tier choices use
hard-forward/soft-backward cumulative prefix gates and a differentiable tier
embedding. The common codeword, quantized XYZ and full-slot paired noise match
the hard progressive path. This is a *biased local gradient surrogate*, not an
exact derivative through an actual discrete transmission.

The visual objective remains multiview RGB MSE against the source PLY render.
No position, parameter or projection auxiliary is added in joint allocation.
Payload has an exact differentiable expected rate. The compressed XYZ stream
has a detached measured per-retained-row proxy; the compressed tier-map byte
count is measured but does not provide a gradient. Actual compressed bytes and
hard deployment q0/q1/q2/q3 counts are logged at validation and export; the
proxy objective must not be presented as an exact total-rate gradient.

## Checks before interpreting a run

`mask_renderer_check.json` compares an ordinary hard scene render to the
masked renderer with zero-gated candidate rows. Training aborts if this
forward discrepancy exceeds the configured tolerance. CPU tests check that
the straight-through codec's hard forward matches the existing paired-noise
codec and that q0 plus positive tiers receive gradients. Those checks do not
establish real-scene visual quality or validate the *direction* of the biased
gradient. The latter requires held-out quality and measured-rate improvement,
and ideally spot-checking sampled one-point tier flips on CUDA.

Use the matched `codec_best_joint.pt` and `route2_best_joint.pt` together.
`--init` starts fresh optimizer state; it is not an exact resume.
