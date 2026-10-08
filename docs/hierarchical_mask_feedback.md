# Hierarchical allocation feedback

Rates remain 0/8/16/32; existing codec weights are unchanged. Allocation tables
now have independent binary keep logits and conditional positive-tier logits:
`P(q0)=1-P(keep)`, `P(qk)=P(keep)*P(k|keep)`.

Both decisions use hard Gumbel-Softmax, temperature 1. The forward is discrete;
the backward is a biased stochastic relaxation. Noise is sampled once and stored
with each batch so replay uses precisely the same mask and conditional tier.
Every sampled q0 enters the masked rasterizer with a zero gate and its sampled
conditional-tier counterfactual attributes, not a compulsory q1 reconstruction.
Counterfactuals are decoded separately and cannot change the live delivered-map
attention context.
No q0 candidate cap or compulsory deletion quota is imposed. Positive-tier image
gradients are gated off for dropped points; the rate expectation can update both
branches. Geometric culling still limits which points receive image feedback.

The rate objective is unchanged: expected payload plus a measured per-retained
XYZ cost proxy and measured tier-map overhead, normalized by full-q3 cost.
Compressed costs are not exact differentiable pointwise costs. Ten-draw deployment
is unchanged: delete only after ten q0 draws, otherwise choose the positive mode.
Consequently sampling proportions and actual deployment proportions still differ.

Allocator Adam uses eps 1e-15 (as in MaskGaussian) instead of 1e-8; keep learning
rate defaults to .01 and conditional tier learning rate to .001. These are
adjustable engineering choices, not a guarantee of correct pruning. Tiny noisy
gradients may now produce larger updates, so monitor keep probabilities, separate
image/rate branch gradient norms, and branch update norms in loss.jsonl.

The test launcher freezes the codec for all 1000 allocation updates by default.
Set MASK_ONLY_STEPS explicitly to enable subsequent joint updates. Compare beta
0/.01/.03 with the same codec, seed, views and stage lengths. Examine deployed
counts, true total communication cost and source/photo quality, not total losses
across different beta values. Version-1 flat allocation checkpoints are rejected
explicitly; initialize a new allocation table with the existing codec instead.

Reference implementation:
https://github.com/kaikai23/MaskGaussian/blob/main/scene/gaussian_model.py
