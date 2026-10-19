#!/usr/bin/env bash
# Independent third loss arm; never launches the running SSIM/LPIPS arms.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:-$PROJECT/output/truck_beta_parallel_gpu2_20261017_213438/beta0p005}"
OUT="${2:-$PROJECT/output/truck_adversarial_$(date +%Y%m%d_%H%M%S)}"
if [[ $# -ge 2 ]]; then shift 2; elif [[ $# -eq 1 ]]; then shift; fi
BOOTSTRAP="${BOOTSTRAP:-$PROJECT/output/truck_clean_color_gpu1_20261014_103802/checkpoints/end_semantic_bootstrap/codec.pt}"
[[ -f "$BOOTSTRAP" ]] || { echo "Missing shared attribute initializer: $BOOTSTRAP (set BOOTSTRAP explicitly)" >&2; exit 1; }
RESUME_FLAG=()
if [[ "${RESUME:-0}" == 1 ]]; then RESUME_FLAG=(--resume); fi
exec "${PYTHON_BIN:-python}" -u -m gaussian_jscc.adversarial_training_controls \
  --run "$RUN" --out "$OUT" --bootstrap "$BOOTSTRAP" \
  --semantic-render-steps "${SEMANTIC_RENDER_STEPS:-1000}" \
  --channel-bootstrap-steps "${CHANNEL_STEPS:-2000}" --channel-render-steps "${JOINT_STEPS:-2000}" \
  --semantic-lr "${SEMANTIC_LR:-1e-4}" --channel-lr "${CHANNEL_LR:-1e-4}" \
  --lpips-weight "${LPIPS_WEIGHT:-1}" --gan-weight "${GAN_WEIGHT:-0.001}" \
  --discriminator-lr "${DISCRIMINATOR_LR:-1e-4}" \
  --channel-latent-weight 1 --validate-every 500 --save-every 500 --device cuda \
  "${RESUME_FLAG[@]}" "$@"
