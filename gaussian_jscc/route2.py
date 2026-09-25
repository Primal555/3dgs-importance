"""Save, load and export per-Gaussian four-way masks. Training: train-learned."""
import json
from pathlib import Path
import numpy as np
import torch
from .allocation import GaussianTierMask, scene_fingerprint, expected_rate
from .data import read_ply
from .transport import model_id, save_checkpoint, load_checkpoint


def save_joint(out, suffix, model, mask, fingerprint, step, record):
    save_checkpoint(out / f"codec{suffix}.pt", model, step, record)
    torch.save({"version": 1, "count": len(mask.logits), "scene_fingerprint": fingerprint,
                "snr_conditioned": mask.snr_slopes is not None,
                "state_dict": mask.state_dict(), "codec_id": model_id(model),
                "rates": list(model.cfg.rates), "step": step, "training": record},
               out / f"route2{suffix}.pt")


def load_mask(path, raw, model, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("version") != 1 or saved["scene_fingerprint"] != scene_fingerprint(raw):
        raise ValueError("route2 checkpoint does not match this PLY and row order")
    if saved["codec_id"] != model_id(model) or tuple(saved["rates"]) != model.cfg.rates:
        raise ValueError("route2 checkpoint requires its matching codec checkpoint")
    mask = GaussianTierMask(saved["count"], snr_conditioned=saved["snr_conditioned"], tier_count=len(model.cfg.rates))
    mask.load_state_dict(saved["state_dict"])
    return mask.to(device).eval()


@torch.no_grad()
def hard_tiers(mask, snr):
    device = mask.logits.device
    return torch.cat([mask.scores(torch.arange(start, min(start + 65536, len(mask.logits)), device=device), snr)
                     .argmax(-1).cpu() for start in range(0, len(mask.logits), 65536)])


def export_mask(args):
    from .cli import device_for
    device = device_for(args.device)
    raw, _ = read_ply(args.ply)
    model = load_checkpoint(args.checkpoint, device)
    mask = load_mask(args.allocation, raw, model, device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(raw), 65536):
            ids = torch.arange(start, min(start + 65536, len(raw)), device=device)
            probabilities.append(mask.probabilities(ids, args.snr).cpu())
    probabilities = torch.cat(probabilities)
    tiers = probabilities.argmax(-1).numpy().astype(np.uint8)
    np.save(out / "tiers.npy", tiers)
    np.save(out / "probabilities.npy", probabilities.numpy())
    info = {"snr_db": args.snr, "original_ply_row_order": True,
            "decision": "argmax; no hard total-budget guarantee",
            "scene_fingerprint": scene_fingerprint(raw), "codec_id": model_id(model),
            "rates": list(model.cfg.rates),
            "tier_counts": np.bincount(tiers, minlength=len(model.cfg.rates)).tolist(),
            "expected_payload_symbols": float(expected_rate(probabilities.double(), model.cfg.rates).sum()),
            "hard_payload_symbols": int(np.asarray(model.cfg.rates)[tiers].sum())}
    (out / "allocation.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    from .plots import safe_plot
    safe_plot("allocation", out, xyz=raw[:, :3].numpy(), rates=model.cfg.rates, snr=args.snr)


def add_parsers(sub):
    p = sub.add_parser("export-route2", help="export learned probabilities and hard q in original PLY order")
    p.set_defaults(func=export_mask)
    p.add_argument("--ply", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--allocation", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--snr", type=float, default=10.)
    p.add_argument("--device", default="cuda")
