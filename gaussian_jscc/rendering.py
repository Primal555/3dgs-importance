"""Differentiable rendering adapter; imported only for CUDA image training/evaluation."""

from pathlib import Path
from types import SimpleNamespace
import math

import torch


def load_cameras(source, resolution=2, white_background=False, images="images", split="test"):
    from scene.dataset_readers import sceneLoadTypeCallbacks
    from utils.camera_utils import cameraList_from_camInfos

    source = str(Path(source).resolve())
    if Path(source, "sparse").exists():
        info = sceneLoadTypeCallbacks["Colmap"](source, images, True)
    elif Path(source, "transforms_train.json").exists():
        info = sceneLoadTypeCallbacks["Blender"](source, white_background, True)
    else:
        raise ValueError("source must be a COLMAP or Blender scene")
    infos = info.train_cameras if split == "train" else info.test_cameras
    if not infos:
        raise ValueError(f"scene has no {split} cameras")
    args = SimpleNamespace(resolution=resolution, data_device="cpu")
    return cameraList_from_camInfos(infos, 1.0, args)


def render(raw, camera, degree, white_background=False, existence=None):
    if existence is None:
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    else:
        # Unlike multiplying opacity before the vanilla rasterizer's cutoff,
        # MaskGaussian's kernel also computes mask gradients for inactive splats.
        from mask_diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        if existence.shape != (len(raw),):
            raise ValueError("existence mask must have shape [N]")

    bg = raw.new_full((3,), float(white_background))
    if not len(raw):
        return bg[:, None, None].expand(3, camera.image_height, camera.image_width)
    settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height), image_width=int(camera.image_width),
        tanfovx=math.tan(camera.FoVx / 2), tanfovy=math.tan(camera.FoVy / 2),
        bg=bg, scale_modifier=1., viewmatrix=camera.world_view_transform.to(raw.device),
        projmatrix=camera.full_proj_transform.to(raw.device), sh_degree=degree,
        campos=camera.camera_center.to(raw.device), prefiltered=False, debug=False)
    dc = raw[:, 11:14].reshape(-1, 1, 3)
    rest = raw[:, 14:].reshape(len(raw), 3, (degree + 1) ** 2 - 1).transpose(1, 2)
    sh = torch.cat((dc, rest), 1).contiguous()
    image, _ = GaussianRasterizer(raster_settings=settings)(
        means3D=raw[:, :3].contiguous(), means2D=torch.zeros_like(raw[:, :3]),
        shs=sh, colors_precomp=None, opacities=raw[:, 3:4].sigmoid().contiguous(),
        scales=raw[:, 4:7].clamp(-20, 10).exp().contiguous(),
        rotations=torch.nn.functional.normalize(raw[:, 7:11], dim=-1).contiguous(),
        cov3D_precomp=None,
        **({} if existence is None else {"masks": existence[:, None].contiguous()}))
    return image


@torch.no_grad()
def evaluate_views(raw, reference, cameras, degree, white_background, directory=None, lpips=False):
    from utils.loss_utils import ssim
    from PIL import Image
    import numpy as np

    if directory:
        Path(directory).mkdir(parents=True, exist_ok=True)
    perceptual = None
    if lpips:
        from lpipsPyTorch.modules.lpips import LPIPS
        perceptual = LPIPS(net_type="vgg").to(raw.device).eval()
    rows = []
    for index, camera in enumerate(cameras):
        received = render(raw, camera, degree, white_background).clamp(0, 1)
        baseline = render(reference, camera, degree, white_background).clamp(0, 1)
        gt = camera.original_image[:3].to(raw.device)
        row = {"view": camera.image_name}
        for name, image in (("received", received), ("reference", baseline)):
            mse = (image - gt).square().mean().clamp_min(1e-12)
            row[name + "_psnr"] = float(-10 * mse.log10())
            row[name + "_ssim"] = float(ssim(image, gt))
            if perceptual is not None:
                row[name + "_lpips"] = float(perceptual(image[None] * 2 - 1, gt[None] * 2 - 1))
        codec_mse = (received - baseline).square().mean().clamp_min(1e-12)
        row["received_vs_reference_psnr"] = float(-10 * codec_mse.log10())
        row["received_vs_reference_ssim"] = float(ssim(received, baseline))
        row["received_vs_reference_l1"] = float((received - baseline).abs().mean())
        row["psnr_delta_received_minus_reference"] = row["received_psnr"] - row["reference_psnr"]
        row["ssim_delta_received_minus_reference"] = row["received_ssim"] - row["reference_ssim"]
        if perceptual is not None:
            row["received_vs_reference_lpips"] = float(
                perceptual(received[None] * 2 - 1, baseline[None] * 2 - 1))
            row["lpips_delta_received_minus_reference"] = row["received_lpips"] - row["reference_lpips"]
        rows.append(row)
        if directory:
            comparison = torch.cat((gt, baseline, received), dim=2)
            pixels = (comparison.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            Image.fromarray(pixels).save(Path(directory) / f"{index:05d}.png")
    keys = [k for k in rows[0] if k != "view"]
    return {"mean": {k: sum(r[k] for r in rows) / len(rows) for k in keys}, "views": rows,
            "lpips_input_range": "[-1,1]" if lpips else None}
