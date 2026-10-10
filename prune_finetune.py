#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import json
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from lpipsPyTorch import lpips
from gaussian_renderer import render, render_orig_3dgs, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from icecream import ic
import random
import copy
import gc
from os import makedirs
import torchvision
from torch.optim.lr_scheduler import ExponentialLR
import csv
from train import training_report, prepare_output_and_logger
from utils.prune_initialization import initialize_from_ply, ply_info, quantized_photo_metrics


to_tensor = (
    lambda x: x.to("cuda")
    if isinstance(x, torch.Tensor)
    else torch.Tensor(x).to("cuda")
)
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10.0 * torch.log(x) / torch.log(to_tensor([10.0]))


@torch.no_grad()
def baseline_metrics(scene, gaussians, pipe, background):
    """Initial PLY versus the same held-out photos used by final metrics.py.

    No random masks or image export: this measures the unmodified input PLY.
    """
    cameras = scene.getTestCameras()
    if not cameras:
        raise ValueError('No held-out cameras; use --eval and a valid source scene')
    totals = {'PSNR': 0.0, 'SSIM': 0.0}
    for camera in cameras:
        image = render_orig_3dgs(camera, gaussians, pipe, background)['render'].clamp(0, 1)
        metrics = quantized_photo_metrics(image, camera.original_image.to(image.device))
        totals['PSNR'] += metrics['PSNR']
        totals['SSIM'] += metrics['SSIM']
    return {**{key: value/len(cameras) for key, value in totals.items()},
            'views': len(cameras), 'reference': 'held-out photos', 'precision': '8-bit RGB'}


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    args,
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, initialize_gaussians=not bool(args.start_pointcloud))
    if checkpoint:
        gaussians.training_setup(opt)
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore_from_3dgs(model_params, opt)
        ic(f"Loaded Gaussians. Number of Gaussians:{gaussians._xyz.shape[0]}")
    elif args.start_pointcloud:
        initialize_from_ply(gaussians, args.start_pointcloud, opt, scene.cameras_extent)
        first_iter = args.ply_iteration
        print(f'PLY initialized: {len(gaussians._xyz):,} Gaussians; mask scores [10, 1]; fresh Adam state.')
        print(f'Post-training: iteration {first_iter} -> {opt.iterations} ({opt.iterations-first_iter} updates).')
    else:
        raise ValueError('Provide --start_checkpoint or --start_pointcloud')
    if first_iter >= opt.iterations:
        raise ValueError('Final --iterations must exceed the initializer iteration')

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    if args.start_pointcloud:
        initialization = {'input_ply': os.path.abspath(args.start_pointcloud),
                          'input_points': len(gaussians._xyz), 'start_iteration': first_iter,
                          'final_iteration': opt.iterations, 'optimizer': 'fresh Adam; PLY has no optimizer state',
                          'mask_logits': [10., 1.], 'lambda_mask': opt.lambda_mask,
                          'lambda_dssim': opt.lambda_dssim, 'spatial_lr_scale': scene.cameras_extent}
        with open(os.path.join(scene.model_path, 'initialization.json'), 'w') as fp:
            json.dump(initialization, fp, indent=2)
        if dataset.eval:
            initial_quality = baseline_metrics(scene, gaussians, pipe, background)
            with open(os.path.join(scene.model_path, 'baseline_metrics.json'), 'w') as fp:
                json.dump(initial_quality, fp, indent=2)
            print(f'Input PLY baseline: PSNR={initial_quality["PSNR"]:.3f}, SSIM={initial_quality["SSIM"]:.4f}')

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    gaussians.scheduler = ExponentialLR(gaussians.optimizer, gamma=0.95)

    for iteration in range(first_iter, opt.iterations + 1):
        if not args.disable_gui and network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe.convert_SHs_python,
                    pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()
                if custom_cam != None:
                    net_image = render(
                        custom_cam, gaussians, pipe, background, scaling_modifer
                    )["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and (
                    (iteration < int(opt.iterations)) or not keep_alive
                ):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        if iteration % 400 == 0:
            gaussians.scheduler.step()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii, mask = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
            render_pkg["mask"],
        )

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        lambda_mask = opt.lambda_mask # *  (iteration - 30000) / 5000 #if iteration <= 31000 else 0
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
            1.0 - ssim(image, gt_image) + lambda_mask * (torch.mean(mask))**2
        )

        loss.backward()

        iter_end.record()
        # A PLY-initialized run performs all requested updates, including its last
        # one, before validation/export. Leave the legacy checkpoint route intact.
        if args.start_pointcloud:
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            num_used_gs = int(mask.detach().sum())
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",  "num_used_gs": num_used_gs})
                progress_bar.update(10)
            if args.start_pointcloud and (iteration % 10 == 0 or iteration == opt.iterations):
                record = {'iteration': iteration, 'extra_step': iteration-args.ply_iteration,
                          'loss': loss.item(), 'l1': Ll1.item(), 'sampled_retained': num_used_gs,
                          'total_parameters': len(gaussians._xyz),
                          'mask_mean': mask.detach().mean().item(),
                          'gpu_ms': iter_start.elapsed_time(iter_end),
                          'learning_rates': {group['name']: group['lr'] for group in gaussians.optimizer.param_groups}}
                with open(os.path.join(scene.model_path, 'loss.jsonl'), 'a') as fp:
                    fp.write(json.dumps(record)+'\n')
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save                

            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
                num_used_gs
            )

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                if not os.path.exists(scene.model_path):
                    os.makedirs(scene.model_path)
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )

            if not args.start_pointcloud and iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[30_001, 30_002, 30_500, 31_000, 31_500, 32_000, 32_500, 33_000, 33_500, 34_000, 34_500, 35_000]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[35_000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--checkpoint_iterations", nargs="+", type=int, default=[]
    )

    initializer = parser.add_mutually_exclusive_group(required=True)
    initializer.add_argument("--start_checkpoint", type=str, default=None)
    initializer.add_argument("--start_pointcloud", type=str, default=None,
                             help="Initialize raw Gaussian parameters from a standard pretrained PLY, with fresh Adam")
    parser.add_argument('--ply_iteration', type=int, default=30000,
                        help='Absolute initializer iteration, NOT a number of extra training steps')
    parser.add_argument('--disable_gui', action='store_true', help='Do not open a GUI listener during batch processing')
    parser.add_argument("--densify_iteration", nargs="+", type=int, default=[-1])
    args = parser.parse_args(sys.argv[1:])
    if args.start_pointcloud:
        if args.ply_iteration < 0 or args.iterations <= args.ply_iteration:
            parser.error('--iterations must exceed nonnegative --ply_iteration')
        info = ply_info(args.start_pointcloud)
        if info['sh_degree'] != args.sh_degree:
            parser.error(f'PLY SH degree={info["sh_degree"]}; pass matching --sh_degree')
        if os.path.exists(args.model_path):
            parser.error('PLY post-training requires a NEW output directory; source PLYs are never overwritten')
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_gui:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args,
    )

    # All done
    print("\nTraining complete.")
