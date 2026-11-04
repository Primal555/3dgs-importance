"""Evaluate received PLYs with the main project's cameras and image protocol.

Run in the main MaskGaussian environment, for example:
python -m comparison_baselines.evaluate --source data/scene --source-ply source.ply \
    --received-ply received.ply --out results/evaluation
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


_PREFIXES = ('source_', 'photo_', 'source_teacher_photo_')
_SCORES = ('mse', 'psnr', 'ssim', 'l1', 'lpips')


def aggregate_view_metrics(rows):
    """Equal view weighting; reject missing scores and nonfinite results."""
    if not rows:
        raise ValueError('evaluation requires at least one view')
    allowed = {prefix + score for prefix in _PREFIXES for score in _SCORES}
    keys = set(rows[0]) & allowed
    if not keys:
        raise ValueError('views contain no image metrics')
    for row in rows:
        if set(row) & allowed != keys:
            raise ValueError('all views must contain the same image metrics')
        for key in keys:
            if isinstance(row[key], bool) or not math.isfinite(float(row[key])):
                raise ValueError(f'nonfinite or invalid metric {key} in view {row.get("view")}')
    result = {key: math.fsum(float(row[key]) / len(rows) for row in rows)
              for key in sorted(keys)}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError('nonfinite aggregated image metrics')
    return result


def _file_identity(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def _tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _validate_image(image, target, name):
    if (not isinstance(image, torch.Tensor) or image.ndim != 3 or image.shape[0] != 3
            or image.shape != target.shape or not image.numel()
            or not torch.isfinite(image).all()):
        raise ValueError(f'{name} must be finite RGB with the same dimensions as the photograph')


def _validated_camera_names(names):
    if not isinstance(names, list) or not names or not all(isinstance(name, str) and name for name in names):
        raise ValueError('camera names must be a nonempty JSON list of nonempty strings')
    if len(set(names)) != len(names):
        raise ValueError('requested camera names contain duplicate names')
    return names


def _save_panel(path, name, photo, teacher, received):
    from PIL import Image, ImageDraw, PngImagePlugin
    import numpy as np

    width, height = photo.shape[2], photo.shape[1]
    header = 44
    panel = Image.new('RGB', (3 * width, height + header), 'white')
    draw = ImageDraw.Draw(panel)
    # Keep the original name in PNG metadata even when the default font cannot
    # display a dataset's Unicode camera name.
    title = f'view: {name}'
    try:
        draw.text((4, 2), title, fill='black')
    except UnicodeEncodeError:
        draw.text((4, 2), title.encode('ascii', 'backslashreplace').decode(), fill='black')
    for index, (label, image) in enumerate((('photo', photo), ('source teacher', teacher),
                                            ('received', received))):
        pixels = (image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        panel.paste(Image.fromarray(pixels), (index * width, header))
        draw.text((index * width + 4, 24), label, fill='black')
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text('view', str(name))
    metadata.add_text('panel_order', 'photo, source teacher, received')
    panel.save(path, pnginfo=metadata)


@torch.no_grad()
def evaluate_plys(source, source_ply, received_ply, out, *, resolution=2, images='images',
                  views=32, image_views=2, lpips=True, white_background=False,
                  device='cuda', renderer=None, camera_names=None):
    """Score independent PLY scenes against the same photographs and teacher.

    Only ``received_ply`` reconstructs the received scene. The source is rendered
    separately as an evaluation teacher. ``renderer`` permits CPU contract tests;
    the default renderer requires CUDA. Camera loading and image scores always
    reuse the main project's implementations. Explicit ``camera_names`` select
    exactly those cameras in their saved order, without a second view subsample.
    """
    from gaussian_jscc.data import read_ply
    from gaussian_jscc.rendering import load_cameras, render
    from gaussian_jscc.render_objective import image_metrics, spaced_indices
    from gaussian_jscc.perceptual_metrics import LPIPSEvaluator, image_metric_protocol

    if not isinstance(views, int) or views < 1:
        raise ValueError('views must be a positive integer')
    if not isinstance(image_views, int) or image_views < 0:
        raise ValueError('image_views must be a nonnegative integer')
    if not isinstance(resolution, int) or (resolution < 1 and resolution != -1):
        raise ValueError('resolution must be a positive integer or -1')
    if camera_names is not None:
        camera_names = _validated_camera_names(camera_names)
    source, source_ply, received_ply, out = (Path(p).resolve()
                                            for p in (source, source_ply, received_ply, out))
    if not source.is_dir():
        raise ValueError(f'source photograph directory does not exist: {source}')
    for path in (source_ply, received_ply):
        if not path.is_file():
            raise ValueError(f'input PLY does not exist: {path}')
    if (out / 'metrics.json').exists():
        raise FileExistsError(f'use an independent evaluation directory: {out}')
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA evaluation requires the main MaskGaussian CUDA environment')
    if renderer is None and device.type != 'cuda':
        raise ValueError('the main project Gaussian renderer requires --device cuda')
    renderer = render if renderer is None else renderer
    if device.type == 'cuda' and device.index is not None:
        # Camera construction uses Tensor.cuda() with the current CUDA device.
        torch.cuda.set_device(device)
    all_cameras = load_cameras(source, resolution=resolution, white_background=white_background,
                               images=images, split='test')
    loaded_names = [str(camera.image_name) for camera in all_cameras]
    if len(set(loaded_names)) != len(loaded_names):
        raise ValueError('loaded test cameras contain duplicate names')
    if camera_names is None:
        indices = spaced_indices(len(all_cameras), views)
        selection = 'gaussian_jscc.render_objective.spaced_indices'
    else:
        index_by_name = {name: index for index, name in enumerate(loaded_names)}
        missing = [name for name in camera_names if name not in index_by_name]
        if missing:
            raise ValueError(f'requested camera names are missing from test cameras: {missing}')
        indices = [index_by_name[name] for name in camera_names]
        selection = 'explicit_camera_names'
    cameras = [all_cameras[index] for index in indices]
    del all_cameras
    panel_indices = set(spaced_indices(len(cameras), min(image_views, 2))) if image_views else set()
    metric = LPIPSEvaluator(device) if lpips else None

    def scores(image, target, prefix):
        result = {prefix + key: value for key, value in image_metrics(image, target).items()}
        if metric is not None:
            result[prefix + 'lpips'] = metric(image, target)
        return result

    identity = {'source': str(source), 'source_ply': _file_identity(source_ply),
                'received_ply': _file_identity(received_ply), 'resolution': resolution,
                'images': images, 'white_background': bool(white_background),
                'camera_split': 'test', 'camera_selection': selection,
                'requested_views': views, 'camera_indices': indices,
                'camera_names': [str(camera.image_name) for camera in cameras],
                'cameras': [], 'device': str(device),
                'renderer': {'implementation': f'{renderer.__module__}.{renderer.__qualname__}',
                             'backend': 'diff_gaussian_rasterization',
                             'torch_version': str(torch.__version__), 'cuda_version': torch.version.cuda,
                             'code_sha256': _file_identity(Path(__file__).parents[1] / 'gaussian_jscc' / 'rendering.py')['sha256']},
                'image_metric_protocol': image_metric_protocol(lpips),
                'aggregation': 'arithmetic mean of per-view scores',
                'source_ply_role': 'evaluation teacher only'}
    rows, teacher_images, panels = [], [], []

    # Render and release the teacher scene before reading the received PLY.
    # CPU teacher images avoid keeping two complete scenes resident on the GPU.
    raw, degree = read_ply(source_ply)
    identity['source_ply'].update(count=len(raw), sh_degree=degree)
    scene = raw.to(device)
    del raw
    for index, camera in enumerate(cameras):
        photo = camera.original_image[:3].detach().to(device)
        _validate_image(photo, photo, 'photograph')
        teacher = renderer(scene, camera, degree, white_background)
        _validate_image(teacher, photo, 'source teacher render')
        identity['cameras'].append({'index': indices[index], 'name': str(camera.image_name),
                                    'width': int(photo.shape[2]), 'height': int(photo.shape[1]),
                                    'photo_rgb_sha256': _tensor_hash(photo),
                                    'world_view_sha256': _tensor_hash(camera.world_view_transform),
                                    'projection_sha256': _tensor_hash(camera.full_proj_transform),
                                    'camera_center_sha256': _tensor_hash(camera.camera_center),
                                    'FoVx': float(camera.FoVx), 'FoVy': float(camera.FoVy)})
        rows.append({'view': str(camera.image_name), 'camera_index': indices[index],
                     **scores(teacher, photo, 'source_teacher_photo_')})
        teacher_images.append(teacher.detach().cpu())
        del teacher, photo
    del scene
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    raw, degree = read_ply(received_ply)
    identity['received_ply'].update(count=len(raw), sh_degree=degree)
    scene = raw.to(device)
    del raw
    for index, camera in enumerate(cameras):
        photo = camera.original_image[:3].detach().to(device)
        teacher = teacher_images[index].to(device)
        received = renderer(scene, camera, degree, white_background)
        _validate_image(received, photo, 'received render')
        rows[index].update(scores(received, teacher, 'source_'))
        rows[index].update(scores(received, photo, 'photo_'))
        if index in panel_indices:
            panels.append((index, str(camera.image_name), photo.cpu(), teacher.cpu(), received.cpu()))
        del photo, teacher, received
    del scene, teacher_images
    metrics = aggregate_view_metrics(rows)
    out.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for index, name, photo, teacher, received in panels:
        safe_name = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in name)[:100] or 'view'
        filename = f'view_{index:03d}_{safe_name}.png'
        _save_panel(out / filename, name, photo, teacher, received)
        image_paths.append(filename)
    result = {'identity': identity, 'metrics': metrics, 'views': rows, 'image_paths': image_paths,
              'metric_scope': 'successfully received and reconstructed PLY'}
    temporary = out / 'metrics.json.tmp'
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                         encoding='utf-8')
    temporary.replace(out / 'metrics.json')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='Unified received-PLY evaluation in the MaskGaussian environment')
    parser.add_argument('--source', required=True, help='COLMAP/Blender photograph scene directory')
    parser.add_argument('--source-ply', required=True, help='shared pruned PLY, evaluation teacher only')
    parser.add_argument('--received-ply', required=True, help='received and reconstructed PLY')
    parser.add_argument('--out', required=True, help='independent evaluation output directory')
    parser.add_argument('--resolution', type=int, default=2)
    parser.add_argument('--images', default='images')
    parser.add_argument('--views', type=int, default=32)
    parser.add_argument('--camera-names-json', help='JSON string list of exact test camera names, in evaluation order')
    parser.add_argument('--image-views', type=int, default=2)
    parser.add_argument('--no-lpips', action='store_true')
    parser.add_argument('--white-background', action='store_true')
    parser.add_argument('--device', default='cuda')
    args = vars(parser.parse_args(argv))
    names_path = args.pop('camera_names_json')
    if names_path is not None:
        args['camera_names'] = _validated_camera_names(json.loads(Path(names_path).read_text(encoding='utf-8')))
    args['lpips'] = not args.pop('no_lpips')
    result = evaluate_plys(**args)
    print(json.dumps({'metrics': result['metrics'], 'metrics_path': str(Path(args['out']) / 'metrics.json')},
                     ensure_ascii=False, allow_nan=False))
    return result


if __name__ == '__main__':
    main()
