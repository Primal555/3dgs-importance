"""Exercise each native environment before transmitting a large scene."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role',choices=['evaluation','mesongs'],required=True)
    p.add_argument('--repo');p.add_argument('--source');p.add_argument('--ply')
    p.add_argument('--resolution',type=int,default=2);p.add_argument('--images',default='images')
    p.add_argument('--device',default='cuda');p.add_argument('--white-background',action='store_true')
    p.add_argument('--no-lpips',action='store_true')
    p.add_argument('--camera-names-json')
    args=p.parse_args()
    import torch
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError(f'{args.role} requires an available CUDA device in this environment')
    device=torch.device(args.device)
    if device.index is not None:torch.cuda.set_device(device)
    with torch.no_grad():
        if args.role=='evaluation':
            from gaussian_jscc.data import read_ply
            from gaussian_jscc.rendering import load_cameras,render
            from gaussian_jscc.perceptual_metrics import LPIPSEvaluator
            raw,degree=read_ply(args.ply)
            cameras=load_cameras(args.source,args.resolution,args.white_background,args.images,'test')
            if args.camera_names_json:
                names=json.loads(Path(args.camera_names_json).read_text(encoding='utf-8'))
                mapping={str(camera.image_name):camera for camera in cameras}
                if len(mapping)!=len(cameras) or not names or len(names)!=len(set(names)):
                    raise ValueError('test camera names must be nonempty and unique')
                missing=set(names)-mapping.keys()
                if missing:raise ValueError(f'saved test cameras missing from dataset: {sorted(missing)}')
                cameras=[mapping[name] for name in names]
            # Tiny real kernel launch checks the rasterizer ABI without a full-scene pass.
            image=render(raw[:1].to(device),cameras[0],degree,args.white_background)
            if not torch.isfinite(image).all():raise RuntimeError('rasterizer preflight returned invalid RGB')
            if not args.no_lpips:
                metric=LPIPSEvaluator(device)
                check=torch.zeros(3,32,32,device=device)
                if abs(metric(check,check))>1e-5:raise RuntimeError('LPIPS identity check failed')
            details=dict(points=len(raw),test_views=len(cameras),degree=degree)
        else:
            if not args.repo:raise ValueError('--repo required for MesonGS')
            sys.path.insert(0,str(Path(args.repo).resolve()))
            import diff_gaussian_rasterization
            from simple_knn._C import distCUDA2
            import weighted_distance._C
            import scene.gaussian_model
            # Importing the three extensions also verifies their shared-library dependencies.
            details=dict(extensions=['diff_gaussian_rasterization','simple_knn','weighted_distance'])
        torch.cuda.synchronize(device)
    print(json.dumps(dict(status='ready',role=args.role,torch=torch.__version__,cuda=torch.version.cuda,
                          device=torch.cuda.get_device_name(device),**details)))


if __name__=='__main__':main()
