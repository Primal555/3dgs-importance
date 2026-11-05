"""Scene compression -> real digital file transport -> shared rendering metrics."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile

from .packing import pack_ply, unpack_ply, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', help='shared codec run; reads its exact source scenes and camera settings')
    p.add_argument('--manifest');p.add_argument('--scene')
    p.add_argument('--ply');p.add_argument('--source');p.add_argument('--out',required=True)
    p.add_argument('--methods',nargs='+',choices=['maskgaussian','mesongs'],default=['maskgaussian','mesongs'])
    p.add_argument('--meson-configs',nargs='+',choices=['c1','c3'],default=['c1'])
    p.add_argument('--meson-iterations',type=int,default=0)
    p.add_argument('--snrs',nargs='+',type=float,default=[10.])
    for key,value in [('trials',4),('seed',42),('k',4096),('n',6144),
                      ('bits-per-symbol',4),('decoder-iterations',20),('batch-size',32)]:
        p.add_argument('--'+key,type=int,default=value)
    p.add_argument('--resolution',type=int);p.add_argument('--images');p.add_argument('--views',type=int)
    p.add_argument('--image-views',type=int,default=2)
    p.add_argument('--white-background',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--lpips',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--device',default='cuda');p.add_argument('--link-device',default='cuda')
    p.add_argument('--digital-python',default=os.environ.get('DIGITAL_PYTHON',sys.executable))
    p.add_argument('--mesongs-python',default=os.environ.get('MESONGS_PYTHON',sys.executable))
    p.add_argument('--eval-python',default=os.environ.get('EVAL_PYTHON',sys.executable))
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--keep-intermediates',action='store_true')
    return p


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _dataset_identity(source,images):
    """Track calibration bytes and photograph inventory without retaining images."""
    import hashlib
    calibration=[]
    for relative in ('sparse/0/cameras.bin','sparse/0/images.bin','sparse/0/cameras.txt',
                     'sparse/0/images.txt','transforms_train.json','transforms_test.json'):
        path=source/relative
        if path.is_file():calibration.append((relative,sha256_file(path)))
    photographs=set()
    image_dir=source/images
    if image_dir.is_dir():
        photographs.update(p.resolve() for p in image_dir.rglob('*') if p.is_file())
    for split in ('train','test'):
        transforms=source/f'transforms_{split}.json'
        if transforms.is_file():
            for frame in _read(transforms).get('frames',[]):
                # The project's Blender loader appends .png to these paths.
                path=source/(frame['file_path']+'.png')
                if path.is_file():photographs.add(path.resolve())
    inventory=[]
    for path in sorted(photographs):
        inventory.append((os.path.relpath(path,source),path.stat().st_size,sha256_file(path)))
    return hashlib.sha256(json.dumps([calibration,inventory],sort_keys=True).encode()).hexdigest()


def _json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temporary,path)


def build_plan(args):
    settings={};selected=None;record=None;scene_records={};run_directory=None
    scene_name=args.scene or 'truck'
    if args.manifest:
        path=Path(args.manifest).resolve();record=_read(path)
        base=(path.parent/record.get('root','.')).resolve()
        entries=record['scenes']
    elif args.run:
        run=Path(args.run).resolve()
        run_directory=run
        candidates=[run/'training.json',run/'training/training.json']
        path=next((p for p in candidates if p.is_file()),None)
        if path:
            record=_read(path)
            settings={key:record[key] for key in ('resolution','images','white_background','lpips','test_views') if key in record}
            settings.update(record.get('protocol',{}));settings.update(record.get('settings',{}))
            entries=list(record.get('selected_scenes',[]))
            scene_records=record.get('scene_records',{})
            for key in ('heldout_scene','heldout_scenes'):
                held=record.get(key)
                if isinstance(held,dict):held=[held]
                if isinstance(held,list):
                    for entry in held:
                        if isinstance(entry,dict) and entry not in entries:entries.append(entry)
            base=path.parent
        else:
            path=run/'dataset.json';record=_read(path)
            base=(path.parent/record.get('root','.')).resolve();entries=record['scenes']
    else:entries=[];base=ROOT
    if entries:
        matches=[e for e in entries if isinstance(e,dict) and e.get('name')==scene_name]
        if len(matches)!=1:raise ValueError(f'need exactly one source record for scene {scene_name}')
        selected=matches[0]
    if not selected and not (args.ply and args.source):
        raise ValueError('provide --run/--manifest or both --ply and --source')
    ply=Path(args.ply).resolve() if args.ply else (base/selected['ply']).resolve()
    source=Path(args.source).resolve() if args.source else (base/selected['source']).resolve()
    if not ply.is_file():raise FileNotFoundError(f'missing input PLY: {ply}')
    if not source.is_dir():raise FileNotFoundError(f'missing photograph directory: {source}')
    if args.ply and args.source and args.scene is None and not entries:scene_name=source.name
    def option(name,fallback):
        value=getattr(args,name)
        return value if value is not None else settings.get(name,fallback)
    views=args.views if args.views is not None else settings.get('test_views',32)
    scene_record=scene_records.get(scene_name,{})
    if not scene_record and run_directory:
        for saved in (run_directory/'scenes'/scene_name/'scene.json',
                      run_directory/'training/scenes'/scene_name/'scene.json'):
            if saved.is_file():scene_record=_read(saved);break
    camera_names=scene_record.get('test_view_names')
    if camera_names is not None:
        if not isinstance(camera_names,list) or not camera_names or not all(isinstance(n,str) for n in camera_names) or len(set(camera_names))!=len(camera_names):
            raise ValueError('saved test camera names must be a nonempty unique string list')
        if args.views is not None:
            from gaussian_jscc.render_objective import spaced_indices
            camera_names=[camera_names[i] for i in spaced_indices(len(camera_names),args.views)]
        views=len(camera_names)
    evaluation=dict(resolution=option('resolution',2),images=option('images','images'),views=views,
                    image_views=args.image_views,white_background=option('white_background',False),
                    lpips=option('lpips',True),device=args.device,camera_names=camera_names)
    link=dict(k=args.k,n=args.n,bits_per_symbol=args.bits_per_symbol,
              decoder_iterations=args.decoder_iterations,batch_size=args.batch_size,device=args.link_device)
    if (args.bits_per_symbol not in (2,4,6,8) or args.k%8 or args.k<=32
        or args.n<=args.k or args.n%args.bits_per_symbol or args.decoder_iterations<=0 or args.batch_size<=0):
        raise ValueError('invalid byte-aligned LDPC/QAM parameters')
    if (args.trials<=0 or args.meson_iterations<0 or
        (evaluation['resolution']<=0 and evaluation['resolution']!=-1) or views<=0
        or not 0<=args.image_views<=2 or not args.snrs or not all(math.isfinite(x) for x in args.snrs)):
        raise ValueError('invalid trial, SNR, resolution or view count')
    if len(set(args.snrs))!=len(args.snrs) or len(set(args.methods))!=len(args.methods):
        raise ValueError('SNRs and methods must be unique')
    if len(set(args.meson_configs))!=len(args.meson_configs):raise ValueError('MesonGS configs must be unique')
    upstream={}
    for name in ['MaskGaussian','MesonGS']:
        path=ROOT/'comparison_baselines'/name
        if name.lower() not in args.methods:continue
        try:revision=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        except subprocess.CalledProcessError as exc:raise RuntimeError('run scripts/prepare_comparison_sources.sh first') from exc
        if not (path/'README.md').is_file():raise RuntimeError(f'empty source submodule: {path}')
        upstream[name]=revision
    return dict(version='digital_scene_delivery_v1',scene=scene_name,
                input=dict(ply=str(ply),source=str(source),sha256=sha256_file(ply),bytes=ply.stat().st_size,
                           dataset_identity=_dataset_identity(source,evaluation['images'])),
                evaluation=evaluation,link=link,snrs=args.snrs,trials=args.trials,seed=args.seed,
                methods=args.methods,meson_configs=args.meson_configs,meson_iterations=args.meson_iterations,
                upstream=upstream,interpreters=dict(digital=args.digital_python,mesongs=args.mesongs_python,
                                                  evaluation=args.eval_python),
                quality_conditioning='successful_file_delivery',arq=False)


def validate_resume(out,plan):
    if _read(Path(out)/'plan.json')!=plan:
        raise ValueError('resume plan differs: keep source, codec, link and evaluation settings identical')


def snr_label(snr):
    return repr(float(snr))


def _subprocess_environment(python):
    env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONUTF8='1')
    executable=shutil.which(str(python)) or str(python)
    # Selecting another environment's Python does not select its C++ runtime.
    # Set the loader path before the child starts importing native extensions.
    library=Path(executable).absolute().parent.parent/'lib'
    if library.is_dir():
        paths=[str(library),*env.get('LD_LIBRARY_PATH','').split(os.pathsep)]
        env['LD_LIBRARY_PATH']=os.pathsep.join(dict.fromkeys(p for p in paths if p))
    return env


def _execute(python,module,arguments,log):
    command=[python,'-u','-m',module,*map(str,arguments)]
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    print(shlex.join(command),flush=True)
    env=_subprocess_environment(python)
    with log.open('a',encoding='utf-8') as handle:
        handle.write('\n'+shlex.join(command)+'\n');handle.flush()
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 text=True,encoding='utf-8',errors='replace')
        for line in process.stdout:
            print(line,end='',flush=True);handle.write(line);handle.flush()
        code=process.wait()
    if code:raise subprocess.CalledProcessError(code,command)


def summarize_trials(rows):
    groups={}
    for row in rows:groups.setdefault((row['method'],row['snr_db']),[]).append(row)
    results=[]
    for (method,snr),items in groups.items():
        successes=[r for r in items if r['status']=='success']
        metrics=None;deviations=None
        if successes:
            keys=set.intersection(*(set(r['metrics']) for r in successes))
            metrics={key:statistics.mean(r['metrics'][key] for r in successes) for key in sorted(keys)}
            deviations={key:statistics.pstdev(r['metrics'][key] for r in successes) for key in sorted(keys)}
        results.append(dict(method=method,snr_db=snr,trials=len(items),successes=len(successes),
                            success_rate=len(successes)/len(items),metrics=metrics,metrics_std=deviations,
                            total_channel_uses=statistics.mean(r['total_channel_uses'] for r in items),
                            quality_conditioning='successful_file_delivery'))
    return results


def _save_results(out,plan,rows,status):
    summary=dict(status=status,plan=plan,results=summarize_trials(rows),trials=rows)
    _json(out/'summary.json',summary)
    flat=[]
    for row in summary['results']:
        item={k:v for k,v in row.items() if k not in ('metrics','metrics_std')}
        item.update(row['metrics'] or {});item.update({k+'_std':v for k,v in (row['metrics_std'] or {}).items()})
        flat.append(item)
    if flat:
        fields=sorted(set().union(*(set(r) for r in flat)))
        with (out/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(flat)


def _review_archive(out):
    temporary=out/'review.tar.gz.tmp'
    with tarfile.open(temporary,'w:gz') as archive:
        candidates=[out/'plan.json',out/'summary.json',out/'comparison.csv',out/'complete.json',out/'evaluation_cameras.json']
        candidates+=list((out/'results').rglob('*')) if (out/'results').exists() else []
        if (out/'artifacts').exists():
            for name in ['sender.json','encode.json','quality.json']:candidates+=list((out/'artifacts').glob('*/'+name))
        for path in candidates:
            if path.is_file():archive.add(path,arcname=str(path.relative_to(out)),recursive=False)
    os.replace(temporary,out/'review.tar.gz')


def _eval_arguments(plan,received,out,image_views,*,camera_file=None):
    settings=plan['evaluation']
    args=['--source',plan['input']['source'],'--source-ply',plan['input']['ply'],
          '--received-ply',received,'--out',out,'--resolution',settings['resolution'],
          '--images',settings['images'],'--views',settings['views'],'--image-views',image_views,
          '--device',settings['device']]
    if settings['white_background']:args.append('--white-background')
    if not settings['lpips']:args.append('--no-lpips')
    if camera_file is not None:args.extend(['--camera-names-json',camera_file])
    return args


def _cached_quality(path,plan,metadata):
    if not path.is_file():return None
    cache=_read(path)
    expected=dict(payload_sha256=metadata['payload_sha256'],input=plan['input'],evaluation=plan['evaluation'])
    if cache['identity']!=expected:raise ValueError('cached digital reconstruction/evaluation identity changed')
    return cache


def execute(args,plan):
    out=Path(args.out).resolve()
    if args.resume:validate_resume(out,plan)
    else:
        out.mkdir(parents=True,exist_ok=False);_json(out/'plan.json',plan)
    rows=[]
    camera_file=None
    if plan['evaluation']['camera_names'] is not None:
        camera_file=out/'evaluation_cameras.json'
        _json(camera_file,plan['evaluation']['camera_names'])
    # Check every environment before starting the long file transmissions.
    native=['--source',plan['input']['source'],'--ply',plan['input']['ply'],
            '--resolution',plan['evaluation']['resolution'],'--images',plan['evaluation']['images'],
            '--device',args.device]
    if plan['evaluation']['white_background']:native.append('--white-background')
    if camera_file is not None:native.extend(['--camera-names-json',camera_file])
    _execute(args.eval_python,'comparison_baselines.preflight',
             ['--role','evaluation',*native,*([] if plan['evaluation']['lpips'] else ['--no-lpips'])],
             out/'logs/evaluation_preflight.log')
    if 'mesongs' in args.methods:
        _execute(args.mesongs_python,'comparison_baselines.preflight',
                 ['--role','mesongs','--repo',ROOT/'comparison_baselines/MesonGS','--device',args.device],
                 out/'logs/mesongs_preflight.log')
    _execute(args.digital_python,'comparison_baselines.digital_link',
             ['self-test','--device',args.link_device,'--snr-db',30],out/'logs/digital_preflight.log')
    cases=[]
    if 'maskgaussian' in args.methods:cases.append(('maskgaussian',None))
    if 'mesongs' in args.methods:cases.extend(('mesongs_'+c,c) for c in args.meson_configs)
    _save_results(out,plan,rows,'running')
    for method,config in cases:
        folder=out/'artifacts'/method
        sender=folder/'sender.json'
        if sender.exists():
            metadata=_read(sender);payload=Path(metadata['payload'])
            if metadata['payload_sha256']!=sha256_file(payload):raise ValueError('saved compressed payload changed')
        else:
            import shutil
            stage_root=out/'scratch';stage_root.mkdir(exist_ok=True)
            folder.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='encode-',dir=stage_root) as temporary:
                stage=Path(temporary).resolve()
                if not stage.is_relative_to(stage_root.resolve()):raise ValueError('encoding scratch outside run')
                if config is None:
                    staged=stage/'scene.ply.zlib';metadata=pack_ply(plan['input']['ply'],staged)
                    if metadata['source_sha256']!=plan['input']['sha256']:
                        raise ValueError('source PLY changed after planning')
                else:
                    encoded=stage/'encoded'
                    _execute(args.mesongs_python,'comparison_baselines.mesongs_adapter',
                             ['encode','--repo',ROOT/'comparison_baselines/MesonGS','--source',plan['input']['source'],
                              '--ply',plan['input']['ply'],'--out',encoded,'--config',config,
                              '--iterations',args.meson_iterations,'--resolution',plan['evaluation']['resolution'],
                              '--images',plan['evaluation']['images'],'--device',args.device,
                              *(['--white-background'] if plan['evaluation']['white_background'] else [])],
                             out/'logs'/f'{method}_encode.log')
                    staged=encoded/'bins.zip';metadata=_read(encoded/'encode.json')
                    if metadata['input_sha256']!=plan['input']['sha256']:
                        raise ValueError('MesonGS input differs from the planned PLY')
                    shutil.copy2(encoded/'encode.json',folder/'encode.json')
                payload=folder/staged.name
                shutil.copy2(staged,folder/(staged.name+'.tmp'))
                os.replace(folder/(staged.name+'.tmp'),payload)
            metadata.update(payload=str(payload),payload_sha256=sha256_file(payload),
                            source_sha256=plan['input']['sha256'],payload_bytes=payload.stat().st_size)
            _json(sender,metadata)
        quality_path=folder/'quality.json'
        quality=_cached_quality(quality_path,plan,metadata)
        for snr in args.snrs:
            saved_images=False
            for trial in range(args.trials):
                tag=snr_label(snr)
                result_dir=out/'results'/method/f'snr_{tag}'/f'trial_{trial:03d}'
                result_file=result_dir/'result.json'
                if result_file.exists():
                    row=_read(result_file);rows.append(row)
                    saved_images|=bool(row.get('images_saved'));continue
                result_dir.mkdir(parents=True,exist_ok=True)
                scratch_root=out/'scratch';scratch_root.mkdir(exist_ok=True)
                with tempfile.TemporaryDirectory(prefix=method+'-',dir=scratch_root) as temporary:
                    scratch=Path(temporary).resolve()
                    if not scratch.is_relative_to(scratch_root.resolve()):raise ValueError('scratch directory outside run')
                    received=scratch/payload.name
                    link_args=['transmit','--source',payload,'--destination',received,'--report',result_dir/'link.json',
                               '--snr-db',snr,'--seed',args.seed+trial]
                    for key,value in plan['link'].items():link_args.extend(['--'+key.replace('_','-'),value])
                    _execute(args.digital_python,'comparison_baselines.digital_link',link_args,
                             out/'logs'/f'{method}_{tag}_{trial:03d}_link.log')
                    link=_read(result_dir/'link.json')
                    if link['status'] not in ('success','failed'):raise ValueError('unknown transport status')
                    row=dict(method=method,snr_db=snr,trial=trial,status=link['status'],
                             total_channel_uses=link['total_channel_uses'],metrics=None,images_saved=False,
                             payload_bytes=payload.stat().st_size,link_report=str((result_dir/'link.json').relative_to(out)))
                    if link['status']=='success':
                        quality_reused=quality is not None
                        if sha256_file(received)!=metadata['payload_sha256']:
                            raise ValueError('successful reception differs from the encoded payload')
                        restored=scratch/'received.ply'
                        if quality is None or args.keep_intermediates:
                            if config is None:unpack_ply(received,restored)
                            else:
                                _execute(args.mesongs_python,'comparison_baselines.mesongs_adapter',
                                         ['decode','--repo',ROOT/'comparison_baselines/MesonGS','--archive',received,
                                          '--ply',restored,'--device',args.device],
                                         out/'logs'/f'{method}_{tag}_{trial:03d}_decode.log')
                        if quality is None:
                            images=0 if saved_images else args.image_views
                            evaluation_path=result_dir/'evaluation/metrics.json'
                            if evaluation_path.exists():
                                quality_reused=True
                                evaluation=_read(evaluation_path)
                                identity=evaluation['identity']
                                if (identity['source_ply']['sha256']!=plan['input']['sha256'] or
                                    identity['received_ply']['sha256']!=sha256_file(restored)):
                                    raise ValueError('incomplete trial evaluation belongs to different PLYs')
                            else:
                                _execute(args.eval_python,'comparison_baselines.evaluate',
                                         _eval_arguments(plan,restored,result_dir/'evaluation',images,camera_file=camera_file),
                                         out/'logs'/f'{method}_{tag}_{trial:03d}_evaluate.log')
                                evaluation=_read(evaluation_path)
                            quality=dict(identity=dict(payload_sha256=metadata['payload_sha256'],
                                                       input=plan['input'],evaluation=plan['evaluation']),
                                         metrics=evaluation['metrics'],
                                         evaluation_report=str(evaluation_path.relative_to(out)))
                            _json(quality_path,quality)
                            row['images_saved']=bool(images);saved_images|=bool(images)
                        row.update(metrics=quality['metrics'],evaluation_report=quality['evaluation_report'],
                                   deterministic_quality_reused=quality_reused)
                        if args.keep_intermediates:
                            import shutil
                            retained=out/'received'/method/f'snr_{tag}'/f'trial_{trial:03d}'
                            retained.mkdir(parents=True,exist_ok=True)
                            shutil.copy2(received,retained/received.name)
                            shutil.copy2(restored,retained/restored.name)
                    _json(result_file,row)
                rows.append(row);_save_results(out,plan,rows,'running')
                print(f'{method} SNR={snr:g} trial={trial}: {row["status"]}; K={row["total_channel_uses"]:,}',flush=True)
    _save_results(out,plan,rows,'complete')
    (out/'error.json').unlink(missing_ok=True)
    _json(out/'complete.json',dict(status='complete',trials=len(rows),successful=sum(r['status']=='success' for r in rows)))
    _review_archive(out)
    print(f'Review: {out/"review.tar.gz"}',flush=True)


def main():
    args=parser().parse_args();plan=build_plan(args)
    if args.dry_run:print(json.dumps(plan,indent=2,ensure_ascii=False));return
    try:execute(args,plan)
    except Exception as exc:
        out=Path(args.out).resolve()
        if (out/'plan.json').is_file() and _read(out/'plan.json')==plan:
            _json(out/'error.json',dict(status='error',type=type(exc).__name__,message=str(exc)))
        raise


if __name__=='__main__':main()
