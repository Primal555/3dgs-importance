"""Match and optionally pause this user's exact direct-allocation source run."""
import argparse
import os
from pathlib import Path
import signal
import time


def running_workers(run,*,proc_root=Path('/proc'),uid=None):
    run,proc_root=Path(run).resolve(),Path(proc_root)
    if not proc_root.is_dir(): return []
    uid=os.getuid() if uid is None else uid
    result=[]
    for folder in proc_root.iterdir():
        if not folder.name.isdecimal(): continue
        try:
            if folder.stat().st_uid!=uid: continue
            args=folder.joinpath('cmdline').read_bytes().decode().split('\0')
            if '-m' not in args or args[args.index('-m')+1]!='gaussian_jscc.direct_rd_allocation': continue
            if '--out' in args: target=args[args.index('--out')+1]
            else: target=next(a[6:] for a in args if a.startswith('--out='))
            path=Path(target)
            if not path.is_absolute(): path=folder.joinpath('cwd').resolve()/path
            if path.resolve()==run: result.append(int(folder.name))
        except (OSError,UnicodeError,ValueError,IndexError,StopIteration): continue
    return sorted(result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run'); parser.add_argument('--pause',action='store_true')
    args=parser.parse_args(); workers=running_workers(args.run)
    if workers and not args.pause:
        raise SystemExit(f'Source still running: {workers}; set PAUSE_SOURCE=1 to pause only this run')
    for pid in workers:
        if pid in running_workers(args.run):
            try: os.kill(pid,signal.SIGTERM)
            except ProcessLookupError: pass
    deadline=time.monotonic()+20
    while running_workers(args.run):
        if time.monotonic()>deadline: raise SystemExit('Source has not exited; diagnostic not started')
        time.sleep(.2)
    print(f'Source stopped; paused PIDs: {workers}. Reading the latest saved checkpoint.',flush=True)


if __name__=='__main__': main()
