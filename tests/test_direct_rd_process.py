import importlib.util
from pathlib import Path


def test_process_matching_requires_same_uid_module_and_absolute_output(tmp_path):
    assert importlib.util.find_spec('gaussian_jscc.direct_rd_process'), 'process guard missing'
    from gaussian_jscc.direct_rd_process import running_workers
    run = tmp_path/'source'; run.mkdir()
    def create(pid,args):
        folder=tmp_path/str(pid); folder.mkdir()
        (folder/'cmdline').write_bytes(b'\0'.join(a.encode() for a in args)+b'\0')
    base=['python','-m','gaussian_jscc.direct_rd_allocation','--out']
    create(101,base+[str(run)])
    create(102,base+[str(tmp_path/'other')])
    create(103,['python','-m','other.module','--out',str(run)])
    create(104,['python','-m','gaussian_jscc.direct_rd_allocation','--out='+str(run)])
    assert running_workers(run,proc_root=tmp_path,uid=Path(tmp_path/'101').stat().st_uid)==[101,104]
    assert running_workers(run,proc_root=tmp_path,uid=-1)==[]
