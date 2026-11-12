import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import pytest
from comparison_baselines.digital_link import LinkConfig, file_channel_cost
from gaussian_jscc import digital_side_delivery as entry
from test_structured_allocation_integration import source, options
import test_budgeted_allocation as fixture


def test_real_ldpc_side_files_when_sionna_is_available(tmp_path):
    pytest.importorskip('sionna.phy.fec.ldpc')
    from gaussian_jscc.successive_delivery_packet import SIDE_FILES
    from comparison_baselines.digital_link import transmit_file
    sender = tmp_path/'sender'; sender.mkdir()
    for name in SIDE_FILES: (sender/name).write_bytes(bytes(range(111))+name.encode())
    config = LinkConfig(k=512,n=768,device='cpu',batch_size=2)
    def runner(src,dest,report,seed):
        return transmit_file(src,dest,config=config,snr_db=30.,seed=seed)
    report = entry.transmit_sides(sender,tmp_path/'receiver',tmp_path/'reports',config,30.,42,runner)
    assert report['status']=='success'
    assert report['side_channel_uses']==sum(file_channel_cost((sender/n).stat().st_size,config)['total_channel_uses'] for n in SIDE_FILES)


@pytest.mark.parametrize('failed', [False, True])
def test_completed_map_actual_received_files_and_failed_delivery_reporting(source, failed):
    from gaussian_jscc import structured_evidence_allocation as teacher
    parent, root = source
    with patch('gaussian_jscc.rendering.load_cameras',side_effect=fixture.cameras), \
         patch('gaussian_jscc.rendering.render',side_effect=fixture.render), contextlib.redirect_stdout(io.StringIO()):
        allocation = root/'allocation'; teacher.main(options(parent,allocation))
        calls = []
        def execute(python,module,arguments,log):
            assert module=='comparison_baselines.digital_link'
            params = {str(arguments[i]):arguments[i+1] for i in range(1,len(arguments),2)}
            src,dest,report = [Path(params[k]) for k in ('--source','--destination','--report')]
            cfg = LinkConfig(**{k:(str(params['--'+k.replace('_','-')]) if k=='device' else int(params['--'+k.replace('_','-')])) for k in LinkConfig.__dataclass_fields__})
            lost = failed and src.name=='metadata.bin'
            if not lost: shutil.copyfile(src,dest)
            result = dict(status='failed' if lost else 'success',config=vars(cfg),
                source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),snr_db=float(params['--snr-db']),
                seed=int(params['--seed']),noiseless=False,**file_channel_cost(src.stat().st_size,cfg))
            report.write_text(json.dumps(result));calls.append(src.name)
        out = root/'delivery'
        args = entry.parser().parse_args(['--allocation',str(allocation),'--out',str(out),
            '--digital-python','external-digital-python','--device','cpu','--digital-device','cpu'])
        with patch.object(entry,'_execute',side_effect=execute):
            report = entry.main(args)
        assert report['status']=='complete' and len(report['trials'])==2
        assert calls==['xyz.bin','metadata.bin','scales.bin']*2
        for row in report['trials']:
            assert row['status']==('failed' if failed else 'success')
            assert (row['metrics'] is None)==failed
            assert row['comparison_accounting']['side_link_simulated']
            assert row['total_channel_uses']==row['payload_complex_symbols']+row['comparison_accounting']['side_channel_uses']
        assert all(r['success_rate']==(0 if failed else 1) for r in report['results'])
        assert (out/'review.tar.gz').is_file() and (out/'execution_check.json').is_file()
        assert not list(out.rglob('received.npy'))
        assert not list(out.rglob('xyz.bin'))
