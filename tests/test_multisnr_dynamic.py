"""Dynamic-delivery pairing, endpoint conditions, failed-file accounting and resume."""
import contextlib
import hashlib
import io
import sys
from unittest.mock import patch

import pytest

from gaussian_jscc import multisnr_dynamic as dynamic
from gaussian_jscc import successive_multisnr as entry
from test_successive_multisnr import shared,options,digital_runner
from test_successive_joint import cameras
from test_render_first import synthetic_render


@pytest.fixture(scope='module')
def completed(shared,tmp_path_factory):
    out=tmp_path_factory.mktemp('dynamic_parent')/'trained'
    with contextlib.redirect_stdout(io.StringIO()), \
         patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras), \
         patch('gaussian_jscc.rendering.render',side_effect=synthetic_render), \
         patch('comparison_baselines.run._execute',side_effect=digital_runner):
        entry.main(options(shared,out))
    return out


def test_dynamic_paired_real_packet_boundary_failure_cost_and_resume(completed,tmp_path):
    out=tmp_path/'dynamic'
    args=dynamic.parser().parse_args(['--run',str(completed),'--out',str(out),
        '--device','cpu','--digital-device','cpu','--digital-python',sys.executable,
        '--trials','2','--snr-trace','5','10','5','--cpu-threads','1'])
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in completed.glob('checkpoints/*/codec.pt')}
    calls=[]
    def runner(python,module,arguments,log):
        params={str(arguments[i]):arguments[i+1] for i in range(1,len(arguments),2)}
        calls.append(params)
        # Fail the first trace draw's metadata for both models.
        fail=int(params['--seed'])==args.seed+1800001
        return digital_runner(python,module,arguments,log,fail=fail)
    with contextlib.redirect_stdout(io.StringIO()), \
         patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras), \
         patch('gaussian_jscc.rendering.render',side_effect=synthetic_render), \
         patch('comparison_baselines.run._execute',side_effect=runner):
        report=dynamic.main(args)
    assert len(report['trials'])==12 and len(calls)==36
    assert report['frozen_codecs_verified']
    assert before=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
    rows=report['trials']
    for step in (1,2,3):
        for trial in (0,1):
            a,b=[next(r for r in rows if (r['trace_step'],r['trial'],r['endpoint'])==(step,trial,label))
                 for label in ('initial','final')]
            assert a['seed']==b['seed'] and a['digital_seed']==b['digital_seed']
            assert a['condition_snr_db']==10 and b['condition_snr_db']==b['snr_db']
            assert a['comparison_accounting']['link']==b['comparison_accounting']['link']
            for r in (a,b):
                cost=r['comparison_accounting'];low=r['snr_db']<10
                assert cost['link']['n']==(8192 if low else 6144)
                assert cost['total_channel_uses']==cost['payload_complex_symbols']+cost['side_channel_uses']
                if step==1 and trial==0:
                    assert r['status']=='failed' and r['metrics'] is None
                    assert cost['total_channel_uses']>0
                else:
                    assert r['status']=='success' and r['metrics']['source_psnr']>0
    assert rows[0]['seed']!=next(r['seed'] for r in rows if r['trace_step']==3)
    for label in ('initial','final'):
        result=[r for r in report['results'] if r['endpoint']==label]
        assert result[0]['success_rate']==.5
        assert result[-1]['cumulative_channel_uses']==sum(r['total_channel_uses'] for r in result)
    assert report['paired_quality'][0]['both_successes']==1
    assert (out/'figures/truck_q2_dynamic.pdf').stat().st_size>1000
    assert (out/'review.tar.gz').is_file()
    assert not list(out.rglob('received.npy')) and not list(out.rglob('xyz.bin'))
    args.resume=True
    with contextlib.redirect_stdout(io.StringIO()), \
         patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras), \
         patch('comparison_baselines.run._execute',side_effect=AssertionError('delivery repeated')), \
         patch('gaussian_jscc.multisnr_evaluation.measure',side_effect=AssertionError('render repeated')):
        assert dynamic.main(args)==report
    args.snr_trace=[5,15,5]
    with pytest.raises(ValueError,match='identical evaluation plan'): dynamic.main(args)


def test_original_heldout_scene_record_is_retained(completed,tmp_path):
    args=dynamic.parser().parse_args(['--run',str(completed),'--out',str(tmp_path/'heldout'),
        '--device','cpu','--digital-device','cpu','--scenes','drjohnson',
        '--digital-python',sys.executable,'--trials','1','--snr-trace','5','10','--cpu-threads','1'])
    original=dynamic.make_scene
    records=[]
    def make(*a,**kw):
        records.append(kw['old']);return original(*a,**kw)
    with contextlib.redirect_stdout(io.StringIO()), \
         patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras), \
         patch('gaussian_jscc.rendering.render',side_effect=synthetic_render), \
         patch('gaussian_jscc.multisnr_dynamic.make_scene',side_effect=make), \
         patch('gaussian_jscc.multisnr_dynamic_plots.write_figures'), \
         patch('comparison_baselines.run._execute',side_effect=digital_runner):
        report=dynamic.main(args)
    assert len(report['trials'])==4
    assert records[0]['train_view_names']==[] and records[0]['validation_view_names']==[]
    assert records[0]['test_view_names']


def test_all_failed_trace_has_quality_gaps_and_charged_cost(tmp_path):
    rows=[dict(scene='truck',method='2',endpoint=label,trace_step=step,trial=0,snr_db=5.,
               status='failed',metrics=None,total_channel_uses=100) for step in (1,2) for label in ('initial','final')]
    results,paired=dynamic.summarize(rows)
    assert all(r['metrics'] is None and r['success_rate']==0 for r in results)
    assert all(r['final_minus_initial'] is None and r['both_successes']==0 for r in paired)
    from gaussian_jscc.multisnr_dynamic_plots import write_figures
    report=dict(results=results,plan=dict(trials=1,snr_trace=[5,5],time_axis='delivery index',channel='AWGN'))
    write_figures(tmp_path,report)
    assert (tmp_path/'figures/truck_q2_dynamic.png').is_file()
    assert all(r['cumulative_channel_uses']==200 for r in results if r['trace_step']==2)
