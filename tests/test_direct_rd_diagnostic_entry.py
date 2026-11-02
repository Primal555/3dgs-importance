import contextlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test_structured_allocation_integration import source
from test_calibrated_spatial_benefit_integration import options as benefit_options
from test_spatial_benefit_integration import Window
from test_direct_rd_allocation import options as direct_options, FrozenWindow
from test_learned_benefit_allocation import cameras
import test_budgeted_allocation as fixture


def test_read_only_diagnostic_runs_from_embedded_frozen_checkpoint(source):
    assert importlib.util.find_spec('gaussian_jscc.direct_rd_diagnostic'), 'diagnostic entry missing'
    from gaussian_jscc import direct_rd_diagnostic as entry
    from gaussian_jscc import direct_rd_allocation as direct
    from gaussian_jscc import calibrated_spatial_benefit_allocation as benefit
    parent,root=source; learned=root/'learned'; run=root/'direct'
    def evaluate(model,scene,q,actions,cams,learner,reference,args,**kw):
        rows={a['source']:dict(old_tier=int(q[a['source']]),new_tier=a['new_tier'],
            predicted_image_delta=-3.,true_image_delta=.001,predicted_rd_proxy_delta=-3.,rate_proxy_delta=0.,
            true_mse_component_delta=.001,true_structure_component_delta=0.) for a in actions}
        return rows,SimpleNamespace(static=None,records=rows)
    with patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras) as camera_loader, \
         patch('gaussian_jscc.rendering.render',side_effect=fixture.render), \
         patch.object(benefit.spatial,'SpatialWindow',Window),patch.object(benefit.spatial,'preflight',return_value={}), \
         patch.object(direct,'BatchedSpatialWindow',FrozenWindow),patch.object(direct,'batched_preflight',return_value={}), \
         patch.object(entry,'BatchedSpatialWindow',FrozenWindow),patch.object(entry,'batched_preflight',return_value={}), \
         patch('gaussian_jscc.direct_rd_diagnostic_inputs.selected_train_cameras',
               side_effect=lambda source,names,*args:[c for c in cameras(source,2,False,'images','train') if c.image_name in names]), \
         patch.object(entry,'evaluate_actions',side_effect=evaluate), \
         patch.object(entry,'evaluate_other_views',side_effect=lambda window,cams:window.records), \
         contextlib.redirect_stdout(io.StringIO()):
        benefit.main(benefit_options(parent,learned)); direct.main(direct_options(direct,parent,learned,run))
        before={str(p.relative_to(run)):(p.stat().st_size,p.stat().st_mtime_ns) for p in run.rglob('*') if p.is_file()}
        args=entry.parser().parse_args(['--run',str(run),'--out',str(root/'diagnostic'),
            '--device','cpu','--proposal-windows','2','--per-stratum','1','--sanity-actions','0'])
        camera_loader.reset_mock()
        args.dry_run=True; entry.main(args)
        assert not Path(args.out).exists()
        args.dry_run=False; result=entry.main(args)
        assert camera_loader.call_count==0
        assert result['status']=='complete' and result['source_unchanged']
        assert (Path(args.out)/'summary.json').exists() and (Path(args.out)/'actions.csv').exists()
        assert (Path(args.out)/'review.tar.gz').exists() and not list(Path(args.out).rglob('*.pt'))
        import json
        records=json.loads((Path(args.out)/'actions.json').read_text())
        assert records[0]['condition_samples']['calibration_accept'][0]['trial']==0
        assert 'seed' in records[0]['condition_samples']['matched'][0]
        after={str(p.relative_to(run)):(p.stat().st_size,p.stat().st_mtime_ns) for p in run.rglob('*') if p.is_file()}
        assert before==after
