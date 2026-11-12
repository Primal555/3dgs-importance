import contextlib
import io
import json
from unittest.mock import patch
import numpy as np
import pytest
import torch
from gaussian_jscc import fast_first_order_allocation as teacher
from gaussian_jscc import first_order_positive_shuffle as control
from gaussian_jscc.modular_allocation_shuffle import shuffle_retained
from test_structured_allocation_integration import source
from test_fast_first_order_entry import args_for
import test_budgeted_allocation as fixture


def test_permutation_preserves_mask_counts_and_seed():
    q=torch.arange(1000)%4
    a=shuffle_retained(q,42);b=shuffle_retained(q,42)
    assert torch.equal(a,b) and torch.equal(a==0,q==0)
    assert torch.equal(torch.bincount(a),torch.bincount(q))
    assert not torch.equal(a,q)


def test_digital_teacher_resume_then_paired_shuffle(source):
    parent,root=source
    with patch('gaussian_jscc.rendering.load_cameras',side_effect=fixture.cameras), \
         patch('gaussian_jscc.rendering.render',side_effect=fixture.render), contextlib.redirect_stdout(io.StringIO()):
        out=root/'teacher64_fixture';args=args_for(parent,out)
        args.cost_protocol='digital_side_file_v1';args.tier_max_log_odds=2.
        assert teacher.main(args,max_updates=1)['status']=='interrupted'
        args.resume=True;done=teacher.main(args)
        assert done['status']=='complete' and done['cost_protocol']=='digital_side_file_v1'
        review=json.loads((out/'review_summary.json').read_text())
        for row in review['rows'].values():
            assert row['total_channel_uses']==row['comparison_accounting']['total_channel_uses']
            assert row['rd_score']==pytest.approx(row['source_distortion']/row['d_ref']+row['beta']*row['total_channel_uses']/row['c_ref'])
        source_bytes=(out/'deployment/selected_tiers.npy').read_bytes()
        options=control.parser().parse_args(['--allocation',str(out),'--out',str(out/'positive_shuffle'),'--device','cpu'])
        result=control.main(options)
        assert result['optimizer_updates']==0 and result['comparison']['fixed_q0_mask']
        assert (out/'deployment/selected_tiers.npy').read_bytes()==source_bytes
        r=json.loads((out/'positive_shuffle/review_summary.json').read_text())
        a,b=r['rows']['selected'],r['rows']['shuffled_positive']
        assert a['payload_complex_symbols']==b['payload_complex_symbols']
        assert a['xyz_bytes']==b['xyz_bytes'] and a['tier_counts']==b['tier_counts']
        assert a['source_psnr']==review['rows']['selected']['source_psnr']
        assert a['total_channel_uses']==review['rows']['selected']['total_channel_uses']
        assert result['comparison']['rd_difference']==pytest.approx(a['rd_score']-b['rd_score'])
        q=np.load(out/'deployment/selected_tiers.npy');s=np.load(out/'positive_shuffle/shuffled_tiers.npy')
        assert np.array_equal(q==0,s==0)
        assert control.main(options)==result
        options.shuffle_seed+=1
        with pytest.raises(ValueError,match='settings changed'):control.main(options)
        assert (out/'positive_shuffle/review.tar.gz').is_file()
