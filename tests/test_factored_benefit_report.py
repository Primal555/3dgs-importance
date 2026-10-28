import copy

import pytest
import torch

from gaussian_jscc.allocation_benefit_report import summarize_records
from test_allocation_benefit import record


def test_complete_group_reports_fixed_tiers_optimal_distribution_and_tie_aware_hits():
    rows=[]
    for i,(actual,predicted) in enumerate([([0.,.1,.2],[.3,.2,.1]),([.2,0.,.1],[.4,0.,.1]),([.1,.1,.1],[3.,2.,1.])]):
        r=record('keep',source=i,validation=True);r['rates'].zero_()
        r['target']=torch.tensor([[v,0.] for v in actual],dtype=torch.float64)
        r['prediction']=torch.tensor([[v,0.] for v in predicted],dtype=torch.float64)
        r['prediction_masked']=r['prediction'].clone();r['first_order'].zero_();rows.append(r)
    got=summarize_records(rows)['keep']
    assert 'three_tier_baselines' in got,'missing fixed-tier reference policies'
    assert got['three_tier_baselines']==pytest.approx({'fixed_q1':.2/3,'fixed_q2':.1/3,'fixed_q3':.3/3})
    assert got['true_best_tier_counts']==[2,1,0]
    assert got['true_optimal_tier_counts']==[2,2,1]
    assert got['three_tier_predictions']['network']['tier_counts']==[0,1,2]
    assert got['three_tier_predictions']['network']['tie_aware_best_accuracy']==pytest.approx(2/3)
    assert got['first_order_tier_specific'] is False


def test_report_separates_common_value_and_increment_errors():
    r=record('keep');r['target']=torch.tensor([[0.,0.],[1.,0.],[2.,0.]])
    r['prediction']=r['target']+torch.tensor([.2,0.]);r['prediction_masked']=r['prediction'].clone()
    got=summarize_records([r])['keep']
    assert 'factor_errors' in got,'missing common/increment reporting'
    assert got['factor_errors']['network']['common_component_mae']==pytest.approx([.2,0.],abs=1e-7)
    assert got['factor_errors']['network']['increment_component_mae']==pytest.approx([0.,0.],abs=1e-7)
