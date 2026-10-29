import copy
import torch


def test_checkpoint_score_accounts_for_different_best_tier_with_equal_numeric_error():
    from gaussian_jscc.response_value_report import summarize_response,value_selection_score
    def record(prediction):
        return dict(branch='keep',source=3,step=1,old=3,new=[1,2,3],
            target=torch.tensor([[-3.,0.],[-2.,0.],[-1.,0.]]),
            prediction=torch.tensor([[prediction[0],0.],[prediction[1],0.],[prediction[2],0.]]),
            rates=torch.zeros(3),conditional=torch.ones(3)/3)
    # Both have component squared errors [4,4,0] and correct retain direction.
    good=summarize_response([record([-5.,0.,-1.])])
    bad=summarize_response([record([-1.,-4.,-1.])])
    assert good['keep']['stages']['network']['component_rmse']==bad['keep']['stages']['network']['component_rmse']
    assert good['keep']['stages']['network']['regret_mean']==bad['keep']['stages']['network']['regret_mean']==0
    # A positive-origin sign metric is deliberately held equal.
    positive=dict(count=1,stages=dict(network=dict(regret_mean=0.)))
    good['positive']=copy.deepcopy(positive);bad['positive']=copy.deepcopy(positive)
    assert value_selection_score(good,1)<value_selection_score(bad,1)
