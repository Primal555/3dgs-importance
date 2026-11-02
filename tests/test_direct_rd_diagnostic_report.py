import importlib.util


def api():
    assert importlib.util.find_spec('gaussian_jscc.direct_rd_diagnostic_report'), 'diagnostic report missing'
    from gaussian_jscc.direct_rd_diagnostic_report import summarize
    return summarize


def test_report_separates_prediction_view_noise_and_interactions(tmp_path):
    summarize = api()
    records = []
    for i in range(64):
        stratum=('top','middle','bottom','ordinary')[i//16]
        records.append(dict(source=i,stratum=stratum,old_tier=2,new_tier=0,
            selection_predicted_gain=.02, measured_rate_delta=-.001,
            matched=dict(predicted_image_delta=-.019,rate_proxy_delta=-.001,
                         predicted_rd_proxy_delta=-.02,true_image_delta=.004,true_rd_delta=.003),
            calibration_same_noise=dict(true_rd_delta=.005,true_image_delta=.006),
            calibration_accept=dict(true_rd_delta=.007,true_image_delta=.008)))
    groups = [dict(name='top_sample',sources=[0],true_rd_delta=.007,
                   summed_single_rd_delta=.007,image_interaction_delta=0.,rate_nonadditivity_delta=0.)]
    report = summarize(records,groups,[dict(abs_error=1e-9,tolerance=1e-6,passed=True)],tie=1e-8)
    assert report['strata']['top']['matched']['sign_error_fraction'] == 1.
    errors = report['strata']['top']['decomposition']
    assert abs(errors['image_prediction_error']-.023) < 1e-10
    assert abs(errors['view_shift']-.002) < 1e-10
    assert abs(errors['noise_shift']-.002) < 1e-10
    assert abs(sum(errors.values())-(.007-(-.02))) < 1e-10
    assert report['execution_consistency_passed']
    assert report['recommended_focus'][0]['code'] == 'selected_action_prediction'
    broken = summarize(records,groups,[dict(abs_error=.1,tolerance=1e-6,passed=False)])
    assert broken['recommended_focus'][0]['code'] == 'execution_consistency'


def test_insufficient_and_tied_evidence_cannot_rank_a_mechanism():
    summarize=api()
    row=dict(source=0,stratum='top',old_tier=2,new_tier=0,selection_predicted_gain=.01,measured_rate_delta=0.,
        matched=dict(predicted_image_delta=-.01,rate_proxy_delta=0.,predicted_rd_proxy_delta=-.01,true_image_delta=0.,true_rd_delta=0.),
        calibration_same_noise=dict(true_rd_delta=0.,true_image_delta=0.),
        calibration_accept=dict(true_rd_delta=0.,true_image_delta=0.))
    report=summarize([row],[],[dict(passed=True)])
    assert report['evidence_status']=='insufficient'
    assert report['recommended_focus'][0]['code']=='insufficient_evidence'
    assert report['strata']['top']['matched']['sign_error_fraction'] is None
    assert report['sample_coverage']['middle']['actual']==0
    assert not report['sample_coverage']['top']['sufficient']
