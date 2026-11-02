"""Condition-separated statistics for selected-action diagnostics."""
import statistics


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def summarize(records,groups,sanity,*,tie=1e-8,requested_per_stratum=16):
    strata = {}
    decomposition = []
    for row in records:
        a,b,c = (row[k] for k in ('matched','calibration_same_noise','calibration_accept'))
        parts = dict(image_prediction_error=a['true_image_delta']-a['predicted_image_delta'],
            rate_proxy_error=row['measured_rate_delta']-a['rate_proxy_delta'],
            view_shift=b['true_rd_delta']-a['true_rd_delta'],
            noise_shift=c['true_rd_delta']-b['true_rd_delta'],
            selection_replay_error=a['predicted_rd_proxy_delta']+row['selection_predicted_gain'])
        row['decomposition'] = parts; decomposition.append(parts)
    for name in ('top','middle','bottom','ordinary'):
        rows = [r for r in records if r['stratum']==name]
        if not rows: continue
        result = dict(count=len(rows),transitions={})
        for condition in ('matched','calibration_same_noise','calibration_accept'):
            true = [r[condition]['true_rd_delta'] for r in rows]
            pred = [r[condition].get('predicted_rd_proxy_delta',-r['selection_predicted_gain']) for r in rows]
            informative = [(p,t) for p,t in zip(pred,true) if abs(t)>tie]
            result[condition] = dict(predicted_delta_mean=mean(pred),true_delta_mean=mean(true),
                informative_count=len(informative),
                sign_error_fraction=mean((p<0)!=(t<0) for p,t in informative),
                harmful_fraction=mean(t>tie for t in true),
                regret_mean=mean(abs(t) if (p<0)!=(t<0) else 0. for p,t in zip(pred,true)),
                rd_mae=mean(abs(p-t) for p,t in zip(pred,true)))
        for row in rows:
            key=f"q{row['old_tier']}->q{row['new_tier']}"
            result['transitions'][key]=result['transitions'].get(key,0)+1
        result['decomposition'] = {key:mean(r['decomposition'][key] for r in rows) for key in decomposition[0]}
        strata[name] = result
    passed = bool(sanity) and all(r['passed'] for r in sanity)
    scores = {k:mean(abs(r[k]) for r in decomposition) for k in decomposition[0]} if decomposition else {}
    focuses = [
        ('selected_action_prediction','image_prediction_error','修正被选高分动作的收益估计与监督分布'),
        ('marginal_rate','rate_proxy_error','修正候选边际通信成本'),
        ('view_alignment','view_shift','对齐候选评价与校准验收的视角任务'),
        ('noise_averaging','noise_shift','增加配对噪声平均并检查高分动作的噪声稳定性'),
        ('selection_replay','selection_replay_error','先核查候选重放和预测累计是否一致')]
    ranked = [dict(code=code,mean_absolute_gap=scores.get(key,0.),action=text) for code,key,text in focuses]
    interaction = mean(abs(g['true_rd_delta']-g['summed_single_rd_delta'])/len(g['sources'])
                       for g in groups if 'summed_single_rd_delta' in g and g['sources']) or 0.
    ranked.append(dict(code='joint_interactions',mean_absolute_gap=interaction,
                       action='拆分相互干扰的动作并调整整批提案与验收方式'))
    ranked.sort(key=lambda r:r['mean_absolute_gap'],reverse=True)
    coverage={}
    for name in ('top','middle','bottom','ordinary'):
        actual=strata.get(name,{}).get('count',0)
        informative={condition:strata.get(name,{}).get(condition,{}).get('informative_count',0)
                     for condition in ('matched','calibration_accept')}
        coverage[name]=dict(requested=requested_per_stratum,actual=actual,informative=informative,
                            sufficient=actual>=8 and min(informative.values())>=8)
    sufficient=all(row['sufficient'] for row in coverage.values())
    evidence=('adequate' if all(row['actual']>=requested_per_stratum for row in coverage.values())
              else 'limited') if sufficient else 'insufficient'
    if not sufficient:
        ranked=[dict(code='insufficient_evidence',action='部分层缺少有效样本，先结合原始动作明细解释，不作机制主因排名')]
    if not passed:
        ranked = [dict(code='execution_consistency',action='先修复执行/局部标签一致性，再解释机制误差')]
    compact_groups=[dict({k:v for k,v in g.items() if k not in ('sources','tiers')},
                         changes=len(g.get('sources',()))) for g in groups]
    return dict(strata=strata,execution_consistency_passed=passed,sanity=sanity,groups=compact_groups,
        mean_absolute_decomposition=scores,recommended_focus=ranked,sample_coverage=coverage,evidence_status=evidence,
        scope='stratified selected-action sample, not a population error rate; ranking is evidence for manual review')
