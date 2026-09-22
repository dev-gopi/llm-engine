from training.promotion import CheckpointCandidate, CheckpointPromoter, MetricRule


def test_promotion_is_multi_axis_and_protects_regressions():
    rules=[MetricRule('loss','min',1,0.02,True),MetricRule('instruction','max',2,0.05,True),MetricRule('math','max',1,0.10,True)]
    p=CheckpointPromoter(rules,baseline={'loss':1.0,'instruction':.8,'math':.7})
    d=p.decide([CheckpointCandidate('a',{'loss':.8,'instruction':.81,'math':.5}),CheckpointCandidate('b',{'loss':.9,'instruction':.84,'math':.72})])
    assert d.promoted and d.selected=='b'

def test_no_candidates_block():
    assert not CheckpointPromoter([MetricRule('loss','min')]).decide([]).promoted
