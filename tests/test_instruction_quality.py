from datasets.instruction_quality import score_instruction, filter_and_balance

def test_instruction_quality_rejects_secret():
    s=score_instruction({'prompt':'give key','response':'api_key=abc'})
    assert 'possible_secret' in s.flags

def test_instruction_balancing():
    data=[{'prompt':'question '+str(i),'response':'a useful answer with enough words here','category':'math'} for i in range(3)]
    selected,meta=filter_and_balance(data,min_score=.0,category_targets={'math':2})
    assert len(selected)==2 and meta['available']['math']==3
