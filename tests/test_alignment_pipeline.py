from alignment.pipeline import audit_preference, audit_preference_set, compare_alignment_methods

def test_preference_audit():
    a=audit_preference({'prompt':'p','chosen':'good answer here','rejected':'bad answer here'})
    assert a.valid
    assert audit_preference({'prompt':'p','chosen':'x','rejected':'x'}).valid is False

def test_alignment_comparison_preserves_raw_deltas():
    r=compare_alignment_methods({'sft':{'capability':.8,'safety':.9,'regression':0},'dpo':{'capability':.82,'safety':.88,'regression':.02}})
    assert abs(r['deltas']['dpo']['capability']-.02)<1e-12
