from evaluation.calibration import calibration_metrics


def test_calibration_metrics():
    r=calibration_metrics([1.,0.,1.,0.],[1,0,0,0],bins=2)
    assert 0<=r['ece']<=1 and 0<=r['brier']<=1
