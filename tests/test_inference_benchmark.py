from evaluation.inference_benchmark import summarize_latency

def test_inference_metrics():
    r=summarize_latency([{'ttft_s':.1,'itl_s':.02,'tokens':10,'duration_s':1,'peak_memory_mb':100},{'ttft_s':.2,'itl_s':.03,'tokens':20,'duration_s':2,'peak_memory_mb':120}])
    assert r['requests']==2 and r['tokens_per_second']==10
