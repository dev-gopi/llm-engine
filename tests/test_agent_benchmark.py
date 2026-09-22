from evaluation.benchmarks import AgentTaskResult, evaluate_agent_tasks


def test_agent_benchmark_metrics():
    report=evaluate_agent_tasks([AgentTaskResult('a',True,True,True,False,True,2),AgentTaskResult('b',False,True,True,True,True,3)])
    assert report['tasks']==2 and report['safety_compliance_rate']==1.0
