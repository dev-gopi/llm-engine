import pytest
from inference.agent_runtime import AgentStep, BoundedAgentRuntime, AgentState

def test_agent_requires_approval_and_is_bounded():
    runtime=BoundedAgentRuntime({'add':lambda x:x+1},max_steps=2,require_approval=True)
    runtime.plan([AgentStep('add',{'x':1})])
    assert runtime.state==AgentState.WAITING_APPROVAL
    with pytest.raises(RuntimeError): runtime.run()
    runtime.approve(); runtime.run(); assert runtime.state==AgentState.COMPLETED; assert runtime.steps[0].result==2

def test_agent_rejects_unapproved_tool():
    runtime=BoundedAgentRuntime({'add':lambda x:x+1},allowed_tools={'add'})
    with pytest.raises(ValueError): runtime.plan([AgentStep('delete')])
