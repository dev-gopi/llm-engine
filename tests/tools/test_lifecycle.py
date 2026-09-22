import asyncio

import pytest

from security.authorization import ToolAuthorization
from tools.executor import ControlledToolExecutor


@pytest.mark.asyncio
async def test_controlled_tool_executor_enforces_schema_approval_timeout_cancellation_size_and_audit():
    events=[]
    async def echo(value): return {"value": value}
    async def slow(value):
        await asyncio.sleep(1)
        return value
    executor=ControlledToolExecutor(
        {"echo": echo, "slow": slow},
        authorization=ToolAuthorization(frozenset({"echo","slow"}), require_approval=True, timeout_seconds=0.02),
        schemas={"echo":{"type":"object","properties":{"value":{"type":"string"}},"required":["value"],"additionalProperties":False}},
        max_result_bytes=128,
        audit=events.append,
    )
    with pytest.raises(PermissionError): await executor.execute("echo", {"value":"x"})
    with pytest.raises(ValueError): await executor.execute("echo", {"value":3}, approved=True)
    assert await executor.execute("echo", {"value":"x"}, approved=True) == {"value":"x"}
    with pytest.raises(TimeoutError): await executor.execute("slow", {"value":"x"}, approved=True)
    cancel=asyncio.Event(); task=asyncio.create_task(executor.execute("slow", {"value":"x"}, approved=True, cancellation=cancel)); await asyncio.sleep(0); cancel.set()
    with pytest.raises(asyncio.CancelledError): await task
    assert any(event["event"] == "tool_completed" for event in events)
