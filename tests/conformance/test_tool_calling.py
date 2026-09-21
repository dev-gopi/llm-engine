import pytest
from pydantic import ValidationError

from serving.schemas import OpenAIChatCompletionRequest


def tool(name="echo"):
    return {"type":"function","function":{"name":name,"description":"Echo","parameters":{"type":"object","properties":{"value":{"type":"string"}},"required":["value"],"additionalProperties":False}}}


def base(**kwargs):
    value={"model":"gopi-test","messages":[{"role":"user","content":"call echo"}],"tools":[tool()],"tool_choice":"auto"}
    value.update(kwargs)
    return OpenAIChatCompletionRequest.model_validate(value)


def test_tool_choice_none_auto_required_and_specific():
    assert base(tool_choice="none").tool_choice == "none"
    assert base(tool_choice="auto").tool_choice == "auto"
    assert base(tool_choice="required").tool_choice == "required"
    assert base(tool_choice={"type":"function","function":{"name":"echo"}}).tool_choice.function.name == "echo"


def test_specific_tool_choice_unknown_and_required_without_tools_are_rejected():
    with pytest.raises(ValidationError):
        base(tool_choice={"type":"function","function":{"name":"missing"}})
    with pytest.raises(ValidationError):
        OpenAIChatCompletionRequest.model_validate({"model":"gopi-test","messages":[{"role":"user","content":"x"}],"tool_choice":"required"})


def test_duplicate_tools_and_malformed_arguments_are_rejected_before_execution():
    with pytest.raises(ValidationError):
        base(tools=[tool(), tool()])
    with pytest.raises(ValidationError):
        base(tools=[{"type":"function","function":{"name":"bad name","parameters":{}}}])
