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


def test_native_tool_protocol_parses_and_validates_arguments():
    from serving.chat_protocol import parse_tool_calls

    request = base()
    parsed = parse_tool_calls(
        '<tool_call>{"name":"echo","arguments":{"value":"hello"}}</tool_call>',
        request.tools or [],
        request.tool_choice,
    )
    assert parsed.error is None
    assert parsed.content == ""
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].function.name == "echo"
    assert parsed.tool_calls[0].function.arguments == '{"value":"hello"}'


def test_native_tool_protocol_rejects_schema_invalid_arguments():
    from serving.chat_protocol import parse_tool_calls

    request = base(tool_choice="required")
    parsed = parse_tool_calls(
        '{"name":"echo","arguments":{"value":123}}',
        request.tools or [],
        request.tool_choice,
    )
    assert parsed.tool_calls == ()
    assert parsed.error is not None
    assert "do not satisfy" in parsed.error


def test_forced_single_tool_accepts_arguments_object_only():
    from serving.chat_protocol import parse_tool_calls

    request = base(tool_choice={"type":"function","function":{"name":"echo"}})
    parsed = parse_tool_calls(
        '{"value":"forced"}',
        request.tools or [],
        request.tool_choice,
    )
    assert parsed.error is None
    assert parsed.tool_calls[0].function.name == "echo"
    assert parsed.tool_calls[0].function.arguments == '{"value":"forced"}'


def test_auto_tool_choice_does_not_misclassify_plain_json_answer_as_tool_call():
    from serving.chat_protocol import parse_tool_calls

    request = base(tool_choice="auto")
    parsed = parse_tool_calls(
        '{"answer":"ordinary structured response"}',
        request.tools or [],
        request.tool_choice,
    )
    assert parsed.error is None
    assert parsed.tool_calls == ()
    assert parsed.content == '{"answer":"ordinary structured response"}'
