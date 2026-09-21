import pytest
from serving.schemas import OpenAIChatCompletionRequest

def test_supported_generation_parameters_are_retained():
    r=OpenAIChatCompletionRequest(model="g",messages=[{"role":"user","content":"x"}],temperature=0.2,top_p=0.8,top_k=7,min_p=0.1,max_tokens=11,seed=4,stop=["END"],response_format={"type":"text"})
    g=r.generation_request("g")
    assert (g.temperature,g.top_p,g.top_k,g.min_p,g.max_tokens,g.seed,g.stop)==(0.2,0.8,7,0.1,11,4,["END"])

def test_invalid_response_format_is_rejected():
    with pytest.raises(ValueError): OpenAIChatCompletionRequest(model="g",messages=[{"role":"user","content":"x"}],response_format={"type":"xml"})

def test_unsupported_penalties_are_rejected_instead_of_ignored():
    with pytest.raises(ValueError, match="presence_penalty"):
        OpenAIChatCompletionRequest(
            model="g", messages=[{"role":"user", "content":"x"}], presence_penalty=0.0
        )
    with pytest.raises(ValueError, match="frequency_penalty"):
        OpenAIChatCompletionRequest(
            model="g", messages=[{"role":"user", "content":"x"}], frequency_penalty=0.2
        )
