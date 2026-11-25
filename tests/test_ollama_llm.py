from echo_sim.llm_policy import WorkOut
from echo_sim.ollama_llm import OllamaLLM


def fake_post(reply: dict, sent: dict):
    def post(url, payload):
        sent.update(url=url, payload=payload)
        return reply

    return post


def test_request_shape_and_parse():
    sent = {}
    reply = {"message": {"content": '{"work": true, "rationale": "ok"}'}, "done_reason": "stop",
             "prompt_eval_count": 120, "eval_count": 15}
    llm = OllamaLLM(model="m", url="http://host:11434/", post=fake_post(reply, sent))
    res = llm("sys", "usr", WorkOut)
    assert res.parsed == WorkOut(work=True, rationale="ok") and res.stop_reason == "end_turn"
    assert res.usage == {"input_tokens": 120, "output_tokens": 15}
    assert sent["url"] == "http://host:11434/api/chat"
    p = sent["payload"]
    assert p["model"] == "m" and p["stream"] is False
    assert p["format"] == WorkOut.model_json_schema()
    assert [m["role"] for m in p["messages"]] == ["system", "user"]
    assert p["options"] == {"num_ctx": 8192}  # context size only; sampling left at model defaults


def test_truncated_and_invalid_outputs_fail_cleanly():
    truncated = OllamaLLM(post=fake_post({"message": {"content": '{"work": tr'}, "done_reason": "length"}, {}))
    res = truncated("s", "u", WorkOut)
    assert res.parsed is None and res.stop_reason == "length"

    invalid = OllamaLLM(post=fake_post({"message": {"content": '{"work": "maybe"}'}, "done_reason": "stop"}, {}))
    res = invalid("s", "u", WorkOut)
    assert res.parsed is None and "invalid output" in res.error


def test_over_long_prompt_raises_instead_of_being_truncated():
    import pytest

    from echo_sim.ollama_llm import PromptTooLong

    sent = {}
    llm = OllamaLLM(num_ctx=2048, post=fake_post({}, sent))
    with pytest.raises(PromptTooLong):
        llm("x" * 4000, "y", WorkOut)  # ~1300 tokens + 1024 reserve > 2048
    assert not sent  # nothing was sent
