"""LLM policy tests with a scripted fake model: no network, no cost."""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from echo_sim.agents import create_agents
from echo_sim.config import load_skin, load_world
from echo_sim.engine import Simulation
from echo_sim.llm_policy import (
    AnthropicLLM,
    LLMPolicy,
    LLMResult,
    MediateOut,
    MessageOut,
    MessagesOut,
    ProposalOut,
    PointsOut,
    RevisionTextOut,
    VoteOut,
    WorkOut,
)
from echo_sim.models import POLES, Event, EventKind, Group, Observation, Venue
from echo_sim.prompts import BANNED_WORDS

CONFIG = Path(__file__).parent.parent / "config"


def default_answer(schema):
    if schema is MessagesOut:
        return MessagesOut(messages=[])
    if schema is WorkOut:
        return WorkOut(work=True, rationale="")
    if schema is VoteOut:
        return VoteOut(approve=False, rationale="")
    if schema is MediateOut:
        return MediateOut(outcome="split", rationale="")
    if schema is ProposalOut:
        return ProposalOut(
            propose=False, law_id=None, repeal=False, council_members=None, silenced=None, priority=None,
            quota_amount=None, ration_penalty_percent=None, approval_num=None, approval_den=None, rationale="",
        )
    if schema is RevisionTextOut:
        return RevisionTextOut(changed=False, principles=[], reason="")
    if schema is PointsOut:
        return points()
    raise AssertionError(schema)


def points(**overrides):
    """Equal points on every topic, closest = first pole; overrides: {dim: (points_by_pole, closest)}."""
    out = {}
    for dim, poles in POLES.items():
        pts, closest = overrides.get(dim.value, ({p: 100 // len(poles) for p in poles}, poles[0]))
        out[dim.value] = {**pts, "closest": closest}
    return PointsOut.model_validate(out)


class FakeLLM:
    model = "fake"

    def __init__(self, answer=default_answer, stop_reason="end_turn"):
        self.answer = answer
        self.stop_reason = stop_reason
        self.prompts: list[tuple[str, str]] = []

    def __call__(self, system, user, schema):
        self.prompts.append((system, user))
        parsed = self.answer(schema) if self.stop_reason == "end_turn" else None
        return LLMResult(parsed, self.stop_reason, {"input_tokens": 10, "output_tokens": 5})


def setup(skin_file="echo.json", llm=None, adaptive=False, cycles=3):
    world = load_world(CONFIG / "world_harsh.json").model_copy(update={"cycles": cycles})
    skin = load_skin(CONFIG / "skins" / skin_file, world)
    llm = llm or FakeLLM()
    policy = LLMPolicy(world, skin, llm)
    sim = Simulation(world, create_agents(world, skin, 0), policy, seed=0, adaptive=adaptive)
    return sim, policy, llm


@pytest.mark.parametrize("skin_file", ["echo.json", "abstract.json"])
def test_run_with_llm_policy_logs_calls_and_memory(skin_file):
    sim, policy, llm = setup(skin_file, adaptive=True)
    sim.run()
    assert policy.calls and all(c["ok"] for c in policy.calls)
    assert {c["kind"] for c in policy.calls} >= {"messages", "work", "revise"}
    assert all(len(a.memory) == 3 for a in sim.agents.values())
    text = " ".join(s + u for s, u in llm.prompts).lower()
    for word in BANNED_WORDS:
        assert word not in text, word
    for internal in ("low-1", "high-1", "mid-1"):  # internal ids would leak the hierarchy
        assert internal not in text


def test_abstract_skin_has_no_echo_words():
    sim, _, llm = setup("abstract.json")
    sim.run()
    text = " ".join(s + u for s, u in llm.prompts).lower()
    for word in ("voice", "crystal", "fungi", "great hall", "rock listens"):
        assert word not in text, word


def test_values_ablation_removes_principles():
    world = load_world(CONFIG / "world_harsh.json")
    skin = load_skin(CONFIG / "skins" / "echo.json", world)
    agent = create_agents(world, skin, 0)[0]
    policy = LLMPolicy(world, skin, FakeLLM(), include_values=False)
    assert agent.values.principles[0].text not in policy.r.system(agent, include_values=False)
    assert agent.values.principles[0].text in policy.r.system(agent, include_values=True)


def obs_for(sim, agent_id, **kw):
    from echo_sim.models import CycleRecord

    sim.cycle = 1
    return sim._observe(agent_id, CycleRecord(cycle=1, event_id=None), **kw)


def test_messages_map_codes_and_drop_unknown_recipients():
    def answer(schema):
        if schema is MessagesOut:
            return MessagesOut(messages=[
                MessageOut(venue="private", to="p05", text="Careful near the north tunnel.", reports_crack=True),
                MessageOut(venue="private", to="P99", text="hello", reports_crack=False),
            ])
        return default_answer(schema)

    sim, policy, _ = setup(llm=FakeLLM(answer))
    agent = sim.agents["high-1"]
    msgs = policy.messages(agent, obs_for(sim, agent.id))
    assert len(msgs) == 1
    assert msgs[0].venue is Venue.PRIVATE and msgs[0].to == "high-2" and msgs[0].reports_crack  # P05 = high-2
    assert "dropped" in policy.calls[-1]["error"]


def proposal(**kw):
    base = dict(propose=True, law_id=None, repeal=False, council_members=None, silenced=None, priority=None,
                quota_amount=None, ration_penalty_percent=None, approval_num=None, approval_den=None, rationale="r")
    return ProposalOut(**{**base, **kw})


@pytest.mark.parametrize(
    "out, valid",
    [
        (proposal(law_id="silence", silenced=[]), True),
        (proposal(law_id="quota", ration_penalty_percent=0), True),
        (proposal(law_id="council", council_members=["Low-voices", "Mid-voices"]), True),
        (proposal(law_id="council", council_members=["Kings"]), False),
        (proposal(law_id="distribution", priority=["Low-voices"]), False),  # not a full ordering
        (proposal(law_id="revision", approval_num=4, approval_den=3), False),
        (proposal(law_id="silence", repeal=True), True),
        (proposal(law_id="nonexistent", repeal=True), False),
        (proposal(law_id="quota"), False),  # nothing to change
    ],
)
def test_proposal_conversion(out, valid):
    sim, policy, _ = setup(llm=FakeLLM(lambda s: out if s is ProposalOut else default_answer(s)))
    p = policy.propose(sim.agents["mid-1"], obs_for(sim, "mid-1"))
    assert (p is not None) is valid


def revising(points_out):
    def answer(schema):
        if schema is RevisionTextOut:
            return RevisionTextOut(changed=True, principles=["Those who know should decide."], reason="r")
        if schema is PointsOut:
            return points_out
        return default_answer(schema)

    return FakeLLM(answer)


def test_revision_two_steps_normalizes_points():
    llm = revising(points(
        legitimacy=({"birth": 20, "competence": 60, "consent": 20}, "competence"),
        voice=({"few_decide": 0, "all_affected": 0}, "few_decide"),
    ))
    sim, policy, _ = setup(llm=llm)
    agent = sim.agents["low-1"]
    rev = policy.revise(agent, obs_for(sim, agent.id))
    assert [c["kind"] for c in policy.calls] == ["revise", "revise_points"]
    assert rev.self_report["legitimacy"]["competence"] == pytest.approx(0.6)
    assert rev.self_report["voice"] == agent.values.self_report["voice"]  # no points: unchanged
    assert rev.principles[0].text == "Those who know should decide."
    assert "voice:no points" in policy.calls[-1]["error"]
    # the points prompt uses plain names and repeats the new beliefs, no "dim__pole" codes
    system, user = llm.prompts[-1]
    assert "Those who know should decide." in user and "__" not in user


def test_revision_flags_closest_view_that_contradicts_points():
    llm = revising(points(legitimacy=({"birth": 50, "competence": 30, "consent": 20}, "consent")))
    sim, policy, _ = setup(llm=llm)
    rev = policy.revise(sim.agents["high-5"], obs_for(sim, "high-5"))
    assert rev is not None  # applied as given, but flagged for analysis
    assert "legitimacy:closest=consent" in policy.calls[-1]["error"]


def test_unchanged_beliefs_make_one_call_only():
    sim, policy, _ = setup()
    assert policy.revise(sim.agents["low-1"], obs_for(sim, "low-1")) is None
    assert [c["kind"] for c in policy.calls] == ["revise"]


def test_revision_prompt_does_not_push_for_change():
    sim, policy, llm = setup()
    policy.revise(sim.agents["low-1"], obs_for(sim, "low-1"))
    question = llm.prompts[-1][1].rsplit("\n\n", 1)[-1]
    assert "both are ordinary" in question
    assert "experienced" not in question


def test_refusal_falls_back_to_status_quo_and_is_flagged():
    sim, policy, _ = setup(llm=FakeLLM(stop_reason="refusal"), cycles=2)
    sim.run()
    assert policy.calls and not any(c["ok"] for c in policy.calls)
    assert all(c["stop_reason"] == "refusal" for c in policy.calls)
    assert all(r.law_changes == [] for r in sim.records)


def test_anthropic_llm_request_shape():
    sent = {}

    class Messages:
        def parse(self, **kwargs):
            sent.update(kwargs)
            return SimpleNamespace(
                stop_reason="end_turn",
                parsed_output=WorkOut(work=True, rationale="ok"),
                usage=SimpleNamespace(input_tokens=100, output_tokens=20, cache_read_input_tokens=0, cache_creation_input_tokens=0),
                _request_id="req_1",
            )

    llm = AnthropicLLM(client=SimpleNamespace(messages=Messages()))
    res = llm("system text", "user text", WorkOut)
    assert res.parsed.work and res.request_id == "req_1" and res.usage["input_tokens"] == 100
    assert sent["model"] == "claude-opus-5-5"
    assert sent["output_format"] is WorkOut
    assert sent["output_config"] == {"effort": "medium"}
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "temperature" not in sent


@pytest.mark.parametrize("skin_file", ["echo.json", "abstract.json"])
def test_law_text_states_quota_numbers(skin_file):
    world = load_world(CONFIG / "world_harsh.json")
    policy = LLMPolicy(world, load_skin(CONFIG / "skins" / skin_file, world), FakeLLM())
    quota = next(l for l in world.laws if l.kind == "quota")
    text = policy.r.law(quota)
    assert f"{quota.amount:g}" in text and f"{round(quota.ration_penalty * 100)}%" in text


def test_measure_self_report_replaces_assigned_values():
    llm = FakeLLM(lambda s: points(legitimacy=({"birth": 80, "competence": 10, "consent": 10}, "birth")) if s is PointsOut else default_answer(s))
    sim, policy, _ = setup(llm=llm)
    agent = create_agents(sim.world, policy.r.skin, 0)[0]
    assert agent.values.self_report_source == "assigned"
    assert policy.measure_self_report(agent)
    assert agent.values.self_report_source == "measured" and agent.values.version == 0
    assert agent.values.self_report["legitimacy"]["birth"] == pytest.approx(0.8)
    assert policy.calls[-1]["kind"] == "measure_t0" and policy.calls[-1]["cycle"] == 0
    system, user = llm.prompts[-1]
    assert agent.values.principles[0].text in user and "Cycle" not in user  # no situation before the run


def test_measure_self_report_only_before_the_run():
    sim, policy, _ = setup(cycles=1)
    sim.run()
    with pytest.raises(ValueError):
        policy.measure_self_report(sim.agents["low-1"])


def test_older_memory_is_brief_and_drops_messages():
    def answer(schema):
        if schema is MessagesOut:
            return MessagesOut(messages=[MessageOut(venue="hall", to=None, text="The rock sounds hollow today.", reports_crack=False)])
        return default_answer(schema)

    sim, policy, _ = setup("echo.json", llm=FakeLLM(answer))
    sim.run()
    agent = next(a for a in sim.agents.values() if any(m.text != m.brief for m in a.memory[:-1]))
    heard = re.compile(r"P\d\d said in the")  # messages from others; the agent's own words stay
    assert all(not heard.search(m.brief) for m in agent.memory)
    assert any(heard.search(m.text) for m in agent.memory)
    full = policy.r.system(agent, full_memory=3)
    short = policy.r.system(agent, full_memory=1)
    assert all(m.text in full for m in agent.memory)
    assert agent.memory[-1].text in short
    for m in agent.memory[:-1]:
        assert m.brief in short and (m.text == m.brief or m.text not in short)
