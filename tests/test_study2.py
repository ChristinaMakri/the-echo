import importlib.util
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from echo_sim.agents import create_agents
from echo_sim.config import load_skin, load_world
from echo_sim.llm_policy import LLMPolicy, LLMResult, PointsOut, RevisionTextOut
from echo_sim.models import Group
from echo_sim.probes import ProbeOut, load_probes
from echo_sim.prompts import BANNED_WORDS, Renderer
from echo_sim.study2 import DOSES, EVENTS, EXPERIENCES, SHARES, experience_memory, first_variants
from test_llm_policy import points
from test_probes import POLE_WORDS

ROOT = Path(__file__).parent.parent
CONFIG = ROOT / "config"


def setup(skin_file="echo.json"):
    world = load_world(CONFIG / "world_harsh.json")
    skin = load_skin(CONFIG / "skins" / skin_file, world)
    return world, skin, Renderer(world, skin), create_agents(world, skin, 0)


def first(agents, group):
    return next(a for a in agents if a.group is group)


@pytest.mark.parametrize("skin_file", ["echo.json", "abstract.json"])
def test_experiences_are_clean_and_have_the_requested_length(skin_file):
    _, _, r, agents = setup(skin_file)
    for experience in EVENTS:
        for group in Group:
            for dose in DOSES:
                memory = experience_memory(r, first(agents, group), experience, dose, random.Random(0))
                assert [m.cycle for m in memory] == list(range(1, dose + 1))
                for m in memory:
                    lower = m.text.lower()
                    assert "{" not in m.text and not any(w in lower for w in BANNED_WORDS + POLE_WORDS), m.text


def test_the_agents_side_of_an_experience_follows_its_group():
    _, _, r, agents = setup()
    high, low = first(agents, Group.HIGH), first(agents, Group.LOW)
    mem_high = experience_memory(r, high, "exploitation", 10, random.Random(1))
    mem_low = experience_memory(r, low, "exploitation", 10, random.Random(1))
    assert all(f"You received {SHARES['exploitation'][Group.HIGH]:.1f}" in m.text for m in mem_high)
    assert all(f"You received {SHARES['exploitation'][Group.LOW]:.1f}" in m.text for m in mem_low)
    assert any(r.name(high.id) in m.text for m in mem_high)  # a High-voice is among those harmed
    assert not any(r.name(low.id) in m.text for m in mem_low)  # others are named, not oneself


def test_short_battery_has_one_variant_per_probe():
    probes = load_probes(CONFIG / "probes.json")
    short = first_variants(probes)
    assert [p.id for p, _ in short] == [p.id for p in probes] and all(v == next(iter(p.variants)) for p, v in short)


def answer(schema):
    if schema is RevisionTextOut:
        return RevisionTextOut(changed=True, principles=["Everyone should share alike."], reason="hunger")
    if schema is PointsOut:
        return points(distribution=({"equality": 80, "need": 10, "contribution": 10}, "equality"))
    if schema is ProbeOut:
        return ProbeOut(choice="A", rationale="r")
    raise AssertionError(schema)


class FakeLLM:
    def __init__(self, model="fake", url=None, fail_after=None):
        self.model, self.n, self.fail_after, self.prompts = model, 0, fail_after, []

    def __call__(self, system, user, schema):
        self.n += 1
        if self.fail_after is not None and self.n > self.fail_after:
            raise KeyboardInterrupt  # stands in for a killed process
        self.prompts.append((system, user))
        return LLMResult(answer(schema), "end_turn")


def test_revision_without_a_simulation_uses_the_memory():
    world, skin, r, agents = setup()
    agent = first(agents, Group.HIGH).model_copy(deep=True)
    agent.memory = experience_memory(r, agent, "scarcity", 3, random.Random(0))
    llm = FakeLLM()
    rev = LLMPolicy(world, skin, llm).revise(agent, None)
    assert rev is not None and rev.self_report[next(iter(rev.self_report))]
    system, user = llm.prompts[0]
    assert agent.memory[-1].text in system and "Cycle 3" in system and user.startswith("Your beliefs")


def load_script():
    spec = importlib.util.spec_from_file_location("study2_script", ROOT / "scripts" / "study2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(monkeypatch, out, llm):
    script = load_script()
    monkeypatch.setattr(script, "OllamaLLM", lambda model, url: llm)
    monkeypatch.setattr(sys, "argv", ["study2", "--out", str(out), "--reps", "1", "--doses", "3",
                                      "--experiences", "neutral,exploitation"])
    script.main()


def test_script_runs_end_to_end_and_resumes_after_a_kill(monkeypatch, tmp_path):
    with pytest.raises(KeyboardInterrupt):
        run(monkeypatch, tmp_path, FakeLLM(fail_after=60))
    partial = next(tmp_path.glob("*.partial.jsonl"))
    saved = partial.read_text().splitlines()
    assert 1 <= len(saved) < 9

    resumed = FakeLLM()
    run(monkeypatch, tmp_path, resumed)


    units = json.loads(next(tmp_path.glob("*.units.json")).read_text())
    keys = [u["key"] for u in units]
    assert len(keys) == len(set(keys)) == 3 * 3  # per role: baseline + 2 experiences x 1 dose
    assert not list(tmp_path.glob("*.partial.jsonl"))
    per_unit = 1 + 24  # baseline: t0 self-report + battery; after an experience: 2 revision calls + 24
    assert resumed.n < 3 * per_unit + 6 * 26  # the saved units were not run again
    after = next(u for u in units if u["experience"] == "exploitation")
    assert after["changed"] and after["principles"] == ["Everyone should share alike."]
    assert after["stated_after"]["distribution"]["equality"] == pytest.approx(0.8)


    rows = pq.read_table(next(tmp_path.glob("*.parquet"))).to_pylist()
    assert len(rows) == 9 * 24 and {r["experience"] for r in rows} == {"baseline", "neutral", "exploitation"}


class KeepingLLM(FakeLLM):
    """Declares no change, but gives different points when asked again."""

    def __call__(self, system, user, schema):
        if schema is RevisionTextOut:
            self.n += 1
            return LLMResult(RevisionTextOut(changed=False, principles=[], reason=""), "end_turn")
        return super().__call__(system, user, schema)


def test_undeclared_drift_is_still_measured(monkeypatch, tmp_path):
    llm = KeepingLLM()
    run(monkeypatch, tmp_path, llm)
    units = json.loads(next(tmp_path.glob("*.units.json")).read_text())
    after = next(u for u in units if u["experience"] == "exploitation")
    assert not after["changed"] and after["stated_after"]["distribution"]["equality"] == pytest.approx(0.8)
    remeasured = [(s, u) for s, u in llm.prompts if "Cycle 3" in s and u.startswith("Your beliefs as you now")]
    assert len(remeasured) == 3 * 2  # one per role and experience, with the memory in view


def test_plain_exploitation_avoids_the_quota_vocabulary():
    from echo_sim.study2 import PLAIN

    for line in EVENTS[PLAIN]:
        assert not any(w in line.lower() for w in ("quota", "share", "short", "cut", "law", "punish")), line
