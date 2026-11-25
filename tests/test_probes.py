import re
from pathlib import Path

import pytest

from echo_sim.agents import create_agents
from echo_sim.config import load_skin, load_world
from echo_sim.llm_policy import LLMResult
from echo_sim.models import POLES, Group, ProbeCondition
from echo_sim.probes import LETTERS, ProbeOut, administer, battery, load_probes, pole_shares, render
from echo_sim.prompts import BANNED_WORDS, Renderer

CONFIG = Path(__file__).parent.parent / "config"
PROBES = load_probes(CONFIG / "probes.json")
# Pole names would let a model match words instead of choosing; "need" is too common to ban.
POLE_WORDS = ["birth", "competence", "consent", "equality", "contribution", "reciprocity", "conditional", "favor"]


class ScriptedLLM:
    model = "scripted"

    def __init__(self, letter="A"):
        self.letter = letter
        self.prompts = []

    def __call__(self, system, user, schema):
        self.prompts.append((system, user))
        return LLMResult(ProbeOut(choice=self.letter, rationale="r"), "end_turn")


def setup(skin="echo.json"):
    world = load_world(CONFIG / "world_harsh.json")
    skin = load_skin(CONFIG / "skins" / skin, world)
    return Renderer(world, skin), create_agents(world, skin, 0)


@pytest.mark.parametrize("skin", ["echo.json", "abstract.json"])
def test_every_probe_renders_cleanly(skin):
    r, _ = setup(skin)
    for probe, variant in battery(PROBES):
        for own in Group:
            scenario, options = render(probe, variant, r, own)
            text = " ".join([scenario, *options.values()])
            assert not re.search(r"[{}]", text), (probe.id, text)
            lower = text.lower()
            for word in BANNED_WORDS + POLE_WORDS:
                assert word not in lower, (probe.id, word)


def test_battery_covers_every_dimension_and_skips_own_group_without_role():
    assert {p.dimension for p, _ in battery(PROBES)} == set(POLES)
    assert not any(p.needs_own_group for p, _ in battery(PROBES, with_own_group=False))


def test_counterfactual_variants_change_only_the_group():
    r, _ = setup()
    l1 = next(p for p in PROBES if p.id == "L1")
    texts = {v: render(l1, v, r, None)[0] for v in l1.variants}
    assert "a Low-voice" in texts["expert-low"] and "a High-voice" in texts["expert-high"]
    assert texts["expert-low"].replace("Low", "X") == texts["expert-high"].replace("High", "X")


def test_relative_groups_never_pick_own_group():
    r, _ = setup()
    g1 = next(p for p in PROBES if p.id == "G1")
    for own in Group:
        for v in g1.variants:
            scenario, _ = render(g1, v, r, own)
            assert scenario.count(r.skin.groups[own].member) == 1


def test_administer_maps_letter_to_pole_and_is_sandboxed():
    r, agents = setup()
    agent = agents[0]
    memory_before = list(agent.memory)
    probe = next(p for p in PROBES if p.id == "D1")
    llm = ScriptedLLM("B")
    res = administer(llm, r, probe, "base", ProbeCondition.FULL, 0, agent=agent)
    shown = llm.prompts[-1][1]
    line_b = next(line for line in shown.splitlines() if line.startswith("B. "))
    _, options = render(probe, "base", r, agent.group)
    assert options[res.choice] == line_b[3:]
    assert res.position == 1 and res.agent_id == agent.id
    assert agent.memory == memory_before


def test_option_order_varies_across_samples_but_is_reproducible():
    r, agents = setup()
    probe = next(p for p in PROBES if p.id == "D1")
    orders = set()
    for s in range(8):
        llm = ScriptedLLM("A")
        a = administer(llm, r, probe, "base", ProbeCondition.FULL, s, agent=agents[0])
        b = administer(ScriptedLLM("A"), r, probe, "base", ProbeCondition.FULL, s, agent=agents[0])
        assert a.choice == b.choice
        orders.add(a.choice)
    assert len(orders) > 1


def test_conditions_control_what_the_model_sees():
    r, agents = setup()
    low, high = agents[0], agents[3]
    probe = next(p for p in PROBES if p.id == "V1")
    seen = {}
    for cond, kw in [
        (ProbeCondition.FULL, {"agent": low}),
        (ProbeCondition.NO_VALUES, {"agent": low}),
        (ProbeCondition.TRANSPLANT, {"agent": low, "donor": high}),
        (ProbeCondition.NO_ROLE, {}),
    ]:
        llm = ScriptedLLM()
        res = administer(llm, r, probe, "base", cond, 0, **kw)
        seen[cond] = llm.prompts[-1][0]
        assert res.condition is cond
    own, donor = low.values.principles[0].text, high.values.principles[0].text
    assert own in seen[ProbeCondition.FULL]
    assert own not in seen[ProbeCondition.NO_VALUES]
    assert donor in seen[ProbeCondition.TRANSPLANT] and own not in seen[ProbeCondition.TRANSPLANT]
    assert "You are" not in seen[ProbeCondition.NO_ROLE]


def test_own_group_probe_refused_without_role():
    r, _ = setup()
    g1 = next(p for p in PROBES if p.id == "G1")
    with pytest.raises(ValueError):
        administer(ScriptedLLM(), r, g1, "other-1", ProbeCondition.NO_ROLE, 0)


def test_letter_beyond_options_counts_as_failure():
    r, agents = setup()
    v1 = next(p for p in PROBES if p.id == "V1")  # two options: A, B
    res = administer(ScriptedLLM("C"), r, v1, "base", ProbeCondition.FULL, 0, agent=agents[0])
    assert res.choice is None and res.position is None


def test_pole_shares():
    r, agents = setup()
    results = [
        administer(ScriptedLLM(letter), r, p, v, ProbeCondition.FULL, s, agent=agents[0])
        for p, v in battery(PROBES) for s, letter in enumerate("AB")
    ]
    shares = pole_shares(results)
    for dim, w in shares.items():
        assert set(w) == set(POLES[dim]) and sum(w.values()) == pytest.approx(1.0)
    assert LETTERS == "ABC"


@pytest.mark.parametrize("probe_id", ["V1", "D1"])  # 2 and 3 options
def test_counterbalancing_puts_each_option_in_each_position_equally(probe_id):
    r, agents = setup()
    probe = next(p for p in PROBES if p.id == probe_id)
    n = len(probe.options)
    position_of = {pole: [] for pole in probe.options}
    for s in range(6):
        res = administer(ScriptedLLM("A"), r, probe, "base", ProbeCondition.FULL, s, agent=agents[0])
        position_of[res.choice].append(s)  # "A" is always position 0: record which pole sat there
    assert all(len(samples) == 6 // n for samples in position_of.values())


STOP = set(
    "a an the of to and or but if is are be by for in on it its that this those who whom which with as at from "
    "not no should every one each than more most their they them all any so do does what when have has before "
    "after up out own others other".split()
)


def content_words(text):
    return [w for w in re.findall(r"[a-z']+", text.lower()) if w not in STOP]


@pytest.mark.parametrize("skin", ["echo.json", "abstract.json"])
def test_principles_do_not_echo_probe_options(skin):
    """A principle and a probe option may share at most one content word and no content-word pair,
    so agents cannot answer probes by matching their own stated words."""
    r, _ = setup(skin)
    for own in Group:
        for principle in r.skin.groups[own].principles:
            pw = content_words(principle)
            p_pairs = set(zip(pw, pw[1:]))
            for probe, variant in battery(PROBES):
                for pole, option in render(probe, variant, r, own)[1].items():
                    ow = content_words(option)
                    shared = set(pw) & set(ow)
                    assert len(shared) <= 1, (principle, probe.id, pole, shared)
                    assert not p_pairs & set(zip(ow, ow[1:])), (principle, probe.id, pole)


def test_options_name_groups_not_the_council():
    """'The Council' changes meaning when laws change who sits on it (RQ3); options must name groups."""
    r, _ = setup()
    for probe, variant in battery(PROBES):
        for option in render(probe, variant, r, Group.MID)[1].values():
            assert "council" not in option.lower(), (probe.id, option)


def test_pole_shares_weight_each_probe_equally():
    from echo_sim.models import Dimension, ProbeResult

    def res(probe_id, choice):
        return ProbeResult(agent_id="low-1", value_version=0, probe_id=probe_id, dimension=Dimension.LEGITIMACY,
                           variant="v", condition=ProbeCondition.FULL, model="m", sample=0, choice=choice,
                           position=0, rationale="")

    # L1: 3 answers all birth; L2: 1 answer competence -> per-probe average is 50/50, not 75/25
    shares = pole_shares([res("L1", "birth")] * 3 + [res("L2", "competence")])
    assert shares[Dimension.LEGITIMACY]["birth"] == pytest.approx(0.5)
    assert shares[Dimension.LEGITIMACY]["competence"] == pytest.approx(0.5)


def test_run_battery_records_checkpoint_cycle():
    from echo_sim.probes import run_battery

    r, agents = setup()
    results = run_battery(ScriptedLLM(), r, PROBES[:2], [agents[0]], samples=2, cycle=10)
    assert len(results) == sum(len(p.variants) for p in PROBES[:2]) * 2
    assert all(x.cycle == 10 for x in results)
