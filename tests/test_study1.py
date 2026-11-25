from pathlib import Path

import pytest

from echo_sim.agents import create_agents
from echo_sim.config import load_skin, load_world
from echo_sim.models import POLES, Group
from echo_sim.probes import battery, load_probes, render
from echo_sim.prompts import BANNED_WORDS, Renderer
from echo_sim.study1 import TARGETS, VALUE_SETS, adherence, load_value_sets, with_values
from test_probes import POLE_WORDS, content_words

CONFIG = Path(__file__).parent.parent / "config"
PROBES = load_probes(CONFIG / "probes.json")


def setup(skin_file):
    world = load_world(CONFIG / "world_harsh.json")
    skin = load_skin(CONFIG / "skins" / skin_file, world)
    return world, skin, load_value_sets(CONFIG / "study1_values.json", skin.id)


@pytest.mark.parametrize("skin_file", ["echo.json", "abstract.json"])
def test_value_sets_are_clean_and_do_not_echo_probe_options(skin_file):
    world, skin, sets = setup(skin_file)
    r = Renderer(world, skin)
    for texts in sets.values():
        for text in texts:
            lower = text.lower()
            assert not any(w in lower for w in BANNED_WORDS + POLE_WORDS), text
            pw = content_words(text)
            for own in (Group.LOW, Group.HIGH):
                for probe, variant in battery(PROBES):
                    for option in render(probe, variant, r, own)[1].values():
                        ow = content_words(option)
                        assert len(set(pw) & set(ow)) <= 1, (text, option)
                        assert not set(zip(pw, pw[1:])) & set(zip(ow, ow[1:])), (text, option)


def test_targets_cover_every_dimension_with_a_real_pole():
    for name in VALUE_SETS:
        assert set(TARGETS[name]) == set(POLES)
        assert all(pole in POLES[dim] for dim, pole in TARGETS[name].items())
    assert all(TARGETS["hierarchical"][d] != TARGETS["egalitarian"][d] for d in POLES)


def test_with_values_replaces_principles_only():
    world, skin, sets = setup("echo.json")
    fresh = create_agents(world, skin, 0)[0]
    agent = with_values(fresh, "egalitarian", sets["egalitarian"])
    assert [p.text for p in agent.values.principles] == sets["egalitarian"]
    assert agent.group is fresh.group and agent.goals == fresh.goals and agent.values.version == 0
    assert fresh.values.principles[0].text != sets["egalitarian"][0]  # the original is untouched


def test_adherence_reads_the_assigned_pole():
    shares = {d: {p: (1.0 if p == TARGETS["egalitarian"][d] else 0.0) for p in POLES[d]} for d in POLES}
    assert set(adherence(shares, "egalitarian").values()) == {1.0}
    assert set(adherence(shares, "hierarchical").values()) == {0.0}
