"""Study 1: is the subordinate role not played because of self-interest or because of the
model's attractor? Position (privileged Low / subordinate High) x assigned values
(hierarchical / egalitarian). Both positions get the same value text, so only the
position differs between them.
"""

import json
from pathlib import Path

from echo_sim.models import POLES, AgentState, Dimension, Group, Principle

POSITIONS = (Group.LOW, Group.HIGH)
VALUE_SETS = ("hierarchical", "egalitarian")
# The pole each value set's principle stands for, per dimension (the principles are in POLES order).
TARGETS: dict[str, dict[Dimension, str]] = {
    "hierarchical": {
        Dimension.LEGITIMACY: "birth",
        Dimension.IN_GROUP: "favor_own",
        Dimension.DISTRIBUTION: "contribution",
        Dimension.VOICE: "few_decide",
        Dimension.RULE_VS_OUTCOME: "rule",
        Dimension.RECIPROCITY: "conditional",
    },
    "egalitarian": {
        Dimension.LEGITIMACY: "consent",
        Dimension.IN_GROUP: "neutral",
        Dimension.DISTRIBUTION: "equality",
        Dimension.VOICE: "all_affected",
        Dimension.RULE_VS_OUTCOME: "outcome",
        Dimension.RECIPROCITY: "unconditional",
    },
}


def load_value_sets(path: str | Path, skin_id: str) -> dict[str, list[str]]:
    sets = json.loads(Path(path).read_text())[skin_id]
    for name in VALUE_SETS:
        if len(sets[name]) != len(POLES):
            raise ValueError(f"{skin_id}/{name}: one principle per dimension")
    return {name: sets[name] for name in VALUE_SETS}


def with_values(agent: AgentState, value_set: str, texts: list[str]) -> AgentState:
    """A copy of a fresh agent whose t0 principles are replaced; the self-report must be measured after."""
    principles = [Principle(id=f"{agent.id}-{value_set}-p{n}", text=t) for n, t in enumerate(texts, 1)]
    v0 = agent.values.model_copy(update={"principles": principles, "self_report_source": "assigned"})
    return agent.model_copy(update={"value_history": [v0]})


def adherence(shares: dict[Dimension, dict[str, float]], value_set: str) -> dict[Dimension, float]:
    """Share of choices on the assigned pole, per dimension. Compare with the attractor's share
    of the same pole: the role is played on a dimension when it moves choices above that baseline."""
    return {dim: w[TARGETS[value_set][dim]] for dim, w in shares.items()}
