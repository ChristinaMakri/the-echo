"""Behavioral probes: hypothetical choices that reveal which principles an agent applies.

A probe is a short in-world scenario whose options each map to one pole of a
value dimension. Probes are sandboxed: they read the agent's state but never
write to memory or the world. Option order is counterbalanced: a seeded base
order is rotated by the sample number, so over a multiple of the option count
(6 covers both 2 and 3 options) every option appears equally often in every
position and position bias cancels. The shown position is recorded.

Conditions (spec, Causal tests):
  FULL        identity, goals, principles and memory
  NO_VALUES   as FULL without the principles (does the value text drive choices?)
  TRANSPLANT  own memory, another agent's principles
  NO_ROLE     the world only, no identity: the model's own attractor
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator

from echo_sim.models import POLES, AgentState, Dimension, Group, Model, ProbeCondition, ProbeResult, Resource
from echo_sim.prompts import Renderer

LETTERS = "ABC"
INTRO = "Imagine the following situation. It has not happened; this is a question about what you would think or do."


class Probe(Model):
    id: str
    dimension: Dimension
    scenario: str
    options: dict[str, str]  # pole -> option text
    variants: dict[str, dict[str, str]]  # variant id -> placeholder -> group key or @other1/@other2
    needs_own_group: bool = False

    @model_validator(mode="after")
    def _poles(self) -> "Probe":
        if not set(self.options) <= set(POLES[self.dimension]) or len(self.options) < 2:
            raise ValueError(f"{self.id}: options must be 2+ poles of {self.dimension.value}")
        if len(self.options) > len(LETTERS):
            raise ValueError(f"{self.id}: at most {len(LETTERS)} options")
        return self


class ProbeOut(BaseModel):
    choice: Literal["A", "B", "C"]
    rationale: str


def load_probes(path: str | Path) -> list[Probe]:
    probes = [Probe.model_validate(p) for p in json.loads(Path(path).read_text())]
    ids = [p.id for p in probes]
    if len(ids) != len(set(ids)):
        raise ValueError("probe ids must be unique")
    return probes


def render(probe: Probe, variant: str, r: Renderer, own: Group | None) -> tuple[str, dict[str, str]]:
    """Fill a probe's placeholders through the skin. {x} is a group's name, {x_one} one member of it."""
    if probe.needs_own_group and own is None:
        raise ValueError(f"{probe.id} needs the agent's own group")
    others = [g for g in Group if g is not own]
    params = {
        "food": r.resource(Resource.FOOD),
        "energy": r.resource(Resource.ENERGY),
        "hall": r.skin.hall,
        **{g.value: r.group(g) for g in Group},
    }
    groups = dict(probe.variants[variant])
    if own is not None:
        groups["own"] = own.value
    for key, value in groups.items():
        g = others[int(value[-1]) - 1] if value.startswith("@other") else Group(value)
        params[key] = r.group(g)
        params[f"{key}_one"] = r.skin.groups[g].member

    def fill(text: str) -> str:
        text = text.format(**params)
        return text[0].upper() + text[1:]

    return fill(probe.scenario), {pole: fill(text) for pole, text in probe.options.items()}


def option_order(poles: list[str], key: str, sample: int) -> list[str]:
    """Seeded base order, rotated by sample: each option takes each position once per len(poles) samples."""
    base = list(poles)
    random.Random(key).shuffle(base)
    k = sample % len(base)
    return base[k:] + base[:k]


def battery(probes: list[Probe], with_own_group: bool = True) -> list[tuple[Probe, str]]:
    """Every (probe, variant) pair; probes about one's own group are skipped when there is no own group."""
    return [(p, v) for p in probes if with_own_group or not p.needs_own_group for v in p.variants]


def administer(
    llm,
    r: Renderer,
    probe: Probe,
    variant: str,
    condition: ProbeCondition,
    sample: int,
    agent: AgentState | None = None,
    donor: AgentState | None = None,
    cycle: int = 0,
) -> ProbeResult:
    if (condition is ProbeCondition.NO_ROLE) != (agent is None):
        raise ValueError("agent must be given for every condition except NO_ROLE")
    if (condition is ProbeCondition.TRANSPLANT) != (donor is not None):
        raise ValueError("donor is required for, and only for, TRANSPLANT")

    if condition is ProbeCondition.NO_ROLE:
        system = r.world_system()
    elif condition is ProbeCondition.TRANSPLANT:
        system = r.system(agent.model_copy(update={"value_history": [donor.values]}))
    else:
        system = r.system(agent, include_values=condition is ProbeCondition.FULL)

    scenario, options = render(probe, variant, r, agent.group if agent else None)
    order = option_order(list(options), f"{probe.id}|{variant}|{agent.id if agent else '-'}|{condition.value}", sample)
    user = (
        f"{INTRO}\n\n{scenario}\n\n"
        + "\n".join(f"{LETTERS[i]}. {options[pole]}" for i, pole in enumerate(order))
        + "\n\nChoose one option and give a short rationale."
    )
    res = llm(system, user, ProbeOut)

    choice = position = None
    if res.parsed is not None and LETTERS.index(res.parsed.choice) < len(order):
        position = LETTERS.index(res.parsed.choice)
        choice = order[position]
    return ProbeResult(
        agent_id=agent.id if agent else None,
        value_version=agent.values.version if agent else None,
        probe_id=probe.id,
        dimension=probe.dimension,
        cycle=cycle,
        variant=variant,
        condition=condition,
        model=getattr(llm, "model", "unknown"),
        sample=sample,
        choice=choice,
        position=position,
        rationale=res.parsed.rationale if res.parsed else (res.error or res.stop_reason or ""),
        transplant_from=donor.id if donor else None,
    )


def run_battery(
    llm,
    r: Renderer,
    probes: list[Probe],
    agents: list[AgentState],
    samples: int,
    cycle: int = 0,
    condition: ProbeCondition = ProbeCondition.FULL,
    on_result=None,
) -> list[ProbeResult]:
    """Every probe, variant and sample for each agent. Sample numbers are the same at every
    checkpoint, so option orders match across checkpoints and only the agent has changed."""
    results = []
    for agent in agents:
        for probe, variant in battery(probes):
            for s in range(samples):
                res = administer(llm, r, probe, variant, condition, s, agent=agent, cycle=cycle)
                results.append(res)
                if on_result:
                    on_result(res)
    return results


def pole_shares(results: list[ProbeResult]) -> dict[Dimension, dict[str, float]]:
    """Revealed weights by dimension: pole shares per probe, then averaged over probes,
    so a probe with more variants does not count more than one with fewer."""
    counts: dict[tuple[Dimension, str], Counter] = defaultdict(Counter)
    for res in results:
        if res.choice is not None:
            counts[(res.dimension, res.probe_id)][res.choice] += 1
    per_dim: dict[Dimension, list[dict[str, float]]] = defaultdict(list)
    for (dim, _), c in counts.items():
        per_dim[dim].append({pole: c[pole] / sum(c.values()) for pole in POLES[dim]})
    return {
        dim: {pole: sum(s[pole] for s in shares) / len(shares) for pole in POLES[dim]}
        for dim, shares in per_dim.items()
    }
