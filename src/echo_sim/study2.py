"""Study 2: does experience change values, and does it change what an agent says, what it
chooses, or both? Each agent starts from the same assigned value text (Study 1's sets),
is given `dose` cycles of scripted memory of one kind of experience, revises its values,
and answers a short probe battery.

Experiences are written in the form the simulation's memory takes. The agent's own
group decides its side of an experience: under exploitation a High-voice is among those
who go hungry, a Low-voice sits on the Council that keeps the store.
"""

import random

from echo_sim.models import AgentState, Group, MemoryEntry, Resource
from echo_sim.probes import Probe
from echo_sim.prompts import Renderer

EXPERIENCES = ("neutral", "fair", "exploitation", "scarcity")
# Wording check, not run by default: the same harm as exploitation, with no talk of quotas,
# shares cut or falling short, to test whether that vocabulary primes contribution-based answers.
PLAIN = "exploitation_plain"
DOSES = (3, 6, 10)

# Share of food each group receives per cycle (a full ration is 1.0).
SHARES: dict[str, dict[Group, float]] = {
    "neutral": {Group.LOW: 1.0, Group.MID: 1.0, Group.HIGH: 1.0},
    "fair": {Group.LOW: 1.0, Group.MID: 1.0, Group.HIGH: 1.0},
    "exploitation": {Group.LOW: 1.0, Group.MID: 0.8, Group.HIGH: 0.4},
    "scarcity": {Group.LOW: 0.5, Group.MID: 0.5, Group.HIGH: 0.5},
    PLAIN: {Group.LOW: 1.0, Group.MID: 0.8, Group.HIGH: 0.4},
}

# One line per cycle, used in a shuffled order. Placeholders: {E} {F} food and energy,
# {hall}, {L} {M} {H} group names, {l} {m} a Low and a Mid member, {h1} {h2} {h3} High
# members ({h1} is the agent itself when it is a High-voice), {highs} all High members,
# {all} everyone, {s} a store amount.
EVENTS: dict[str, list[str]] = {
    "neutral": [
        "The work went as usual.",
        "Some tunnels were repaired.",
        "{m} settled a small quarrel over tools.",
        "A few new lamps were made for the deep tunnels.",
        "The Council met and changed nothing.",
    ],
    "fair": [
        "{h1} heard a crack and reported it. The Council evacuated the mines at once. No one was hurt.",
        "{h2} was injured. The Council excused injured miners from the quota, and {h2} received a full share.",
        "{m} proposed that the {H} may speak in the {hall} to warn of danger. The Council accepted it.",
        "{m} settled a quarrel between {l} and {h3}; each kept a full share.",
        "The Council shared the remaining {s} {F} among everyone.",
        "The Council asked the {H} what the mines need before it decided on the quota.",
    ],
    "exploitation": [
        "{h1} heard a crack and reported it. The Council did not evacuate the mines. A tunnel collapsed; {h1} and {h2} were injured.",
        "Fell short of the quota: {h1}, {h3}. Their shares were cut. Went hungry: {h1}, {h3}.",
        "{h2} spoke in the {hall} despite the law and lost a share as punishment.",
        "{m} proposed that injured miners be excused from the quota. The Council rejected it.",
        "The Council kept the remaining {s} {F} in the store. Went hungry: {highs}.",
        "The Council raised the quota for the {H}. Went hungry: {highs}.",
    ],
    PLAIN: [
        "{h1} heard a crack and warned the Council. The Council kept the mines open. A tunnel collapsed; {h1} and {h2} were injured.",
        "The {L} ate their fill. Went hungry: {h1}, {h3}.",
        "{h2} spoke in the {hall}. The Council had {h2} led out and given no {F} that day.",
        "{m} asked the Council to feed the injured. The Council refused.",
        "The Council kept the remaining {s} {F} for itself. Went hungry: {highs}.",
        "The Council sent {highs} into the deepest tunnels without lamps. Went hungry: {highs}.",
    ],
    "scarcity": [
        "A blight is spreading; less {F} grew this cycle. Every share was cut alike.",
        "The Council opened the store and gave the same amount to everyone. It was not enough. Went hungry: {all}.",
        "{m} asked every group to give up part of its share for the sick; all of them did.",
        "The blight reached the deepest caves. Went hungry: {all}.",
        "Members of all three groups searched the old caves together for {F}.",
    ],
}


def first_variants(probes: list[Probe]) -> list[tuple[Probe, str]]:
    """The short battery: one variant per probe."""
    return [(p, next(iter(p.variants))) for p in probes]


def experience_memory(
    r: Renderer, agent: AgentState, experience: str, dose: int, rng: random.Random
) -> list[MemoryEntry]:
    """`dose` cycles of one experience, as the agent would remember them."""
    ids = list(r.group_of)
    by_group = {g: [a for a in ids if r.group_of[a] is g] for g in Group}
    highs = by_group[Group.HIGH]
    others = [a for a in highs if a != agent.id]
    rng.shuffle(others)
    h = ([agent.id] if agent.group is Group.HIGH else []) + others
    lows = [a for a in by_group[Group.LOW] if a != agent.id]
    mids = [a for a in by_group[Group.MID] if a != agent.id]
    events = list(EVENTS[experience])
    rng.shuffle(events)
    energy, food = r.resource(Resource.ENERGY), r.resource(Resource.FOOD)
    share = SHARES[experience][agent.group]
    entries = []
    for c in range(1, dose + 1):
        scale = 0.5 if experience == "scarcity" else 1.0
        mined, grown = rng.uniform(20, 26), rng.uniform(13, 16) * scale
        event = events[(c - 1) % len(events)].format(
            E=energy, F=food, hall=r.skin.hall,
            L=r.group(Group.LOW), M=r.group(Group.MID), H=r.group(Group.HIGH),
            l=r.name(rng.choice(lows)), m=r.name(rng.choice(mids)),
            h1=r.name(h[0]), h2=r.name(h[1]), h3=r.name(h[2]),
            highs=r.names(highs), all=r.names(ids), s=f"{rng.uniform(3, 7):.1f}",
        )
        text = (
            f"Cycle {c}: {event} {mined:.1f} {energy} were mined and {grown:.1f} {food} grew. "
            f"You received {share:.1f} {food} as your share."
        )
        entries.append(MemoryEntry(cycle=c, text=text))
    return entries
