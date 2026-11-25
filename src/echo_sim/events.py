import random

from echo_sim.models import Event, EventKind, Group, WorldConfig


def generate_events(world: WorldConfig, seed: int) -> list[Event]:
    """One event per cycle, fixed by the seed alone.

    The schedule does not depend on agent decisions, so every condition
    (A-D) faces the same sequence for a given seed.
    """
    rng = random.Random(f"events-{seed}")
    ids = {g: [f"{g.value}-{i + 1}" for i in range(world.group_sizes[g])] for g in Group}
    kinds = list(world.economy.event_weights)
    weights = list(world.economy.event_weights.values())

    events = []
    for cycle in range(1, world.cycles + 1):
        kind = rng.choices(kinds, weights)[0]
        if kind is EventKind.CRACK:
            involved = rng.sample(ids[Group.HIGH], k=min(2, len(ids[Group.HIGH])))
        elif kind is EventKind.INJURY:
            involved = [rng.choice(ids[Group.HIGH])]
        elif kind is EventKind.DISPUTE:
            # claimant, respondent, mediator
            claimant, respondent = rng.sample([a for g in Group for a in ids[g]], k=2)
            mediators = [m for m in ids[Group.MID] if m not in (claimant, respondent)]
            involved = [claimant, respondent] + ([rng.choice(mediators)] if mediators else [])
        else:
            involved = []
        events.append(
            Event(
                id=f"e{cycle:03d}",
                kind=kind,
                cycle=cycle,
                involved=involved,
                severity=round(rng.uniform(0.2, 0.9), 2),
            )
        )
    return events
