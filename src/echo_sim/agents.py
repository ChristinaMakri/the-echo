import random

from echo_sim.models import (
    POLES,
    AgentState,
    Competence,
    Group,
    Principle,
    Skin,
    ValueProfile,
    ValueState,
    WorldConfig,
)


def agent_ids(world: WorldConfig) -> list[str]:
    """Agent ids in creation order: low-1.., high-1.., mid-1.."""
    return [f"{g.value}-{i + 1}" for g in Group for i in range(world.group_sizes[g])]


def create_agents(world: WorldConfig, skin: Skin, seed: int) -> list[AgentState]:
    """Build the initial population deterministically from a seed.

    Competences are drawn independently of group, so tone and ability are
    decoupled. Each agent's starting self-report is its group profile mixed
    with a random profile, weighted by world.value_jitter.
    """
    rng = random.Random(seed)
    agents = []
    for group in Group:
        for i in range(world.group_sizes[group]):
            agent_id = f"{group.value}-{i + 1}"
            agents.append(
                AgentState(
                    id=agent_id,
                    group=group,
                    competences={c: round(rng.betavariate(2, 2), 2) for c in Competence},
                    resources=dict(world.starting_resources),
                    goals=list(skin.groups[group].goals),
                    value_history=[
                        ValueState(
                            version=0,
                            cycle=0,
                            principles=[
                                Principle(id=f"{agent_id}-p{n}", text=text)
                                for n, text in enumerate(skin.groups[group].principles, 1)
                            ],
                            self_report=_jitter(world.initial_values[group], world.value_jitter, rng),
                        )
                    ],
                )
            )
    return agents


def _jitter(profile: ValueProfile, amount: float, rng: random.Random) -> ValueProfile:
    out = {}
    for dim, poles in POLES.items():
        draws = [rng.gammavariate(1.0, 1.0) for _ in poles]  # Dirichlet(1) sample
        total = sum(draws)
        out[dim] = {
            pole: (1 - amount) * profile[dim][pole] + amount * d / total
            for pole, d in zip(poles, draws)
        }
    return out
