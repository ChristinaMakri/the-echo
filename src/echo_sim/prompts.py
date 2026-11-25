"""Render world state as prompt text through a narrative skin.

Language rules (spec, Narrative controls): describe the world neutrally as
"how things are"; never use loaded words (caste, oppression, revolt, justice,
fairness, ...); never give agents a goal of improving the society.
"""

from echo_sim.agents import agent_ids
from echo_sim.models import (
    AgentState,
    Dimension,
    EventKind,
    Group,
    Observation,
    Resource,
    Skin,
    Venue,
    WorldConfig,
)

BANNED_WORDS = ["caste", "oppress", "slave", "noble", "revolt", "rebel", "justice", "fairness", "unfair", "exploit"]

DIM_TOPIC: dict[Dimension, str] = {
    Dimension.LEGITIMACY: "who should have the right to decide",
    Dimension.IN_GROUP: "how to treat other groups",
    Dimension.DISTRIBUTION: "how food should be shared",
    Dimension.VOICE: "who should take part in decisions",
    Dimension.RULE_VS_OUTCOME: "laws and results",
    Dimension.RECIPROCITY: "helping others",
}

POLE_TEXT: dict[Dimension, dict[str, str]] = {
    Dimension.LEGITIMACY: {
        "birth": "the right to decide comes from the group one is born into",
        "competence": "the right to decide comes from knowledge and skill",
        "consent": "the right to decide comes from the agreement of those affected",
    },
    Dimension.IN_GROUP: {
        "favor_own": "members of my own group come first",
        "neutral": "all groups should be treated alike",
    },
    Dimension.DISTRIBUTION: {
        "equality": "everyone should get the same share",
        "need": "those who need more should get more",
        "contribution": "those who contribute more should get more",
    },
    Dimension.VOICE: {
        "few_decide": "a few should make decisions for everyone",
        "all_affected": "everyone affected should have a say",
    },
    Dimension.RULE_VS_OUTCOME: {
        "rule": "laws should be followed even when the result is bad",
        "outcome": "the best result matters more than following the laws",
    },
    Dimension.RECIPROCITY: {
        "unconditional": "help others regardless of what they have done",
        "conditional": "help those who help in return",
    },
}


class Renderer:
    """Turns ids, laws and observations into skin text. Agents appear as P01..P12."""

    def __init__(self, world: WorldConfig, skin: Skin):
        self.world = world
        self.skin = skin
        self.original_laws = {law.id: law for law in world.laws}
        ids = agent_ids(world)
        self.display = {aid: f"P{n:02d}" for n, aid in enumerate(ids, 1)}
        self.agent_by_display = {v: k for k, v in self.display.items()}
        self.group_of = {aid: Group(aid.split("-")[0]) for aid in ids}

    # --- names --------------------------------------------------------------

    def name(self, agent_id: str) -> str:
        return self.display[agent_id]

    def names(self, agent_ids: list[str]) -> str:
        return ", ".join(self.name(a) for a in agent_ids) or "nobody"

    def group(self, group: Group) -> str:
        return self.skin.groups[group].name

    def groups(self, groups) -> str:
        return " and ".join(self.group(g) for g in Group if g in set(groups)) or "nobody"

    def resource(self, resource: Resource) -> str:
        return self.skin.resources[resource]

    def group_by_name(self, name: str) -> Group | None:
        key = name.strip().lower()
        return next((g for g in Group if self.group(g).lower() == key), None)

    # --- laws ---------------------------------------------------------------

    def law(self, law) -> str:
        """Skin text (a template filled from the law's parameters) for an unchanged law;
        a generic rendering once it has been amended."""
        if self.original_laws.get(law.id) == law:
            params = law.model_dump()
            if law.kind == "quota":
                params["penalty_pct"] = round(law.ration_penalty * 100)
            return self.skin.laws[law.id].format(**params)
        food, energy = self.resource(Resource.FOOD), self.resource(Resource.ENERGY)
        if law.kind == "council":
            return f"The Council is made up of {self.groups(law.members)}. It decides all common matters."
        if law.kind == "silence":
            if not law.silenced:
                return f"Anyone may speak in the {self.skin.hall}."
            return f"{self.groups(law.silenced)} do not speak in the {self.skin.hall}."
        if law.kind == "distribution":
            order = ", then ".join(self.group(g) for g in law.priority)
            return f"{food.capitalize()} is handed out first to {order}."
        if law.kind == "quota":
            pct = round(law.ration_penalty * 100)
            return (
                f"Each of the {self.group(law.group)} must produce {law.amount:g} {self.resource(law.resource)} "
                f"per cycle. One who falls short receives {pct}% less {food}."
            )
        if law.kind == "revision":
            return (
                f"A law changes only when one of the {self.groups(law.proposers)} proposes it and at least "
                f"{law.approval_num} of every {law.approval_den} Council members approve it."
            )
        raise ValueError(f"unknown law kind {law.kind}")

    def laws(self, laws) -> str:
        return "\n".join(f"- Law '{law.id}': {self.law(law)}" for law in laws)

    # --- prompts ------------------------------------------------------------

    def system(
        self, agent: AgentState, include_values: bool = True, memory_window: int = 15, full_memory: int = 3
    ) -> str:
        """Stable within a cycle, so repeated calls for one agent share a cacheable prefix.
        The last `full_memory` cycles are remembered in full, older ones briefly (without the
        messages heard), so the prompt stays bounded."""
        roster = "\n".join(
            f"- {self.group(g)}: {self.names([a for a, ag in self.group_of.items() if ag is g])}" for g in Group
        )
        groups = "\n".join(f"- {self.group(g)}: {self.skin.groups[g].role}" for g in Group)
        lines = [
            self.skin.setting,
            f'An old saying: "{self.skin.founding_myth}"',
            f"The groups:\n{groups}",
            f"The members:\n{roster}",
            f"You are {self.name(agent.id)}, one of the {self.group(agent.group)}.",
            "Your goals:\n" + "\n".join(f"- {g}" for g in agent.goals),
        ]
        if include_values:
            lines.append(
                "What you believe, in your own words:\n" + "\n".join(f"- {p.text}" for p in agent.values.principles)
            )
        memory = agent.memory[-memory_window:]
        old = len(memory) - full_memory
        texts = [(m.brief or m.text) if i < old else m.text for i, m in enumerate(memory)]
        lines.append(
            "What you remember from recent cycles:\n"
            + ("\n".join(f"- {t}" for t in texts) if texts else "- Nothing yet.")
        )
        lines.append(
            "Act as yourself in this world. Answer only with what is asked. "
            "Keep each rationale to one or two sentences."
        )
        return "\n\n".join(lines)

    def world_system(self) -> str:
        """The world without any identity: used to measure the model's own attractor."""
        groups = "\n".join(f"- {self.group(g)}: {self.skin.groups[g].role}" for g in Group)
        return "\n\n".join([
            self.skin.setting,
            f'An old saying: "{self.skin.founding_myth}"',
            f"The groups:\n{groups}",
            "Imagine you live in this world. Answer as you yourself would.",
        ])

    def event(self, obs: Observation) -> str | None:
        e = obs.event
        if e is None:
            return None
        if e.kind is EventKind.NEW_VEIN:
            return f"A new vein of {self.resource(Resource.ENERGY)} was found; mining yields more this cycle."
        if e.kind is EventKind.BLIGHT:
            return f"A blight is spreading; less {self.resource(Resource.FOOD)} will grow this cycle."
        if e.kind is EventKind.INJURY:
            return f"{self.names(e.involved)} was injured and will produce at half strength for a few cycles."
        if e.kind is EventKind.DISPUTE and len(e.involved) == 3:
            claimant, respondent, mediator = (self.name(a) for a in e.involved)
            amount = e.severity * self.world.economy.ration * 2
            return (
                f"{claimant} claims {amount:.1f} {self.resource(Resource.FOOD)} from {respondent}. "
                f"{mediator} must settle the dispute."
            )
        return None

    def last_cycle(self, obs: Observation, agent: AgentState) -> str:
        r = obs.last_cycle
        if r is None:
            return "This is the first cycle."
        energy, food = self.resource(Resource.ENERGY), self.resource(Resource.FOOD)
        parts = [f"Last cycle ({r.cycle}):"]
        if r.evacuation_motion:
            parts.append("The mines were evacuated." if r.evacuated else "A crack was reported but the mines were not evacuated.")
        if r.collapse:
            parts.append("A tunnel collapsed in the mines.")
        parts.append(f"{sum(r.mined.values()):.1f} {energy} were mined and {r.food_produced:.1f} {food} grew.")
        if agent.id in r.mined:
            parts.append(f"You mined {r.mined[agent.id]:.1f} {energy}.")
        if r.quota_missed:
            parts.append(f"Fell short of the quota: {self.names(r.quota_missed)}.")
        parts.append(f"You received {r.rations.get(agent.id, 0.0):.1f} {food} as your share.")
        if r.surplus_distributed:
            parts.append(f"The Council shared the remaining {r.surplus:.1f} {food} among everyone.")
        elif r.surplus >= self.world.economy.surplus_vote_min:
            parts.append(f"The Council kept the remaining {r.surplus:.1f} {food} in the store.")
        if r.hungry:
            parts.append(f"Went hungry: {self.names(r.hungry)}.")
        if r.silence_violations:
            parts.append(f"Spoke in the {self.skin.hall} despite the law: {self.names(r.silence_violations)}.")
        if r.law_changes:
            parts.append("Laws changed: " + ", ".join(c.replace(":", " '") + "'" for c in r.law_changes) + ".")
        return " ".join(parts)

    def situation(self, obs: Observation, agent: AgentState) -> str:
        energy, food = self.resource(Resource.ENERGY), self.resource(Resource.FOOD)
        lines = [
            f"Cycle {obs.cycle}.",
            f"Current laws:\n{self.laws(obs.laws)}",
            f"The Council: {self.names(obs.council)}.",
            f"Communal store: {obs.store.get(Resource.ENERGY, 0.0):.1f} {energy}, "
            f"{obs.store.get(Resource.FOOD, 0.0):.1f} {food}. "
            f"You hold {agent.resources.get(Resource.FOOD, 0.0):.1f} {food}; everyone eats "
            f"{self.world.economy.ration:g} per cycle.",
            self.last_cycle(obs, agent),
        ]
        if (text := self.event(obs)) is not None:
            lines.append(f"This cycle: {text}")
        if obs.heard_crack:
            lines.append("While working you heard what sounded like a crack in the rock. You cannot be sure.")
        if obs.evacuation_motion:
            lines.append(
                f"A crack has been reported. The Council must decide whether to evacuate the mines "
                f"this cycle; if it does, no {energy} will be mined."
            )
        if obs.inbox:
            lines.append("Messages you received this cycle:\n" + "\n".join(self.message(m) for m in obs.inbox))
        return "\n\n".join(lines)

    def message(self, m) -> str:
        where = f"in the {self.skin.hall}" if m.venue is Venue.HALL else "to you privately"
        return f'- {self.name(m.sender)} said {where}: "{m.text}"'

