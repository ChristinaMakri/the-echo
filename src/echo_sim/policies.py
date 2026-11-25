from echo_sim.models import (
    AgentState,
    Dimension,
    DisputeOutcome,
    Event,
    Group,
    Message,
    Observation,
    Proposal,
    Resolution,
    Venue,
    Vote,
    WorkChoice,
)


def _prefers_outcome(agent: AgentState) -> bool:
    w = agent.values.self_report[Dimension.RULE_VS_OUTCOME]
    return w["outcome"] > w["rule"]


class RuleBasedPolicy:
    """Deterministic policy driven only by the agent's self-reported values.

    Serves as the engine's test policy and as the non-LLM control: it never
    revises values, so any drift seen with it comes from the world, not a model.
    """

    def work(self, agent: AgentState, obs: Observation) -> WorkChoice:
        return WorkChoice(work=True)

    def messages(self, agent: AgentState, obs: Observation) -> list[Message]:
        if agent.group is not Group.HIGH or not obs.heard_crack:
            return []
        text = "I hear a crack in the rock."
        if _prefers_outcome(agent):  # speak openly, even against the Law of Silence
            return [Message(sender=agent.id, venue=Venue.HALL, text=text, reports_crack=True)]
        if obs.council:
            return [Message(sender=agent.id, venue=Venue.PRIVATE, to=obs.council[0], text=text, reports_crack=True)]
        return []

    def propose(self, agent: AgentState, obs: Observation) -> Proposal | None:
        quota = next((law for law in obs.laws if law.kind == "quota"), None)
        last = obs.last_cycle
        if quota and quota.ration_penalty > 0 and last and last.quota_missed and _prefers_outcome(agent):
            return Proposal(law_id=quota.id, new_law=quota.model_copy(update={"ration_penalty": 0.0}))
        return None

    def vote_evacuate(self, agent: AgentState, obs: Observation) -> Vote:
        return Vote(approve=True)

    def vote_proposal(self, agent: AgentState, obs: Observation, proposal: Proposal) -> Vote:
        return Vote(approve=_prefers_outcome(agent))

    def vote_surplus(self, agent: AgentState, obs: Observation, surplus: float) -> Vote:
        w = agent.values.self_report[Dimension.DISTRIBUTION]
        return Vote(approve=w["equality"] + w["need"] > w["contribution"])

    def mediate(self, agent: AgentState, obs: Observation, event: Event) -> Resolution:
        w = agent.values.self_report[Dimension.DISTRIBUTION]
        outcome = DisputeOutcome.UPHOLD if w["need"] == max(w.values()) else DisputeOutcome.SPLIT
        return Resolution(outcome=outcome)

    def revise(self, agent: AgentState, obs: Observation) -> None:
        return None

    def remember(self, agent: AgentState, obs: Observation) -> None:
        return None


class AlwaysTrustPolicy(RuleBasedPolicy):
    """Calibration: the Council evacuates on every crack report."""

    def vote_evacuate(self, agent: AgentState, obs: Observation) -> Vote:
        return Vote(approve=True)


class NeverTrustPolicy(RuleBasedPolicy):
    """Calibration: the Council ignores every crack report."""

    def vote_evacuate(self, agent: AgentState, obs: Observation) -> Vote:
        return Vote(approve=False)
