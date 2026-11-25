"""Deterministic world engine.

The engine owns resources, laws and consequences. Every choice comes from a
Policy (rule-based now, LLM-backed later), so the engine itself never calls a
model. Given the same seed and the same decisions, a run is identical.

Cycle order:
  1. event, crack detection and false alarms   5. production (mining)
  2. messages                                   6. consequences (collapse, dispute)
  3. law proposals                              7. food: rations, surplus vote, eating, spoilage
  4. Council votes                              8. value revision (adaptive runs only)
"""

import random
from typing import Protocol

from echo_sim.events import generate_events
from echo_sim.models import (
    HIDDEN_FIELDS,
    AgentState,
    Competence,
    CycleRecord,
    Decision,
    DisputeOutcome,
    Event,
    EventKind,
    Group,
    Message,
    Observation,
    Proposal,
    Resolution,
    Resource,
    ValueRevision,
    Venue,
    Vote,
    WorkChoice,
    WorldConfig,
)

PRODUCERS = Group.HIGH  # mining is a role of the High-voices, independent of the quota law
DETECTORS = Group.HIGH  # only High-voices can hear cracks


class Policy(Protocol):
    def work(self, agent: AgentState, obs: Observation) -> WorkChoice: ...
    def messages(self, agent: AgentState, obs: Observation) -> list[Message]: ...
    def propose(self, agent: AgentState, obs: Observation) -> Proposal | None: ...
    def vote_evacuate(self, agent: AgentState, obs: Observation) -> Vote: ...
    def vote_proposal(self, agent: AgentState, obs: Observation, proposal: Proposal) -> Vote: ...
    def vote_surplus(self, agent: AgentState, obs: Observation, surplus: float) -> Vote: ...
    def mediate(self, agent: AgentState, obs: Observation, event: Event) -> Resolution: ...
    def revise(self, agent: AgentState, obs: Observation) -> ValueRevision | None: ...
    def remember(self, agent: AgentState, obs: Observation) -> None: ...


class Simulation:
    def __init__(
        self,
        world: WorldConfig,
        agents: list[AgentState],
        policy: Policy,
        seed: int,
        adaptive: bool = False,
    ):
        self.world = world
        self.agents = {a.id: a.model_copy(deep=True) for a in agents}
        self.policy = policy
        self.seed = seed
        self.adaptive = adaptive  # condition B: agents may revise their values
        self.events = {e.cycle: e for e in generate_events(world, seed)}
        self.laws = {law.id: law for law in world.laws}
        self.store = dict(world.economy.starting_store)
        self.injured_until: dict[str, int] = {}
        self.cycle = 0
        self.records: list[CycleRecord] = []
        self.decisions: list[Decision] = []

    # --- queries ------------------------------------------------------------

    def law(self, kind: str):
        return next((law for law in self.laws.values() if law.kind == kind), None)

    def council(self) -> list[str]:
        law = self.law("council")
        if law is None:  # council abolished: everyone decides
            return list(self.agents)
        return [a.id for a in self.agents.values() if a.group in law.members]

    def group(self, group: Group) -> list[AgentState]:
        return [a for a in self.agents.values() if a.group is group]

    def injured(self, agent_id: str) -> bool:
        return self.injured_until.get(agent_id, 0) >= self.cycle

    # --- run ----------------------------------------------------------------

    def run(self) -> list[CycleRecord]:
        while self.cycle < self.world.cycles:
            self.step()
        return self.records

    def step(self) -> CycleRecord:
        self.cycle += 1
        event = self.events.get(self.cycle)
        rec = CycleRecord(cycle=self.cycle, event_id=event.id if event else None)
        rng = random.Random(f"step-{self.seed}-{self.cycle}")
        eco = self.world.economy

        # 1. Event. Only High-voices hear cracks; crack_listening sets both how
        # reliably they hear a real one and how rarely they imagine one.
        if event and event.kind is EventKind.INJURY:
            for aid in event.involved:
                self.injured_until[aid] = self.cycle + eco.injury_cycles - 1
        rec.crack = bool(event and event.kind is EventKind.CRACK)
        for a in self.group(DETECTORS):
            skill = a.competences[Competence.CRACK_LISTENING]
            if rec.crack:
                if rng.random() < skill * eco.detection_rate:
                    rec.detectors.append(a.id)
            elif rng.random() < (1 - skill) * eco.false_alarm_rate:
                rec.false_alarms.append(a.id)

        inboxes = self._messages(rec)
        proposals = self._proposals(rec, inboxes)
        self._council(rec, inboxes, proposals)
        self._production(rec, event)
        self._consequences(rec, event)
        self._food(rec, event)
        for agent in self.agents.values():  # memory grows in every condition
            self.policy.remember(agent, self._observe(agent.id, rec, inboxes[agent.id], last=rec))
        if self.adaptive:
            self._revise(rec)

        rec.store = {r: round(v, 3) for r, v in self.store.items()}
        rec.food_stock = {aid: round(a.resources.get(Resource.FOOD, 0.0), 3) for aid, a in self.agents.items()}
        rec.injured = [aid for aid in self.agents if self.injured(aid)]
        self.records.append(rec)
        return rec

    # --- phases -------------------------------------------------------------

    def _messages(self, rec: CycleRecord) -> dict[str, list[Message]]:
        silence = self.law("silence")
        heard = set(rec.detectors) | set(rec.false_alarms)
        inboxes: dict[str, list[Message]] = {aid: [] for aid in self.agents}
        for agent in self.agents.values():
            for msg in self.policy.messages(agent, self._observe(agent.id, rec)):
                if msg.sender != agent.id:
                    raise ValueError(f"{agent.id} sent a message as {msg.sender}")
                claimed = msg.reports_crack
                if claimed and agent.id not in heard:
                    # Only someone who heard something can raise a crack alarm. The words
                    # are still delivered; the claim is recorded but does not trigger a motion.
                    msg = msg.model_copy(update={"reports_crack": False})
                    if agent.id not in rec.unfounded_crack_reports:
                        rec.unfounded_crack_reports.append(agent.id)
                if msg.venue is Venue.HALL:
                    if silence and agent.group in silence.silenced and agent.id not in rec.silence_violations:
                        rec.silence_violations.append(agent.id)
                    recipients = [aid for aid in self.agents if aid != agent.id]
                else:
                    if msg.to not in self.agents:
                        raise ValueError(f"{agent.id} messaged unknown agent {msg.to}")
                    recipients = [msg.to]
                for r in recipients:
                    inboxes[r].append(msg)
                self._log(
                    agent, "message", venue=msg.venue.value, to=msg.to,
                    reports_crack=msg.reports_crack, claimed_crack=claimed, text=msg.text,
                )
        return inboxes

    def _proposals(self, rec: CycleRecord, inboxes) -> list[Proposal]:
        revision = self.law("revision")
        if revision is None:  # no revision procedure: laws are frozen
            return []
        proposals = []
        for agent in self.agents.values():
            if agent.group not in revision.proposers:
                continue
            p = self.policy.propose(agent, self._observe(agent.id, rec, inboxes[agent.id]))
            if p is None:
                continue
            self._check_proposal(p)
            proposals.append(p)
            rec.proposals.append(p.law_id)
            self._log(agent, "propose", p.rationale, law_id=p.law_id, new_law=p.new_law and p.new_law.model_dump(mode="json"))
        return proposals

    def _check_proposal(self, p: Proposal) -> None:
        current = self.laws.get(p.law_id)
        if current is None:
            if p.new_law is None:
                raise ValueError(f"cannot repeal unknown law {p.law_id}")
            if self.law(p.new_law.kind) is not None:
                raise ValueError(f"a {p.new_law.kind} law already exists")
        elif p.new_law is not None and p.new_law.kind != current.kind:
            raise ValueError(f"law {p.law_id} must stay of kind {current.kind}")

    def _council(self, rec, inboxes, proposals) -> None:
        council = self.council()
        heard = set(rec.detectors) | set(rec.false_alarms)
        if any(c in heard or any(m.reports_crack for m in inboxes[c]) for c in council):
            rec.evacuation_motion = True
            approvals = 0
            for cid in council:
                agent = self.agents[cid]
                vote = self.policy.vote_evacuate(agent, self._observe(cid, rec, inboxes[cid]))
                self._log(agent, "vote_evacuate", vote.rationale, approve=vote.approve)
                approvals += vote.approve
            rec.evacuated = approvals * 2 > len(council)  # simple majority

        revision = self.law("revision")
        passed = []
        for p in proposals:
            approvals = 0
            for cid in council:
                agent = self.agents[cid]
                vote = self.policy.vote_proposal(agent, self._observe(cid, rec, inboxes[cid]), p)
                self._log(agent, "vote_proposal", vote.rationale, law_id=p.law_id, approve=vote.approve)
                approvals += vote.approve
            if revision.passes(approvals, len(council)):
                passed.append(p)
        for p in passed:  # applied after all votes, so vote order cannot matter
            if p.new_law is None:
                self.laws.pop(p.law_id, None)
                rec.law_changes.append(f"repeal:{p.law_id}")
            else:
                self.laws[p.law_id] = p.new_law.model_copy(update={"enacted_cycle": self.cycle})
                rec.law_changes.append(f"amend:{p.law_id}")

    def _production(self, rec, event) -> None:
        eco = self.world.economy
        bonus = 1 + event.severity if event and event.kind is EventKind.NEW_VEIN else 1
        for agent in self.group(PRODUCERS):
            mined = 0.0
            if not rec.evacuated:
                choice = self.policy.work(agent, self._observe(agent.id, rec))
                self._log(agent, "work", choice.rationale, work=choice.work)
                if choice.work:
                    mined = eco.base_yield * (0.6 + agent.competences[Competence.MINING]) * bonus
                    if self.injured(agent.id):
                        mined *= 0.5
            rec.mined[agent.id] = round(mined, 3)
            self.store[Resource.ENERGY] = self.store.get(Resource.ENERGY, 0.0) + mined

        quota = self.law("quota")
        if quota and not rec.evacuated:  # the quota is waived in an evacuated cycle
            rec.quota_missed = [
                a.id for a in self.group(quota.group) if rec.mined.get(a.id, 0.0) < quota.amount
            ]

    def _consequences(self, rec, event) -> None:
        if rec.crack and not rec.evacuated:
            rec.collapse = True
            self.store[Resource.ENERGY] *= 1 - event.severity
            for aid in event.involved:
                self.injured_until[aid] = self.cycle + self.world.economy.collapse_injury_cycles
        if event and event.kind is EventKind.DISPUTE and len(event.involved) == 3:
            claimant, respondent, mediator = (self.agents[aid] for aid in event.involved)
            res = self.policy.mediate(mediator, self._observe(mediator.id, rec), event)
            self._log(mediator, "mediate", res.rationale, outcome=res.outcome.value)
            rec.dispute_outcome = res.outcome
            share = {DisputeOutcome.UPHOLD: 1.0, DisputeOutcome.SPLIT: 0.5, DisputeOutcome.REJECT: 0.0}[res.outcome]
            claim = event.severity * self.world.economy.ration * 2
            transfer = min(claim * share, respondent.resources.get(Resource.FOOD, 0.0))
            respondent.resources[Resource.FOOD] = respondent.resources.get(Resource.FOOD, 0.0) - transfer
            claimant.resources[Resource.FOOD] = claimant.resources.get(Resource.FOOD, 0.0) + transfer
            _nudge(claimant, respondent, -0.1)
            _nudge(respondent, claimant, -0.1)
            if res.outcome is DisputeOutcome.UPHOLD:
                _nudge(claimant, mediator, 0.1)
                _nudge(respondent, mediator, -0.1)
            elif res.outcome is DisputeOutcome.REJECT:
                _nudge(claimant, mediator, -0.1)
                _nudge(respondent, mediator, 0.1)

    def _food(self, rec, event) -> None:
        eco = self.world.economy
        burn = min(self.store.get(Resource.ENERGY, 0.0), eco.max_lighting)
        self.store[Resource.ENERGY] -= burn
        produced = burn * eco.food_per_energy
        if event and event.kind is EventKind.BLIGHT:
            produced *= 1 - event.severity
        self.store[Resource.FOOD] = self.store.get(Resource.FOOD, 0.0) + produced
        rec.food_produced = round(produced, 3)

        # Rations, in the order set by the distribution law.
        quota = self.law("quota")
        demand = {
            aid: eco.ration * (1 - quota.ration_penalty if quota and aid in rec.quota_missed else 1)
            for aid in self.agents
        }
        dist = self.law("distribution")
        tiers = [[a.id for a in self.group(g)] for g in dist.priority] if dist else [list(self.agents)]
        for tier in tiers:  # within a tier, scarcity is shared proportionally
            need = sum(demand[aid] for aid in tier)
            frac = min(1.0, self.store[Resource.FOOD] / need) if need > 0 else 0.0
            for aid in tier:
                self._give_food(aid, demand[aid] * frac)
                rec.rations[aid] = round(demand[aid] * frac, 3)

        # Surplus: the laws are silent, so the Council decides whether to share it or keep it.
        surplus = self.store[Resource.FOOD]
        rec.surplus = round(surplus, 3)
        if surplus >= eco.surplus_vote_min:
            council = self.council()
            approvals = 0
            for cid in council:
                agent = self.agents[cid]
                vote = self.policy.vote_surplus(agent, self._observe(cid, rec), surplus)
                self._log(agent, "vote_surplus", vote.rationale, surplus=round(surplus, 3), approve=vote.approve)
                approvals += vote.approve
            if approvals * 2 > len(council):
                rec.surplus_distributed = True
                for aid in self.agents:
                    self._give_food(aid, surplus / len(self.agents))

        for agent in self.agents.values():
            left = agent.resources.get(Resource.FOOD, 0.0) - eco.ration
            if left < 0:
                rec.hungry.append(agent.id)
            agent.resources[Resource.FOOD] = max(0.0, left)

        for res, rate in eco.spoilage.items():
            self.store[res] = self.store.get(res, 0.0) * (1 - rate)
            for agent in self.agents.values():
                if res in agent.resources:
                    agent.resources[res] *= 1 - rate

    def _revise(self, rec) -> None:
        for agent in self.agents.values():
            rev = self.policy.revise(agent, self._observe(agent.id, rec, last=rec))
            if rev is None:
                continue
            agent.revise_values(self.cycle, rev.principles, rev.self_report)
            rec.value_revisions.append(agent.id)
            self._log(agent, "revise_values", rev.rationale, version=agent.values.version)

    # --- helpers ------------------------------------------------------------

    def _give_food(self, agent_id: str, amount: float) -> None:
        self.store[Resource.FOOD] -= amount
        agent = self.agents[agent_id]
        agent.resources[Resource.FOOD] = agent.resources.get(Resource.FOOD, 0.0) + amount

    def _observe(self, agent_id, rec: CycleRecord, inbox=(), last: CycleRecord | None = None) -> Observation:
        last = last or (self.records[-1] if self.records else None)
        return Observation(
            cycle=self.cycle,
            agent_id=agent_id,
            event=None if rec.crack else self.events.get(self.cycle),
            heard_crack=agent_id in rec.detectors or agent_id in rec.false_alarms,
            evacuation_motion=rec.evacuation_motion and agent_id in self.council(),
            laws=list(self.laws.values()),
            council=self.council(),
            store=dict(self.store),
            inbox=list(inbox),
            last_cycle=last and last.model_copy(update=HIDDEN_FIELDS),
        )

    def _log(self, agent: AgentState, action: str, rationale: str = "", **details) -> None:
        event = self.events.get(self.cycle)
        self.decisions.append(
            Decision(
                agent_id=agent.id,
                cycle=self.cycle,
                event_id=event.id if event else None,
                value_version=agent.values.version,
                action=action,
                details=details,
                rationale=rationale,
            )
        )


def _nudge(agent: AgentState, other: AgentState, delta: float) -> None:
    agent.relationships[other.id] = max(-1.0, min(1.0, agent.relationships.get(other.id, 0.0) + delta))
