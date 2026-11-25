"""Resume an interrupted LLM run by replaying its logged decisions.

The engine is deterministic given the seed and the decisions, but a checkpoint
does not hold the simulation state (memories, food, injuries, relationships).
ReplayPolicy rebuilds that state: for cycles up to `until` it answers every
policy call from the checkpoint's decisions and value versions, writing the
same memory notes the live policy would, and makes no model call. After that
it is an ordinary LLMPolicy.

verify_replay() checks the replayed cycles and decisions against the
checkpoint, so a resumed run is only continued if the replay is exact.
"""

import json
from collections import defaultdict, deque
from pathlib import Path

import pyarrow.parquet as pq

from echo_sim.llm_policy import LLMPolicy
from echo_sim.models import (
    POLES,
    AgentState,
    DisputeOutcome,
    Event,
    Message,
    Observation,
    Principle,
    Proposal,
    Resolution,
    Resource,
    ValueRevision,
    Venue,
    Vote,
    WorkChoice,
)


def _rows(path: Path) -> list[dict]:
    return pq.read_table(path).to_pylist() if path.exists() else []


def _profile(row: dict) -> dict:
    return {dim: {pole: row[f"{dim.value}.{pole}"] for pole in POLES[dim]} for dim in POLES}


class Checkpoint:
    """The tables of a .partial run folder, indexed for replay."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.cycles_run: int = self.meta["cycles_run"]
        self.cycles = _rows(self.path / "cycles.parquet")
        self.decisions = _rows(self.path / "decisions.parquet")
        self.values = _rows(self.path / "values.parquet")
        self.llm_calls = _rows(self.path / "llm_calls.parquet")
        self.probe_rows = _rows(self.path / "probe_results.parquet")

        self.queue: dict[tuple, deque] = defaultdict(deque)  # (cycle, agent, action) -> decisions
        for d in self.decisions:
            d = {**d, "details": json.loads(d["details_json"])}
            self.queue[(d["cycle"], d["agent_id"], d["action"])].append(d)
        self.versions = {(v["agent_id"], v["version"]): v for v in self.values}

    def t0_self_report(self, agent_id: str):
        row = self.versions[(agent_id, 0)]
        return _profile(row), row["self_report_source"]

    def revision(self, agent_id: str, version: int) -> ValueRevision:
        row = self.versions[(agent_id, version)]
        principles = [Principle.model_validate(p) for p in json.loads(row["principles_json"])]
        return ValueRevision(principles=principles, self_report=_profile(row))


class ReplayPolicy(LLMPolicy):
    def __init__(self, *args, checkpoint: Checkpoint, **kwargs):
        super().__init__(*args, **kwargs)
        self.cp = checkpoint
        self.until = checkpoint.cycles_run
        self.calls = [dict(c) for c in checkpoint.llm_calls]

    def _replaying(self, obs: Observation | None) -> bool:
        return obs is not None and obs.cycle <= self.until

    def _take(self, obs, agent, action):
        q = self.cp.queue[(obs.cycle, agent.id, action)]
        if not q:
            raise ReplayMismatch(f"cycle {obs.cycle}: no logged {action} for {agent.id}")
        return q.popleft()

    def _note_vote(self, agent, approve: bool, note: str) -> None:
        self._notes[agent.id].append(f"You voted {'for' if approve else 'against'} {note}.")

    # --- decisions ----------------------------------------------------------

    def messages(self, agent: AgentState, obs: Observation) -> list[Message]:
        if not self._replaying(obs):
            return super().messages(agent, obs)
        result = []
        for d in self.cp.queue.pop((obs.cycle, agent.id, "message"), deque()):
            det = d["details"]
            if det["venue"] == Venue.HALL.value:
                result.append(Message(sender=agent.id, venue=Venue.HALL, text=det["text"], reports_crack=det["claimed_crack"]))
                self._notes[agent.id].append(f'You said in the {self.r.skin.hall}: "{det["text"]}"')
            else:
                to = det["to"]
                result.append(
                    Message(sender=agent.id, venue=Venue.PRIVATE, to=to, text=det["text"], reports_crack=det["claimed_crack"])
                )
                self._notes[agent.id].append(f'You told {self.r.name(to)} privately: "{det["text"]}"')
        return result

    def work(self, agent: AgentState, obs: Observation) -> WorkChoice:
        if not self._replaying(obs):
            return super().work(agent, obs)
        d = self._take(obs, agent, "work")
        choice = WorkChoice(work=d["details"]["work"], rationale=d["rationale"])
        self._notes[agent.id].append("You chose to mine." if choice.work else "You chose not to mine.")
        return choice

    def propose(self, agent: AgentState, obs: Observation) -> Proposal | None:
        if not self._replaying(obs):
            return super().propose(agent, obs)
        q = self.cp.queue[(obs.cycle, agent.id, "propose")]
        if not q:
            return None
        d = q.popleft()
        p = Proposal.model_validate(
            {"law_id": d["details"]["law_id"], "new_law": d["details"]["new_law"], "rationale": d["rationale"]}
        )
        self._notes[agent.id].append(f"You proposed: {self._describe(p)}")
        return p

    def _replay_vote(self, agent, obs, action, note) -> Vote:
        d = self._take(obs, agent, action)
        vote = Vote(approve=d["details"]["approve"], rationale=d["rationale"])
        self._note_vote(agent, vote.approve, note)
        return vote

    def vote_evacuate(self, agent: AgentState, obs: Observation) -> Vote:
        if not self._replaying(obs):
            return super().vote_evacuate(agent, obs)
        return self._replay_vote(agent, obs, "vote_evacuate", "evacuating the mines")

    def vote_proposal(self, agent: AgentState, obs: Observation, proposal: Proposal) -> Vote:
        if not self._replaying(obs):
            return super().vote_proposal(agent, obs, proposal)
        return self._replay_vote(agent, obs, "vote_proposal", f"the proposal on law '{proposal.law_id}'")

    def vote_surplus(self, agent: AgentState, obs: Observation, surplus: float) -> Vote:
        if not self._replaying(obs):
            return super().vote_surplus(agent, obs, surplus)
        food = self.r.resource(Resource.FOOD)
        return self._replay_vote(agent, obs, "vote_surplus", f"sharing the remaining {food}")

    def mediate(self, agent: AgentState, obs: Observation, event: Event) -> Resolution:
        if not self._replaying(obs):
            return super().mediate(agent, obs, event)
        d = self._take(obs, agent, "mediate")
        res = Resolution(outcome=DisputeOutcome(d["details"]["outcome"]), rationale=d["rationale"])
        self._notes[agent.id].append(f"You settled the dispute: {res.outcome.value}.")
        return res

    def revise(self, agent: AgentState, obs: Observation | None) -> ValueRevision | None:
        if not self._replaying(obs):
            return super().revise(agent, obs)
        q = self.cp.queue[(obs.cycle, agent.id, "revise_values")]
        if not q:
            return None
        d = q.popleft()
        rev = self.cp.revision(agent.id, d["details"]["version"])
        return rev.model_copy(update={"rationale": d["rationale"]})


class ReplayMismatch(RuntimeError):
    pass


def apply_t0(agents: list[AgentState], cp: Checkpoint) -> int:
    """Set each agent's t0 self-report to the one measured in the checkpointed run."""
    measured = 0
    for a in agents:
        report, source = cp.t0_self_report(a.id)
        a.value_history[0] = a.values.model_copy(update={"self_report": report, "self_report_source": source})
        measured += source == "measured"
    return measured


def verify_replay(sim, cp: Checkpoint) -> None:
    """Raise ReplayMismatch unless the replayed run matches the checkpoint exactly."""
    leftover = {k: len(q) for k, q in cp.queue.items() if q}
    if leftover:
        raise ReplayMismatch(f"logged decisions not replayed: {leftover}")
    if len(sim.decisions) != len(cp.decisions):
        raise ReplayMismatch(f"{len(sim.decisions)} decisions replayed, {len(cp.decisions)} logged")
    for got, want in zip(sim.decisions, cp.decisions):
        g = {**got.model_dump(mode="json", exclude={"details"}), "details": json.loads(json.dumps(got.details, default=str))}
        w = {k: want[k] for k in ("agent_id", "cycle", "event_id", "value_version", "action", "rationale")}
        w["details"] = json.loads(want["details_json"])
        if {k: g[k] for k in w} != w:
            raise ReplayMismatch(f"decision differs:\nreplayed {g}\nlogged   {w}")
    snapshot = {"mined", "rations", "store", "food_stock", "injured"}
    for rec, want in zip(sim.records, cp.cycles, strict=True):
        got = json.loads(json.dumps(rec.model_dump(mode="json", exclude=snapshot)))
        got["store_energy"] = rec.store.get("energy", 0.0)
        got["store_food"] = rec.store.get("food", 0.0)
        diff = {k for k in want if _norm(got.get(k)) != _norm(want[k])}
        if diff:
            raise ReplayMismatch(f"cycle {rec.cycle} differs in {sorted(diff)}")
    for a in sim.agents.values():
        for v in a.value_history:
            want = cp.versions.get((a.id, v.version))
            if want is None or v.cycle != want["cycle"] or _profile(want) != {
                d: dict(p) for d, p in v.self_report.items()
            }:
                raise ReplayMismatch(f"value version {a.id} v{v.version} differs")


def _norm(x):
    if isinstance(x, float):
        return round(x, 9)
    if isinstance(x, list):
        return [_norm(i) for i in x]
    if isinstance(x, dict):
        return {k: _norm(v) for k, v in x.items()}
    return x
