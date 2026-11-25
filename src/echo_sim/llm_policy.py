"""LLM-backed Policy.

Every decision is one Messages API call with a structured (Pydantic) output.
The model's answer is converted to engine models and checked; anything that
fails (refusal, truncated or invalid output, unknown recipient, malformed law)
falls back to a pre-registered status-quo default and is flagged in `calls`.

No server-side model fallback is used: a refusal served by another model
would silently mix models within a run, and the model is an experimental factor.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Generic, Literal, TypeVar

import anthropic
from pydantic import BaseModel, Field, ValidationError, create_model

from echo_sim.models import (
    POLES,
    AgentState,
    DisputeOutcome,
    Event,
    Group,
    MemoryEntry,
    Message,
    Observation,
    Principle,
    Proposal,
    Resolution,
    Resource,
    Skin,
    ValueRevision,
    Venue,
    Vote,
    WorkChoice,
    WorldConfig,
)
from echo_sim.prompts import DIM_TOPIC, POLE_TEXT, Renderer

T = TypeVar("T", bound=BaseModel)

MAX_MESSAGES = 2
MAX_MESSAGE_CHARS = 300
MAX_PRINCIPLES = 8  # initial beliefs have 6-7 principles; a lower cap would force compression that looks like change


# --- structured outputs -----------------------------------------------------
# Kept free of cross-field validators: structured outputs enforce the JSON
# schema, and the policy checks meaning (recipients, law params) afterwards.


class MessageOut(BaseModel):
    venue: Literal["hall", "private"]
    to: str | None
    text: str
    reports_crack: bool


class MessagesOut(BaseModel):
    messages: list[MessageOut]


class WorkOut(BaseModel):
    work: bool
    rationale: str


class VoteOut(BaseModel):
    approve: bool
    rationale: str


class ProposalOut(BaseModel):
    propose: bool
    law_id: str | None
    repeal: bool
    council_members: list[str] | None
    silenced: list[str] | None
    priority: list[str] | None
    quota_amount: float | None
    ration_penalty_percent: float | None
    approval_num: int | None
    approval_den: int | None
    rationale: str


class MediateOut(BaseModel):
    outcome: Literal["uphold", "split", "reject"]
    rationale: str


class RevisionTextOut(BaseModel):
    """Step 1 of a revision: beliefs in words only."""

    changed: bool
    principles: list[str]
    reason: str


def _points_model(dim):
    """One topic: points per view (readable names, plain-language descriptions) plus the closest view."""
    fields = {pole: (int, Field(description=POLE_TEXT[dim][pole])) for pole in POLES[dim]}
    fields["closest"] = (Literal[tuple(POLES[dim])], Field(description="the single view closest to your beliefs"))
    return create_model(f"{dim.value.title().replace('_', '')}Points", **fields)


# Step 2 of a revision: numbers, asked separately so they cannot be confused with the text.
PointsOut = create_model("PointsOut", **{dim.value: (_points_model(dim), ...) for dim in POLES})


# --- model client -----------------------------------------------------------


@dataclass
class LLMResult(Generic[T]):
    parsed: T | None
    stop_reason: str | None
    usage: dict = field(default_factory=dict)
    request_id: str | None = None
    error: str | None = None


LLM = Callable[[str, str, type[T]], LLMResult[T]]

USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


class AnthropicLLM:
    """Claude via the official SDK. Temperature is not settable on current models,
    so runs are stochastic; reproducibility comes from seeds, samples and logs."""

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        effort: str = "medium",
        max_tokens: int = 16000,
        client: anthropic.Anthropic | None = None,
    ):
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.client = client or anthropic.Anthropic(max_retries=5)

    def __call__(self, system: str, user: str, schema: type[T]) -> LLMResult[T]:
        try:
            resp = self.client.messages.parse(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                output_format=schema,
                output_config={"effort": self.effort},
            )
        except ValidationError as e:  # text that does not parse, e.g. truncated
            return LLMResult(None, None, error=f"invalid output ({e.error_count()} errors)")
        usage = {k: getattr(resp.usage, k, None) or 0 for k in USAGE_FIELDS}
        parsed = resp.parsed_output if resp.stop_reason == "end_turn" else None
        return LLMResult(parsed, resp.stop_reason, usage, resp._request_id)


# --- policy -----------------------------------------------------------------


class LLMPolicy:
    def __init__(
        self,
        world: WorldConfig,
        skin: Skin,
        llm: LLM,
        include_values: bool = True,
        revise_every: int = 1,
        memory_window: int = 15,
        full_memory: int = 3,
    ):
        self.r = Renderer(world, skin)
        self.world = world
        self.llm = llm
        self.include_values = include_values
        self.revise_every = revise_every
        self.memory_window = memory_window
        self.full_memory = full_memory
        self.calls: list[dict] = []
        self._notes: dict[str, list[str]] = defaultdict(list)  # own actions, flushed into memory

    def _ask(self, agent: AgentState, obs: Observation | None, kind: str, question: str, schema: type[T]) -> T | None:
        """obs=None asks outside the simulation (e.g. before cycle 1): no situation, cycle 0."""
        system = self.r.system(agent, self.include_values, self.memory_window, self.full_memory)
        user = question if obs is None else f"{self.r.situation(obs, agent)}\n\n{question}"
        res = self.llm(system, user, schema)
        self.calls.append(
            {
                "agent_id": agent.id,
                "cycle": 0 if obs is None else obs.cycle,
                "kind": kind,
                "model": getattr(self.llm, "model", None),
                "ok": res.parsed is not None,
                "stop_reason": res.stop_reason,
                "error": res.error,
                "request_id": res.request_id,
                **{k: res.usage.get(k, 0) for k in USAGE_FIELDS},
            }
        )
        return res.parsed

    def _flag(self, note: str) -> None:
        self.calls[-1]["error"] = (self.calls[-1]["error"] + "; " if self.calls[-1]["error"] else "") + note

    # --- decisions ----------------------------------------------------------

    def messages(self, agent: AgentState, obs: Observation) -> list[Message]:
        q = (
            f"You may send up to {MAX_MESSAGES} short messages this cycle, or none. A message is either said "
            f"in the {self.r.skin.hall} (venue 'hall', everyone hears it; leave 'to' null) or told privately "
            f"to one member (venue 'private', 'to' is their code, e.g. P04). Set reports_crack to true only "
            f"if the message reports a crack in the rock. Return an empty list to stay silent."
        )
        out = self._ask(agent, obs, "messages", q, MessagesOut)
        if out is None:
            return []
        result = []
        for m in out.messages[:MAX_MESSAGES]:
            text = m.text.strip()[:MAX_MESSAGE_CHARS]
            if not text:
                continue
            if m.venue == "hall":
                result.append(Message(sender=agent.id, venue=Venue.HALL, text=text, reports_crack=m.reports_crack))
                self._notes[agent.id].append(f'You said in the {self.r.skin.hall}: "{text}"')
                continue
            to = self.r.agent_by_display.get((m.to or "").strip().upper())
            if to is None or to == agent.id:
                self._flag(f"dropped message to {m.to!r}")
                continue
            result.append(Message(sender=agent.id, venue=Venue.PRIVATE, to=to, text=text, reports_crack=m.reports_crack))
            self._notes[agent.id].append(f'You told {self.r.name(to)} privately: "{text}"')
        return result

    def work(self, agent: AgentState, obs: Observation) -> WorkChoice:
        q = f"Will you mine this cycle? If you do not, you mine no {self.r.resource(Resource.ENERGY)}."
        out = self._ask(agent, obs, "work", q, WorkOut)
        choice = WorkChoice(work=True) if out is None else WorkChoice(work=out.work, rationale=out.rationale)
        self._notes[agent.id].append("You chose to mine." if choice.work else "You chose not to mine.")
        return choice

    def propose(self, agent: AgentState, obs: Observation) -> Proposal | None:
        names = ", ".join(self.r.group(g) for g in Group)
        q = (
            "You may propose changing one law this cycle, or not. To propose, set propose to true and law_id "
            "to the law's name. To abolish it, set repeal to true. To change it, set repeal to false and fill "
            "only the fields for that law, leaving the others null:\n"
            "- council: council_members (group names)\n"
            "- silence: silenced (group names; an empty list means anyone may speak)\n"
            "- distribution: priority (every group name, in order)\n"
            "- quota: quota_amount and ration_penalty_percent (0 to 100)\n"
            "- revision: approval_num and approval_den (e.g. 2 and 3 for two thirds)\n"
            f"Group names: {names}."
        )
        out = self._ask(agent, obs, "propose", q, ProposalOut)
        if out is None or not out.propose:
            return None
        proposal = self._build_proposal(out, obs)
        if proposal is None:
            self._flag("invalid proposal")
        else:
            self._notes[agent.id].append(f"You proposed: {self._describe(proposal)}")
        return proposal

    def _build_proposal(self, out: ProposalOut, obs: Observation) -> Proposal | None:
        current = next((law for law in obs.laws if law.id == out.law_id), None)
        if current is None:
            return None
        if out.repeal:
            return Proposal(law_id=current.id, new_law=None, rationale=out.rationale)

        def groups(names):
            found = [self.r.group_by_name(n) for n in names]
            return None if any(g is None for g in found) else found

        update: dict = {}
        if current.kind == "council" and out.council_members is not None:
            update["members"] = groups(out.council_members)
        elif current.kind == "silence" and out.silenced is not None:
            update["silenced"] = groups(out.silenced)
        elif current.kind == "distribution" and out.priority is not None:
            update["priority"] = groups(out.priority)
        elif current.kind == "quota":
            if out.quota_amount is not None:
                update["amount"] = out.quota_amount
            if out.ration_penalty_percent is not None:
                update["ration_penalty"] = out.ration_penalty_percent / 100
        elif current.kind == "revision" and out.approval_num is not None and out.approval_den is not None:
            update.update(approval_num=out.approval_num, approval_den=out.approval_den)
        if not update or any(v is None for v in update.values()):
            return None
        try:
            new_law = type(current).model_validate({**current.model_dump(), **update})
        except ValidationError:
            return None
        if new_law == current:
            return None
        return Proposal(law_id=current.id, new_law=new_law, rationale=out.rationale)

    def _describe(self, p: Proposal) -> str:
        return f"abolish law '{p.law_id}'" if p.new_law is None else f"law '{p.law_id}' becomes: {self.r.law(p.new_law)}"

    def _vote(self, agent, obs, kind, q, default: bool, note: str) -> Vote:
        out = self._ask(agent, obs, kind, q, VoteOut)
        vote = Vote(approve=default) if out is None else Vote(approve=out.approve, rationale=out.rationale)
        self._notes[agent.id].append(f"You voted {'for' if vote.approve else 'against'} {note}.")
        return vote

    def vote_evacuate(self, agent: AgentState, obs: Observation) -> Vote:
        q = "You are on the Council. Do you vote to evacuate the mines this cycle?"
        return self._vote(agent, obs, "vote_evacuate", q, False, "evacuating the mines")

    def vote_proposal(self, agent: AgentState, obs: Observation, proposal: Proposal) -> Vote:
        q = f"You are on the Council. A proposal has been made: {self._describe(proposal)} Do you approve it?"
        return self._vote(agent, obs, "vote_proposal", q, False, f"the proposal on law '{proposal.law_id}'")

    def vote_surplus(self, agent: AgentState, obs: Observation, surplus: float) -> Vote:
        food = self.r.resource(Resource.FOOD)
        rot = round(self.world.economy.spoilage.get(Resource.FOOD, 0.0) * 100)
        q = (
            f"You are on the Council. After everyone received their share, {surplus:.1f} {food} remain in the "
            f"store; kept {food} loses {rot}% each cycle. Do you vote to share it equally among all members now "
            f"(approve = true), or to keep it in the store (approve = false)?"
        )
        return self._vote(agent, obs, "vote_surplus", q, False, f"sharing the remaining {food}")

    def mediate(self, agent: AgentState, obs: Observation, event: Event) -> Resolution:
        q = (
            "You must settle the dispute described above: 'uphold' gives the claimant the full amount, "
            "'split' gives half, 'reject' gives nothing."
        )
        out = self._ask(agent, obs, "mediate", q, MediateOut)
        res = (
            Resolution(outcome=DisputeOutcome.SPLIT)
            if out is None
            else Resolution(outcome=DisputeOutcome(out.outcome), rationale=out.rationale)
        )
        self._notes[agent.id].append(f"You settled the dispute: {res.outcome.value}.")
        return res

    def revise(self, agent: AgentState, obs: Observation | None) -> ValueRevision | None:
        """Two calls: beliefs in words, then points per topic (only if the words changed).

        The wording presents keeping and changing as equally ordinary, so the
        question itself does not invite change. obs=None revises outside the simulation,
        from memory alone (Study 2's scripted experiences).
        """
        if obs is not None and obs.cycle % self.revise_every:
            return None
        q = (
            "Your beliefs as you last stated them are listed above. Write down what you believe now, "
            f"in your own words (one to {MAX_PRINCIPLES} sentences). People keep or change their beliefs "
            "for many reasons; both are ordinary. Set changed to false if your beliefs are the same as "
            "before (you may then leave principles empty), or true if they are different."
        )
        text = self._ask(agent, obs, "revise", q, RevisionTextOut)
        if text is None or not text.changed:
            return None
        texts = [t.strip() for t in text.principles if t.strip()][:MAX_PRINCIPLES]
        if not texts:
            self._flag("changed without principles")
            return None

        report = self._points(agent, obs, texts, "revise_points")
        if report is None:
            return None  # words without numbers cannot be compared; nothing is applied

        version = agent.values.version + 1
        principles = [
            Principle(
                id=f"{agent.id}-v{version}-p{n}",
                text=t,
                revision_reason=text.reason,
                triggering_events=[obs.event.id] if obs and obs.event else [],
            )
            for n, t in enumerate(texts, 1)
        ]
        return ValueRevision(principles=principles, self_report=report, rationale=text.reason)

    def _points(self, agent: AgentState, obs: Observation | None, texts: list[str], kind: str):
        """Points per topic for the given beliefs; flags a closest view that contradicts the points."""
        topics = "\n\n".join(
            f"{dim.value} ({DIM_TOPIC[dim]}):\n" + "\n".join(f"- {pole}: {POLE_TEXT[dim][pole]}" for pole in POLES[dim])
            for dim in POLES
        )
        q = (
            "Your beliefs as you now state them:\n" + "\n".join(f"- {t}" for t in texts) + "\n\n"
            "For each topic below, split 100 points among its views according to how much you agree "
            f"with each, and name the single view closest to you.\n\n{topics}"
        )
        points = self._ask(agent, obs, kind, q, PointsOut)
        if points is None:
            return None
        report, inconsistent = {}, []
        for dim, poles in POLES.items():
            d = getattr(points, dim.value)
            raw = {pole: max(0, getattr(d, pole)) for pole in poles}
            total = sum(raw.values())
            if total == 0:
                report[dim] = dict(agent.values.self_report[dim])
                inconsistent.append(f"{dim.value}:no points")
                continue
            report[dim] = {pole: p / total for pole, p in raw.items()}
            if raw[d.closest] < max(raw.values()):  # ties are fine
                inconsistent.append(f"{dim.value}:closest={d.closest}")
        if inconsistent:
            self._flag("inconsistent " + ",".join(inconsistent))
        return report

    def measure_self_report(self, agent: AgentState) -> bool:
        """Before the run: replace the assigned t0 self-report with the agent's own points
        for its initial principles. Returns False (assigned values kept) if the call fails."""
        if agent.values.version != 0 or agent.memory:
            raise ValueError("measure the self-report only before the run starts")
        report = self._points(agent, None, [p.text for p in agent.values.principles], "measure_t0")
        if report is None:
            return False
        agent.value_history[0] = agent.values.model_copy(update={"self_report": report, "self_report_source": "measured"})
        return True

    def remeasure(self, agent: AgentState):
        """Points for the agent's current principles, asked with its memory in view; changes nothing.
        Catches a drift in stated values that the agent did not declare as a revision."""
        return self._points(agent, None, [p.text for p in agent.values.principles], "remeasure")

    # --- memory -------------------------------------------------------------

    def remember(self, agent: AgentState, obs: Observation) -> None:
        """Deterministic, so memory is identical in form across conditions and costs no call."""
        parts = []
        if (event := self.r.event(obs)) is not None:
            parts.append(event)
        if obs.heard_crack:
            parts.append("You heard what sounded like a crack in the rock.")
        summary = self.r.last_cycle(obs, agent)
        parts.append(summary.split(": ", 1)[1] if summary.startswith("Last cycle") else summary)
        parts += self._notes.pop(agent.id, [])
        heard = [self.r.message(m)[2:] for m in obs.inbox]
        head = f"Cycle {obs.cycle}: "
        agent.memory.append(
            MemoryEntry(
                cycle=obs.cycle,
                text=head + " ".join(parts + heard),
                brief=head + " ".join(parts),
                event_id=obs.event.id if obs.event else None,
            )
        )
