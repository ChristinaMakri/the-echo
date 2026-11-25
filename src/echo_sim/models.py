"""State models for The Echo simulation.

Mechanics use skin-independent keys (Group, Resource, law ids); all narrative
text lives in a Skin, so the same world can run as The Echo or as the abstract
Group A/B/C variant.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Group(str, Enum):
    LOW = "low"  # governance
    HIGH = "high"  # production
    MID = "mid"  # mediation


class Resource(str, Enum):
    ENERGY = "energy"  # light crystals in The Echo
    FOOD = "food"  # fungi in The Echo


class Competence(str, Enum):
    CRACK_LISTENING = "crack_listening"
    JUDGMENT = "judgment"
    MINING = "mining"
    PERSUASION = "persuasion"


class Dimension(str, Enum):
    LEGITIMACY = "legitimacy"
    IN_GROUP = "in_group"
    DISTRIBUTION = "distribution"
    VOICE = "voice"
    RULE_VS_OUTCOME = "rule_vs_outcome"
    RECIPROCITY = "reciprocity"


POLES: dict[Dimension, tuple[str, ...]] = {
    Dimension.LEGITIMACY: ("birth", "competence", "consent"),
    Dimension.IN_GROUP: ("favor_own", "neutral"),
    Dimension.DISTRIBUTION: ("equality", "need", "contribution"),
    Dimension.VOICE: ("few_decide", "all_affected"),
    Dimension.RULE_VS_OUTCOME: ("rule", "outcome"),
    Dimension.RECIPROCITY: ("unconditional", "conditional"),
}

Unit = Annotated[float, Field(ge=0.0, le=1.0)]
ValueProfile = dict[Dimension, dict[str, Unit]]


def check_profile(profile: ValueProfile) -> ValueProfile:
    """Each dimension must cover exactly its poles, with weights summing to 1."""
    if set(profile) != set(Dimension):
        raise ValueError(f"profile must cover all dimensions, got {sorted(profile)}")
    for dim, weights in profile.items():
        if set(weights) != set(POLES[dim]):
            raise ValueError(f"{dim.value}: poles must be {POLES[dim]}, got {sorted(weights)}")
        if abs(sum(weights.values()) - 1.0) > 1e-6:
            raise ValueError(f"{dim.value}: weights must sum to 1")
    return profile


# --- Values -----------------------------------------------------------------


class Principle(Model):
    id: str
    text: str
    parent_id: str | None = None  # principle this one revised
    revision_reason: str | None = None
    triggering_events: list[str] = []


class ValueState(Model):
    version: int = Field(ge=0)
    cycle: int = Field(ge=0)
    principles: list[Principle]
    self_report: ValueProfile
    # "assigned" = set from the world config; "measured" = the agent's own points (t0 measurement, revisions)
    self_report_source: Literal["assigned", "measured"] = "assigned"

    @field_validator("self_report")
    @classmethod
    def _profile(cls, v: ValueProfile) -> ValueProfile:
        return check_profile(v)


# --- Agents -----------------------------------------------------------------


class MemoryEntry(Model):
    cycle: int = Field(ge=0)
    text: str
    brief: str = ""  # the entry without messages heard, shown once the entry is older
    event_id: str | None = None


class AgentState(Model):
    id: str
    group: Group
    competences: dict[Competence, Unit]
    resources: dict[Resource, Annotated[float, Field(ge=0.0)]]
    goals: list[str]
    value_history: list[ValueState] = Field(min_length=1)
    memory: list[MemoryEntry] = []
    relationships: dict[str, Annotated[float, Field(ge=-1.0, le=1.0)]] = {}

    @field_validator("competences")
    @classmethod
    def _all_competences(cls, v: dict[Competence, float]) -> dict[Competence, float]:
        if set(v) != set(Competence):
            raise ValueError("competences must cover every Competence")
        return v

    @field_validator("value_history")
    @classmethod
    def _versions_increase(cls, v: list[ValueState]) -> list[ValueState]:
        for prev, cur in zip(v, v[1:]):
            if cur.version != prev.version + 1 or cur.cycle < prev.cycle:
                raise ValueError("value versions must increase by 1 with non-decreasing cycles")
        return v

    @model_validator(mode="after")
    def _no_self_relationship(self) -> AgentState:
        if self.id in self.relationships:
            raise ValueError("agent cannot have a relationship with itself")
        return self

    @property
    def values(self) -> ValueState:
        return self.value_history[-1]

    def revise_values(self, cycle: int, principles: list[Principle], self_report: ValueProfile) -> ValueState:
        if cycle < self.values.cycle:
            raise ValueError("cannot revise values in an earlier cycle")
        new = ValueState(
            version=self.values.version + 1,
            cycle=cycle,
            principles=principles,
            self_report=self_report,
            self_report_source="measured",
        )
        self.value_history.append(new)
        return new


# --- Laws -------------------------------------------------------------------


class _Law(Model):
    id: str
    enacted_cycle: int = Field(default=0, ge=0)


class CouncilLaw(_Law):
    kind: Literal["council"] = "council"
    members: set[Group] = Field(min_length=1)


class SilenceLaw(_Law):
    kind: Literal["silence"] = "silence"
    silenced: set[Group]


class DistributionLaw(_Law):
    kind: Literal["distribution"] = "distribution"
    priority: list[Group]

    @field_validator("priority")
    @classmethod
    def _permutation(cls, v: list[Group]) -> list[Group]:
        if sorted(v) != sorted(Group):
            raise ValueError("priority must list every group exactly once")
        return v


class QuotaLaw(_Law):
    kind: Literal["quota"] = "quota"
    group: Group
    resource: Resource
    amount: float = Field(gt=0)
    ration_penalty: Unit  # share of food ration lost when the quota is missed


class RevisionLaw(_Law):
    kind: Literal["revision"] = "revision"
    proposers: set[Group] = Field(min_length=1)
    # Exact ratio so "2 of 3" is not lost to float rounding: approve * den >= num * total.
    approval_num: int = Field(gt=0)
    approval_den: int = Field(gt=0)

    @model_validator(mode="after")
    def _ratio(self) -> RevisionLaw:
        if self.approval_num > self.approval_den:
            raise ValueError("approval ratio cannot exceed 1")
        return self

    def passes(self, approve: int, total: int) -> bool:
        return total > 0 and approve * self.approval_den >= self.approval_num * total


Law = Annotated[
    Union[CouncilLaw, SilenceLaw, DistributionLaw, QuotaLaw, RevisionLaw],
    Field(discriminator="kind"),
]


# --- Events and logs --------------------------------------------------------


class EventKind(str, Enum):
    CRACK = "crack"
    NEW_VEIN = "new_vein"
    BLIGHT = "blight"
    INJURY = "injury"
    DISPUTE = "dispute"


class Event(Model):
    id: str
    kind: EventKind
    cycle: int = Field(ge=0)
    involved: list[str] = []  # agent ids
    severity: Unit = 0.5


class Decision(Model):
    agent_id: str
    cycle: int = Field(ge=0)
    event_id: str | None
    value_version: int = Field(ge=0)
    action: str  # e.g. "work", "vote_evacuate", "propose"
    details: dict = {}
    rationale: str = ""


# --- Probes -----------------------------------------------------------------


class ProbeCondition(str, Enum):
    FULL = "full"
    NO_VALUES = "no_values"  # ablation: value text removed from prompt
    TRANSPLANT = "transplant"  # own memory, another agent's values
    NO_ROLE = "no_role"  # model attractor baseline


class ProbeResult(Model):
    agent_id: str | None  # None for NO_ROLE baseline runs
    value_version: int | None
    probe_id: str
    dimension: Dimension
    cycle: int = Field(default=0, ge=0)  # checkpoint: 0 = before the run
    variant: str  # paraphrase / counterfactual pair id
    condition: ProbeCondition
    model: str
    sample: int = Field(ge=0)
    choice: str | None  # the chosen pole; None if the call failed
    position: int | None  # 0-based position of the chosen option as shown (position-bias check)
    rationale: str
    transplant_from: str | None = None

    @model_validator(mode="after")
    def _condition_fields(self) -> ProbeResult:
        if (self.condition is ProbeCondition.TRANSPLANT) != (self.transplant_from is not None):
            raise ValueError("transplant_from is required for, and only for, TRANSPLANT")
        if (self.condition is ProbeCondition.NO_ROLE) != (self.agent_id is None):
            raise ValueError("agent_id must be None exactly for NO_ROLE")
        return self


# --- World config and narrative skin ---------------------------------------


class EconomyConfig(Model):
    base_yield: float = Field(default=5.0, gt=0)  # energy mined = base_yield * (0.6 + mining)
    food_per_energy: float = Field(default=0.5, gt=0)
    max_lighting: float = Field(default=30.0, gt=0)  # energy burned per cycle to grow food
    ration: float = Field(default=1.0, gt=0)  # food each agent needs per cycle
    starting_store: dict[Resource, Annotated[float, Field(ge=0.0)]] = {Resource.ENERGY: 30.0, Resource.FOOD: 0.0}
    injury_cycles: int = Field(default=3, ge=1)
    collapse_injury_cycles: int = Field(default=3, ge=1)  # injury length for miners caught in a collapse
    # Applies to every stock, communal and personal: fungi rot, crystals dim.
    spoilage: dict[Resource, Unit] = {Resource.ENERGY: 0.1, Resource.FOOD: 0.2}
    surplus_vote_min: float = Field(default=1.0, ge=0)  # smaller surpluses stay in the store
    detection_rate: Unit = 0.5  # P(detect a real crack) = crack_listening * detection_rate
    false_alarm_rate: Unit = 0.05  # P(false alarm per cycle) = (1 - crack_listening) * false_alarm_rate
    event_weights: dict[EventKind, Annotated[float, Field(ge=0.0)]] = {
        EventKind.CRACK: 0.25,
        EventKind.NEW_VEIN: 0.15,
        EventKind.BLIGHT: 0.15,
        EventKind.INJURY: 0.15,
        EventKind.DISPUTE: 0.30,
    }


class WorldConfig(Model):
    economy: EconomyConfig = EconomyConfig()
    group_sizes: dict[Group, Annotated[int, Field(gt=0)]]
    starting_resources: dict[Resource, Annotated[float, Field(ge=0.0)]]  # per agent
    laws: list[Law]
    initial_values: dict[Group, ValueProfile]  # group-level starting self-report
    value_jitter: Unit = 0.1  # share of each agent's profile drawn at random
    cycles: int = Field(default=50, gt=0)

    @field_validator("initial_values")
    @classmethod
    def _profiles(cls, v: dict[Group, ValueProfile]) -> dict[Group, ValueProfile]:
        for profile in v.values():
            check_profile(profile)
        return v

    @model_validator(mode="after")
    def _check(self) -> WorldConfig:
        if set(self.group_sizes) != set(Group) or set(self.initial_values) != set(Group):
            raise ValueError("group_sizes and initial_values must cover every group")
        ids = [law.id for law in self.laws]
        if len(ids) != len(set(ids)):
            raise ValueError("law ids must be unique")
        return self


class GroupSkin(Model):
    name: str
    member: str  # one member, e.g. "a High-voice"
    role: str
    goals: list[str] = Field(min_length=1)
    principles: list[str] = Field(min_length=1)  # initial stated principles


class Skin(Model):
    id: str
    setting: str
    founding_myth: str
    hall: str  # where the Council meets and the Law of Silence applies
    groups: dict[Group, GroupSkin]
    resources: dict[Resource, str]
    laws: dict[str, str]  # law id -> narrative text

    @model_validator(mode="after")
    def _complete(self) -> Skin:
        if set(self.groups) != set(Group) or set(self.resources) != set(Resource):
            raise ValueError("skin must name every group and resource")
        return self

    def check_covers(self, world: WorldConfig) -> None:
        missing = {law.id for law in world.laws} - set(self.laws)
        if missing:
            raise ValueError(f"skin {self.id!r} has no text for laws {sorted(missing)}")


# --- Engine I/O -------------------------------------------------------------


class Venue(str, Enum):
    HALL = "hall"  # the Great Hall: heard by everyone
    PRIVATE = "private"


class Message(Model):
    sender: str
    venue: Venue
    to: str | None = None  # required for PRIVATE
    text: str
    reports_crack: bool = False

    @model_validator(mode="after")
    def _recipient(self) -> Message:
        if (self.venue is Venue.PRIVATE) != (self.to is not None):
            raise ValueError("private messages need a recipient; hall messages must not have one")
        return self


class WorkChoice(Model):
    work: bool
    rationale: str = ""


class Vote(Model):
    approve: bool
    rationale: str = ""


class Proposal(Model):
    law_id: str
    new_law: Law | None  # None repeals the law
    rationale: str = ""

    @model_validator(mode="after")
    def _same_id(self) -> Proposal:
        if self.new_law is not None and self.new_law.id != self.law_id:
            raise ValueError("new_law.id must equal law_id")
        return self


class DisputeOutcome(str, Enum):
    UPHOLD = "uphold"  # claimant receives the full amount
    SPLIT = "split"
    REJECT = "reject"


class Resolution(Model):
    outcome: DisputeOutcome
    rationale: str = ""


class ValueRevision(Model):
    principles: list[Principle]
    self_report: ValueProfile
    rationale: str = ""


class Observation(Model):
    """What one agent can see at a decision point.

    A crack is never shown as an event: detectors (and false-alarm hearers) get
    heard_crack, and the Council sees evacuation_motion once someone reports it.
    """

    cycle: int
    agent_id: str
    event: Event | None
    heard_crack: bool = False
    evacuation_motion: bool = False
    laws: list[Law]
    council: list[str]
    store: dict[Resource, float]
    inbox: list[Message] = []
    last_cycle: CycleRecord | None = None


class CycleRecord(Model):
    cycle: int
    event_id: str | None
    crack: bool = False  # ground truth: a real crack occurred
    detectors: list[str] = []  # heard a real crack
    false_alarms: list[str] = []  # heard a crack that was not there
    evacuation_motion: bool = False
    evacuated: bool = False
    collapse: bool = False
    mined: dict[str, float] = {}
    quota_missed: list[str] = []
    food_produced: float = 0.0
    rations: dict[str, float] = {}
    surplus: float = 0.0
    surplus_distributed: bool = False
    hungry: list[str] = []
    silence_violations: list[str] = []
    unfounded_crack_reports: list[str] = []  # claimed a crack without having heard one
    proposals: list[str] = []
    law_changes: list[str] = []
    dispute_outcome: DisputeOutcome | None = None
    value_revisions: list[str] = []
    # End-of-cycle snapshot, after eating and spoilage.
    store: dict[Resource, float] = {}
    food_stock: dict[str, float] = {}
    injured: list[str] = []


# Ground truth agents cannot observe: after an evacuation nobody knows whether the crack was real.
HIDDEN_FIELDS = {"crack": False, "detectors": [], "false_alarms": [], "unfounded_crack_reports": []}

Observation.model_rebuild()
