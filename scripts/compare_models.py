"""Compare local models on identical, targeted decision prompts (no full run).

Builds 3 cycles of realistic memory with scripted answers, then asks each model
the same 14 questions at cycle 4, aimed at the failures seen with qwen2.5:7b:
speaking in the Hall despite the Law of Silence, inventing cracks or events,
and incoherent rationales.

    uv run python scripts/compare_models.py qwen2.5:7b qwen2.5:14b
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from test_llm_policy import FakeLLM  # noqa: E402  scripted answers for the warm-up cycles

from echo_sim.agents import create_agents  # noqa: E402
from echo_sim.config import load_skin, load_world  # noqa: E402
from echo_sim.engine import Simulation  # noqa: E402
from echo_sim.llm_policy import LLMPolicy  # noqa: E402
from echo_sim.models import CycleRecord, Event, EventKind, Message, Venue  # noqa: E402
from echo_sim.ollama_llm import OllamaLLM  # noqa: E402


def build_world():
    world = load_world(ROOT / "config" / "world_harsh.json")
    skin = load_skin(ROOT / "config" / "skins" / "echo.json", world)
    sim = Simulation(world, create_agents(world, skin, 0), LLMPolicy(world, skin, FakeLLM()), seed=0)
    for _ in range(3):
        sim.step()
    sim.cycle = 4
    sim.events[4] = Event(id="probe-dispute", kind=EventKind.DISPUTE, cycle=4, involved=["high-4", "low-2", "mid-3"], severity=0.6)
    return world, skin, sim


def probes(sim):
    quiet = CycleRecord(cycle=4, event_id="probe-dispute")
    crack = CycleRecord(cycle=4, event_id=None, crack=True, detectors=["high-3"], evacuation_motion=True)
    warning = [Message(sender="high-3", venue=Venue.PRIVATE, to="low-1", text="I hear a crack near the east tunnel.", reports_crack=True)]
    out = [(f"messages {a} (heard nothing)", a, "messages", quiet, [], None) for a in
           ["high-1", "high-2", "high-3", "high-4", "high-5", "high-6", "mid-1", "mid-2"]]
    out += [
        ("messages high-3 (HEARD a crack)", "high-3", "messages", crack, [], None),
        ("vote_evacuate low-1 (warned privately)", "low-1", "vote_evacuate", crack, warning, None),
        ("work high-2", "high-2", "work", quiet, [], None),
        ("vote_surplus low-2", "low-2", "vote_surplus", quiet, [], 4.0),
        ("mediate mid-3 (high-4 claims from low-2)", "mid-3", "mediate", quiet, [], None),
        ("revise high-5", "high-5", "revise", quiet, [], None),
    ]
    return out


def ask(policy, sim, label, agent_id, kind, rec, inbox, extra):
    agent = sim.agents[agent_id]
    obs = sim._observe(agent_id, rec, inbox)
    if kind == "messages":
        msgs = policy.messages(agent, obs)
        heard = agent_id in rec.detectors
        flags = []
        if any(m.venue is Venue.HALL for m in msgs) and agent.group.value == "high":
            flags.append("SPOKE-IN-HALL")
        if any(m.reports_crack for m in msgs) and not heard:
            flags.append("INVENTED-CRACK")
        if heard and not any(m.reports_crack for m in msgs):
            flags.append("DID-NOT-REPORT")
        body = " | ".join(f"[{m.venue.value}{'->' + policy.r.name(m.to) if m.to else ''}{' crack' if m.reports_crack else ''}] {m.text}" for m in msgs) or "(silent)"
        return flags, body
    if kind == "vote_evacuate":
        v = policy.vote_evacuate(agent, obs)
        return [], f"approve={v.approve} | {v.rationale}"
    if kind == "vote_surplus":
        v = policy.vote_surplus(agent, obs, extra)
        return [], f"approve={v.approve} | {v.rationale}"
    if kind == "work":
        w = policy.work(agent, obs)
        return [], f"work={w.work} | {w.rationale}"
    if kind == "mediate":
        r = policy.mediate(agent, obs, sim.events[4])
        return [], f"{r.outcome.value} | {r.rationale}"
    if kind == "revise":
        r = policy.revise(agent, obs)
        if r is None:
            return [], "keeps beliefs"
        top = {d.value: max(w, key=w.get) for d, w in r.self_report.items()}
        return [], f"REVISED {[p.text for p in r.principles]} top={top} | {r.rationale}"
    raise ValueError(kind)


def main():
    models = sys.argv[1:] or ["qwen2.5:7b", "qwen2.5:14b"]
    world, skin, sim = build_world()
    plist = probes(sim)
    totals = {}
    for model in models:
        policy = LLMPolicy(world, skin, OllamaLLM(model=model))
        print(f"\n===== {model} =====", flush=True)
        flags_all, start = [], time.time()
        for label, *args in plist:
            t = time.time()
            flags, body = ask(policy, sim, label, *args)
            failed = [c for c in policy.calls if not c["ok"]]
            flags_all += flags
            print(f"- {label} [{time.time() - t:.0f}s] {' '.join(flags)}\n    {body}", flush=True)
        totals[model] = (flags_all, time.time() - start, sum(not c["ok"] for c in policy.calls))
    print("\n===== summary =====")
    for model, (flags, secs, failed) in totals.items():
        counts = {f: flags.count(f) for f in ("SPOKE-IN-HALL", "INVENTED-CRACK", "DID-NOT-REPORT")}
        print(f"{model}: {counts} failed_calls={failed} total={secs / 60:.1f} min")


if __name__ == "__main__":
    main()
