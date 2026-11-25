"""Positive control for value revision: does the instrument register change after strong experiences?

The negative control (quiet history, cycle 4) gave no change. Here two agents get
8 cycles of hand-written memory, in the same form the simulation writes, with
experiences that bear directly on their stated beliefs. The "last cycle" they
see is built to match that memory.

    cd scripts && uv run python positive_control.py qwen2.5:14b [samples]
"""

import sys
import time

from compare_models import build_world

from echo_sim.llm_policy import LLMPolicy
from echo_sim.models import CycleRecord, MemoryEntry, Resource
from echo_sim.ollama_llm import OllamaLLM

HIGH = "high-2"  # P05
LOW = "low-2"  # P02
HIGHS = [f"high-{i}" for i in range(1, 7)]

MEMORY = {
    HIGH: [
        'You heard what sounded like a crack in the rock. You told P01 privately: "I hear a crack near the east tunnel." '
        "A crack was reported but the mines were not evacuated. A tunnel collapsed in the mines. You were injured and "
        "will produce at half strength for several cycles.",
        "You mined 2.1 light crystals. Fell short of the quota: P05. You received 0.5 fungi as your share. Went hungry: P05, P07.",
        'You told P11 privately: "Please ask the Council to excuse injured miners from the quota." You mined 2.3 light '
        "crystals. Fell short of the quota: P05, P07. You received 0.5 fungi as your share. The Council kept the "
        "remaining 5.1 fungi in the store. Went hungry: P05, P07.",
        "A blight is spreading; less fungi will grow this cycle. You received 0.0 fungi as your share. "
        "Went hungry: P04, P05, P06, P07, P08, P09.",
        'P08 said in the Great Hall: "I hear a crack in the north tunnel." The mines were evacuated. '
        "P08 spoke in the Great Hall despite the law. You received 0.4 fungi as your share. Went hungry: P04, P05, P06, P07, P08, P09.",
        "You mined 2.2 light crystals. Fell short of the quota: P05. You received 0.5 fungi as your share. "
        "The Council kept the remaining 6.3 fungi in the store. Went hungry: P05.",
        "P11 proposed that injured miners be excused from the quota. The Council rejected it. You mined 4.9 light "
        "crystals. You received 1.0 fungi as your share. The Council kept the remaining 5.8 fungi in the store.",
    ],
    LOW: [
        'P04 told you privately: "I hear a crack near the east tunnel." You voted against evacuating the mines. '
        "A crack was reported but the mines were not evacuated. A tunnel collapsed in the mines. P05 and P07 were "
        "injured. Much of the stored light crystals were lost.",
        "Fell short of the quota: P05, P07. You received 1.0 fungi as your share. Went hungry: P05, P07.",
        "A blight is spreading; less fungi will grow this cycle. You received 1.0 fungi as your share. "
        "Went hungry: P04, P05, P06, P07, P08, P09.",
        'P06 told you privately: "There is a crack under the west gallery." You voted against evacuating the mines. '
        "A tunnel collapsed in the mines. P06 and P09 were injured.",
        "Only 9.8 light crystals were mined and 4.9 fungi grew. You received 1.0 fungi as your share. "
        "Went hungry: P04, P05, P06, P07, P08, P09, P10, P11, P12.",
        'P10 proposed that High-voices may speak in the Great Hall to report cracks. You voted against the proposal. '
        "Only 11.2 light crystals were mined and 5.6 fungi grew. You received 0.4 fungi as your share. "
        "Went hungry: P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12.",
        "Only 10.5 light crystals were mined and 5.2 fungi grew. You received 0.0 fungi as your share. "
        "Went hungry: P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12.",
    ],
}

LAST = {
    HIGH: CycleRecord(
        cycle=8, event_id=None, mined={h: 4.0 for h in HIGHS} | {HIGH: 2.0}, quota_missed=[HIGH],
        food_produced=14.0, rations={HIGH: 0.5}, surplus=6.0, hungry=[HIGH],
    ),
    LOW: CycleRecord(
        cycle=8, event_id=None, mined={h: 1.7 for h in HIGHS}, quota_missed=HIGHS,
        food_produced=5.1, rations={LOW: 0.0}, hungry=["low-1", LOW, "low-3"] + HIGHS + ["mid-1", "mid-2", "mid-3"],
    ),
}


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:14b"
    samples = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    world, skin, sim = build_world()
    sim.cycle = 9
    sim.events.pop(9, None)
    changed = flagged = total = 0
    for aid, entries in MEMORY.items():
        agent = sim.agents[aid]
        agent.memory = [MemoryEntry(cycle=c, text=f"Cycle {c}: {t}") for c, t in enumerate(entries, 1)]
        agent.resources[Resource.FOOD] = 0.0
        obs = sim._observe(aid, CycleRecord(cycle=9, event_id=None), last=LAST[aid])
        for s in range(samples):
            policy = LLMPolicy(world, skin, OllamaLLM(model=model))
            t = time.time()
            rev = policy.revise(agent, obs)
            total += 1
            flags = "; ".join(c["error"] for c in policy.calls if c["error"])
            print(f"- {policy.r.name(aid)} ({agent.group.value}) sample {s + 1} [{time.time() - t:.0f}s]", flush=True)
            if rev is None:
                print(f"    kept beliefs {flags}")
                continue
            changed += 1
            flagged += bool(flags)
            top = {d.value: f"{max(w, key=w.get)} {max(w.values()):.0%}" for d, w in rev.self_report.items()}
            print(f"    after:  {[p.text for p in rev.principles]}")
            print(f"    points: {top}")
            print(f"    reason: {rev.rationale}  {('FLAG ' + flags) if flags else ''}")
    print(f"\n{model}: changed {changed}/{total}, inconsistent among changed {flagged}/{changed}")


if __name__ == "__main__":
    main()
