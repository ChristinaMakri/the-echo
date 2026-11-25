"""Repeat the value-revision step on a local model: does it change, and are words and numbers consistent?

    uv run python scripts/test_revision.py qwen2.5:14b [samples]
"""

import sys
import time

from compare_models import build_world

from echo_sim.llm_policy import LLMPolicy
from echo_sim.models import CycleRecord
from echo_sim.ollama_llm import OllamaLLM

AGENTS = ["low-1", "low-2", "high-1", "high-5", "mid-1", "mid-2"]


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:14b"
    samples = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    world, skin, sim = build_world()
    rec = CycleRecord(cycle=4, event_id="probe-dispute")
    changed = flagged = total = 0
    for aid in AGENTS:
        agent = sim.agents[aid]
        for s in range(samples):
            policy = LLMPolicy(world, skin, OllamaLLM(model=model))
            t = time.time()
            rev = policy.revise(agent, sim._observe(aid, rec))
            total += 1
            flags = "; ".join(c["error"] for c in policy.calls if c["error"]) or ""
            name = policy.r.name(aid)
            print(f"- {name} ({agent.group.value}) sample {s + 1} [{time.time() - t:.0f}s]", flush=True)
            print(f"    before: {[p.text for p in agent.values.principles]}")
            if rev is None:
                print(f"    kept beliefs {('[' + flags + ']') if flags else ''}")
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
