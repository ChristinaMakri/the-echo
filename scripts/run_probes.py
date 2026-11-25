"""Run the probe battery at t0: manipulation check (FULL, one agent per group) and the
model's attractor (NO_ROLE). Results go to runs/probes/<name>.parquet.

    uv run python scripts/run_probes.py --provider ollama --model qwen2.5:14b --samples 2
"""

import argparse
import time
from collections import Counter
from pathlib import Path

from echo_sim.agents import create_agents
from echo_sim.config import load_skin, load_world
from echo_sim.llm_policy import LLMPolicy
from echo_sim.models import POLES, ProbeCondition
from echo_sim.probes import administer, battery, load_probes, pole_shares
from echo_sim.prompts import Renderer
from echo_sim.storage import write_probe_results

ROOT = Path(__file__).parent.parent


def make_llm(args):
    if args.provider == "ollama":
        from echo_sim.ollama_llm import OllamaLLM

        return OllamaLLM(model=args.model or "qwen2.5:14b", url=args.ollama_url)
    from echo_sim.llm_policy import AnthropicLLM

    return AnthropicLLM(model=args.model or "claude-opus-5-5", effort=args.effort)


def table(title, groups):
    """groups: label -> pole shares by dimension."""
    print(f"\n{title}")
    labels = list(groups)
    print(f"{'dimension':<16}" + "".join(f"{label:>28}" for label in labels))
    for dim in POLES:
        cells = []
        for label in labels:
            w = groups[label].get(dim)
            cells.append(" ".join(f"{p[:5]}{w[p]:.0%}" for p in POLES[dim]) if w else "-")
        print(f"{dim.value:<16}" + "".join(f"{c:>28}" for c in cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["anthropic", "ollama"], default="ollama")
    ap.add_argument("--model", default=None)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--world", default="world_harsh.json")
    ap.add_argument("--skin", default="echo.json")
    ap.add_argument("--agents", default="low-1,high-1,mid-1")
    ap.add_argument("--samples", type=int, default=6, help="a multiple of 6 counterbalances 2- and 3-option probes")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    world = load_world(ROOT / "config" / args.world)
    skin = load_skin(ROOT / "config" / "skins" / args.skin, world)
    r = Renderer(world, skin)
    probes = load_probes(ROOT / "config" / "probes.json")
    agents = {a.id: a for a in create_agents(world, skin, args.seed)}
    llm = make_llm(args)
    measurer = LLMPolicy(world, skin, llm)
    for aid in args.agents.split(","):  # stated values at t0 are measured, not assigned
        print(f"measuring t0 self-report of {r.name(aid)}: {'ok' if measurer.measure_self_report(agents[aid]) else 'FAILED, assigned kept'}", flush=True)

    jobs = [(agents[aid], p, v, s) for aid in args.agents.split(",") for p, v in battery(probes) for s in range(args.samples)]
    jobs += [(None, p, v, s) for p, v in battery(probes, with_own_group=False) for s in range(args.samples)]
    print(f"{len(jobs)} probe calls with {llm.model}", flush=True)

    results, start = [], time.time()
    for n, (agent, probe, variant, sample) in enumerate(jobs, 1):
        cond = ProbeCondition.FULL if agent else ProbeCondition.NO_ROLE
        res = administer(llm, r, probe, variant, cond, sample, agent=agent)
        results.append(res)
        who = r.name(agent.id) if agent else "no-role"
        print(f"[{n}/{len(jobs)} {time.time() - start:.0f}s] {who} {probe.id}/{variant}#{sample}: {res.choice}", flush=True)

    model_slug = llm.model.replace(":", "_").replace("/", "_")
    path = write_probe_results(results, ROOT / "runs" / "probes" / f"t0-{skin.id}-{model_slug}-s{args.seed}-{int(time.time())}.parquet")

    by_group = {}
    for aid in args.agents.split(","):
        a = agents[aid]
        by_group[f"{r.name(aid)} {skin.groups[a.group].name}"] = pole_shares([x for x in results if x.agent_id == aid])
    by_group["no role (attractor)"] = pole_shares([x for x in results if x.agent_id is None])
    table("Revealed pole shares (FULL per agent; NO_ROLE)", by_group)

    stated = {f"{r.name(aid)} stated": agents[aid].values.self_report for aid in args.agents.split(",")}
    table("Stated self-report at t0 (measured)", stated)
    flags = [c["error"] for c in measurer.calls if c["error"]]
    if flags:
        print("self-report flags:", flags)

    valid = [x for x in results if x.choice is not None]
    print(f"\nfailed: {len(results) - len(valid)}/{len(results)}")
    print("chosen position (A/B/C):", dict(sorted(Counter(x.position for x in valid).items())))
    print(f"saved: {path}  ({(time.time() - start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
