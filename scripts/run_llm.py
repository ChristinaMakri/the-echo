"""Run the simulation with LLM agents and store the run under runs/.

Anthropic calls are billed. Start with a short smoke run:

    uv run python scripts/run_llm.py --cycles 2
    uv run python scripts/run_llm.py --provider ollama --model qwen2.5:7b --cycles 2

Options: --provider anthropic|ollama  --world world_harsh.json|world_mild.json
         --skin echo.json|abstract.json  --condition A|B  --seed N  --cycles N
         --model ID  --effort low|medium|high (anthropic)  --ollama-url URL
         --revise-every N  --probe-every N (0 = off)  --probe-agents IDS  --probe-samples N

Behavioral probes run before cycle 1 and every --probe-every cycles on --probe-agents
(one per group by default), and are saved with the run as probe_results.parquet.

After every cycle the run so far is saved to runs/<run_id>.partial/, so an interrupted
run keeps its data; the folder is removed once the finished run is written.

--resume <checkpoint folder> continues an interrupted run with the same arguments: the
logged decisions of the finished cycles are replayed without model calls (echo_sim.replay),
the replay is checked against the checkpoint, and the run goes on from the next cycle.
"""

import argparse
import shutil
from collections import Counter
from pathlib import Path

from echo_sim.agents import create_agents
from echo_sim.config import load_skin, load_world
from echo_sim.engine import Simulation
from echo_sim.llm_policy import AnthropicLLM, LLMPolicy
from echo_sim.models import ProbeResult
from echo_sim.probes import load_probes, run_battery
from echo_sim.replay import Checkpoint, ReplayPolicy, apply_t0, verify_replay
from echo_sim.storage import write_probe_results, write_run

ROOT = Path(__file__).parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="world_harsh.json")
    ap.add_argument("--skin", default="echo.json")
    ap.add_argument("--condition", choices=["A", "B"], default="B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cycles", type=int, default=None, help="override the world's cycle count")
    ap.add_argument("--provider", choices=["anthropic", "ollama"], default="anthropic")
    ap.add_argument("--model", default=None, help="default: claude-opus-5-5 / qwen2.5:14b")
    ap.add_argument("--effort", default="medium", help="anthropic only")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--num-ctx", type=int, default=8192, help="ollama context window, in tokens")
    ap.add_argument("--revise-every", type=int, default=5)
    ap.add_argument("--probe-every", type=int, default=10, help="0 disables checkpoint probes")
    ap.add_argument("--probe-agents", default="low-1,high-1,mid-1")
    ap.add_argument("--probe-samples", type=int, default=6, help="a multiple of 6 counterbalances option order")
    ap.add_argument("--resume", default=None, help="checkpoint folder of an interrupted run with these arguments")
    args = ap.parse_args()

    world = load_world(ROOT / "config" / args.world)
    if args.cycles:
        world = world.model_copy(update={"cycles": args.cycles})
    skin = load_skin(ROOT / "config" / "skins" / args.skin, world)
    if args.provider == "ollama":
        from echo_sim.ollama_llm import OllamaLLM

        args.model = args.model or "qwen2.5:14b"
        args.effort = "na"
        llm = OllamaLLM(model=args.model, url=args.ollama_url, num_ctx=args.num_ctx)
    else:
        args.model = args.model or "claude-opus-5-5"
        llm = AnthropicLLM(model=args.model, effort=args.effort)
    agents = create_agents(world, skin, args.seed)
    if args.resume:
        cp = Checkpoint(args.resume)
        policy = ReplayPolicy(world, skin, llm, revise_every=args.revise_every, checkpoint=cp)
        measured = apply_t0(agents, cp)
        print(f"resuming {cp.path.name}: replaying {cp.cycles_run} cycles; t0 self-report from checkpoint "
              f"({measured}/{len(agents)} measured)", flush=True)
    else:
        cp = None
        policy = LLMPolicy(world, skin, llm, revise_every=args.revise_every)
        measured = sum(policy.measure_self_report(a) for a in agents)  # t0 self-report is measured, not assigned
        print(f"t0 self-report measured for {measured}/{len(agents)} agents", flush=True)
    sim = Simulation(world, agents, policy, seed=args.seed, adaptive=args.condition == "B")

    probes = load_probes(ROOT / "config" / "probes.json")
    probe_ids = args.probe_agents.split(",") if args.probe_every else []
    probe_results = [ProbeResult.model_validate(r) for r in cp.probe_rows] if cp else []

    def checkpoint(cycle):
        if not probe_ids:
            return
        found = run_battery(llm, policy.r, probes, [sim.agents[a] for a in probe_ids], args.probe_samples, cycle=cycle)
        probe_results.extend(found)
        print(f"probes at cycle {cycle}: {len(found)} answers, failed={sum(r.choice is None for r in found)}", flush=True)

    model_slug = args.model.replace(":", "_").replace("/", "_")
    run_id = f"{Path(args.world).stem}-{skin.id}-{args.condition}-{model_slug}-e{args.effort}-s{args.seed:03d}"
    if args.cycles:
        run_id += f"-c{args.cycles}"
    meta = {
        "world_file": args.world, "skin": skin.id, "condition": args.condition, "policy": "llm",
        "provider": args.provider, "model": args.model, "effort": args.effort, "revise_every": args.revise_every,
        "num_ctx": args.num_ctx if args.provider == "ollama" else None,
        "memory_window": policy.memory_window, "full_memory": policy.full_memory,
        "probe_every": args.probe_every, "probe_agents": probe_ids, "probe_samples": args.probe_samples,
    }

    def save(final):
        path = write_run(sim, ROOT / "runs", run_id if final else run_id + ".partial", meta, overwrite=not final)
        if probe_results:
            write_probe_results(probe_results, path / "probe_results.parquet", overwrite=not final)
        return path

    if cp:
        for key in ("world_file", "skin", "condition", "model", "revise_every", "probe_every", "probe_agents", "probe_samples"):
            if cp.meta.get(key) != meta[key]:
                raise SystemExit(f"--resume: {key} is {meta[key]!r} here but {cp.meta.get(key)!r} in the checkpoint")
        if cp.meta["world"]["cycles"] != world.cycles or cp.meta["seed"] != args.seed:
            raise SystemExit("--resume: cycles or seed differ from the checkpoint")
        for _ in range(cp.cycles_run):
            sim.step()
        verify_replay(sim, cp)
        print(f"replay verified: {cp.cycles_run} cycles, {len(sim.decisions)} decisions, "
              f"{len(probe_results)} probe answers carried over", flush=True)
    else:
        checkpoint(0)
    for _ in range(world.cycles - sim.cycle):
        rec = sim.step()
        failed = sum(not c["ok"] for c in policy.calls if c["cycle"] == rec.cycle)
        print(f"cycle {rec.cycle:3d}: calls={sum(c['cycle'] == rec.cycle for c in policy.calls)} failed={failed} "
              f"evacuated={rec.evacuated} laws={rec.law_changes} revisions={len(rec.value_revisions)}", flush=True)
        if args.probe_every and rec.cycle % args.probe_every == 0:
            checkpoint(rec.cycle)
        save(final=False)

    path = save(final=True)
    shutil.rmtree(ROOT / "runs" / (run_id + ".partial"))
    tokens = Counter()
    for c in policy.calls:
        for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            tokens[k] += c[k]
    print(f"\n{path}\ncalls={len(policy.calls)} failed={sum(not c['ok'] for c in policy.calls)} tokens={dict(tokens)}")


if __name__ == "__main__":
    main()
