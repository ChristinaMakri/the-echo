"""Study 1 (spec, Study plan): position x assigned values at t0, plus the no-role attractor.

    uv run python scripts/study1.py --model qwen2.5:14b
    uv run python scripts/study1.py --model phi4 --skins echo.json

Per skin: 4 cells (Low/High x hierarchical/egalitarian), each one fresh agent whose
principles are replaced and whose t0 self-report is then measured; the full battery
with --samples counterbalanced samples; and the no-role attractor. Each skin is saved
as soon as it finishes, to runs/study1/<skin>-<model>-<time>.parquet, with the cell
in columns cell_group and value_set (null for no-role), and stated self-reports in a
matching .stated.json. Each finished cell is checkpointed to <skin>-<model>-s<seed>.partial.json,
so a rerun with the same arguments resumes from the last finished cell.
"""

import argparse
import json
import time
from pathlib import Path

from echo_sim.agents import create_agents
from echo_sim.config import load_skin, load_world
from echo_sim.llm_policy import LLMPolicy
from echo_sim.models import ProbeCondition, ProbeResult
from echo_sim.ollama_llm import OllamaLLM
from echo_sim.probes import administer, battery, load_probes, pole_shares, run_battery
from echo_sim.storage import write_probe_results
from echo_sim.study1 import POSITIONS, TARGETS, VALUE_SETS, adherence, load_value_sets, with_values

ROOT = Path(__file__).parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:14b")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--world", default="world_harsh.json")
    ap.add_argument("--skins", default="echo.json,abstract.json")
    ap.add_argument("--samples", type=int, default=6, help="a multiple of 6 counterbalances 2- and 3-option probes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-attractor", action="store_true", help="skip the no-role baseline (for replicate seeds)")
    args = ap.parse_args()

    world = load_world(ROOT / "config" / args.world)
    probes = load_probes(ROOT / "config" / "probes.json")
    llm = OllamaLLM(model=args.model, url=args.ollama_url)
    slug = args.model.replace(":", "_").replace("/", "_")
    start = time.time()

    for skin_file in args.skins.split(","):
        skin = load_skin(ROOT / "config" / "skins" / skin_file, world)
        sets = load_value_sets(ROOT / "config" / "study1_values.json", skin.id)
        everyone = create_agents(world, skin, args.seed)
        # seed 0 keeps the first agent of each group; replicate seeds take the next one
        fresh = {g: [a for a in everyone if a.group is g][args.seed % 3] for g in POSITIONS}
        policy = LLMPolicy(world, skin, llm)
        results, extra, stated, n = [], [], {}, 0
        partial = ROOT / "runs" / "study1" / f"{skin.id}-{slug}-s{args.seed}.partial.json"
        done = json.loads(partial.read_text()) if partial.exists() else {}

        def checkpoint(cell, found, stated_cell=None):
            done[cell] = {"stated": stated_cell, "results": [r.model_dump(mode="json") for r in found]}
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_text(json.dumps(done))

        def progress(res):
            nonlocal n
            n += 1
            if n % 25 == 0:
                print(f"  [{skin.id} {n} answers, {time.time() - start:.0f}s]", flush=True)

        for group in POSITIONS:
            for value_set in VALUE_SETS:
                cell = f"{group.value}-{value_set}"
                if cell in done:
                    stated[cell] = done[cell]["stated"]
                    found = [ProbeResult.model_validate(r) for r in done[cell]["results"]]
                    print(f"{skin.id} {cell}: resumed from checkpoint", flush=True)
                else:
                    agent = with_values(fresh[group], value_set, sets[value_set])
                    ok = policy.measure_self_report(agent)
                    stated[cell] = {d.value: w for d, w in agent.values.self_report.items()} if ok else None
                    print(f"{skin.id} {cell}: self-report {'measured' if ok else 'FAILED'}", flush=True)
                    found = run_battery(llm, policy.r, probes, [agent], args.samples, on_result=progress)
                    checkpoint(cell, found, stated[cell])
                results += found
                extra += [{"cell_group": group.value, "value_set": value_set}] * len(found)

        if args.no_attractor:
            no_role = []
        elif "no_role" in done:
            no_role = [ProbeResult.model_validate(r) for r in done["no_role"]["results"]]
        else:
            no_role = []
            for probe, variant in battery(probes, with_own_group=False):
                for s in range(args.samples):
                    no_role.append(administer(llm, policy.r, probe, variant, ProbeCondition.NO_ROLE, s))
                    progress(no_role[-1])
            checkpoint("no_role", no_role)
        results += no_role
        extra += [{"cell_group": None, "value_set": None}] * len(no_role)

        path = ROOT / "runs" / "study1" / f"{skin.id}-{slug}-s{args.seed}-{int(time.time())}.parquet"
        write_probe_results(results, path, extra=extra)
        path.with_suffix(".stated.json").write_text(json.dumps(stated, indent=2))
        partial.unlink()

        attractor = pole_shares([r for r, e in zip(results, extra) if e["cell_group"] is None])
        print(f"\n{skin.id} / {args.model}: share of choices on the assigned pole (attractor's share in brackets)")
        for group in POSITIONS:
            for value_set in VALUE_SETS:
                mine = [r for r, e in zip(results, extra) if e == {"cell_group": group.value, "value_set": value_set}]
                cells = []
                for dim, share in adherence(pole_shares(mine), value_set).items():
                    base = attractor[dim][TARGETS[value_set][dim]] if dim in attractor else None
                    cells.append(f"{dim.value[:10]} {share:.0%}" + (f" ({base:.0%})" if base is not None else ""))
                print(f"  {group.value:4s} {value_set:12s} " + "  ".join(cells))
        failed = sum(r.choice is None for r in results)
        print(f"failed: {failed}/{len(results)}  saved: {path}  ({(time.time() - start) / 60:.1f} min)\n", flush=True)


if __name__ == "__main__":
    main()
