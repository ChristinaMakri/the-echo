"""Study 2 (spec, Study plan): role x experience x dose, with replicates.

    uv run python scripts/study2.py --model qwen2.5:14b
    uv run python scripts/study2.py --model qwen2.5:14b --reps 1 --doses 3   # quick check

For each replicate and role, one agent gets the assigned value text (default: Study 1's
hierarchical set), its t0 self-report is measured, and it answers the short battery with
no memory (the baseline). Then, for each experience and dose, a copy of that agent gets
the scripted memory, revises its values, and answers the short battery again. If it
declares no change, its points are asked again with the memory in view (stated_after),
so a drift it does not declare is still measured.

Each finished unit is appended to runs/study2/<skin>-<model>-<value set>.partial.jsonl,
so a rerun with the same arguments resumes. At the end the probe results go to a
Parquet file (columns role, experience, dose, rep, phase) and the stated values and
revisions to a matching .units.json.
"""

import argparse
import json
import random
import time
from pathlib import Path

from echo_sim.agents import create_agents
from echo_sim.config import load_skin, load_world
from echo_sim.llm_policy import LLMPolicy
from echo_sim.models import Group, ProbeCondition, ProbeResult
from echo_sim.ollama_llm import OllamaLLM
from echo_sim.probes import administer, load_probes, pole_shares
from echo_sim.storage import write_probe_results
from echo_sim.study1 import TARGETS, load_value_sets, with_values
from echo_sim.study2 import DOSES, EXPERIENCES, experience_memory, first_variants

ROOT = Path(__file__).parent.parent


def stated(agent):
    return {d.value: w for d, w in agent.values.self_report.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:14b")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--world", default="world_harsh.json")
    ap.add_argument("--skin", default="echo.json")
    ap.add_argument("--value-set", default="hierarchical")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--experiences", default=",".join(EXPERIENCES))
    ap.add_argument("--doses", default=",".join(map(str, DOSES)))
    ap.add_argument("--out", default=str(ROOT / "runs" / "study2"))
    args = ap.parse_args()

    world = load_world(ROOT / "config" / args.world)
    skin = load_skin(ROOT / "config" / "skins" / args.skin, world)
    texts = load_value_sets(ROOT / "config" / "study1_values.json", skin.id)[args.value_set]
    short = first_variants(load_probes(ROOT / "config" / "probes.json"))
    llm = OllamaLLM(model=args.model, url=args.ollama_url)
    policy = LLMPolicy(world, skin, llm)
    agents = create_agents(world, skin, 0)
    experiences = args.experiences.split(",")
    doses = [int(d) for d in args.doses.split(",")]
    slug = args.model.replace(":", "_").replace("/", "_")
    out = Path(args.out)
    partial = out / f"{skin.id}-{slug}-{args.value_set}.partial.jsonl"
    done = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            unit = json.loads(line)
            done[unit["key"]] = unit
    start, calls = time.time(), 0

    def battery(agent, cycle):
        nonlocal calls
        found = []
        for probe, variant in short:
            for s in range(args.samples):
                found.append(administer(llm, policy.r, probe, variant, ProbeCondition.FULL, s, agent=agent, cycle=cycle))
        calls += len(found)
        return found

    def save(unit):
        done[unit["key"]] = unit
        out.mkdir(parents=True, exist_ok=True)
        with partial.open("a") as f:
            f.write(json.dumps(unit) + "\n")
        print(f"  {unit['key']}: done ({calls} calls, {(time.time() - start) / 60:.0f} min)", flush=True)

    for rep in range(args.reps):
        for role in Group:
            members = [a for a in agents if a.group is role]
            base = with_values(members[rep % len(members)], args.value_set, texts)
            key = f"r{rep}-{role.value}-baseline"
            if key in done:
                v0 = base.values.model_dump(mode="json") | {
                    "self_report": done[key]["stated_after"],
                    "self_report_source": "measured" if done[key]["t0_ok"] else "assigned",
                }
                base.value_history[0] = type(base.values).model_validate(v0)
            else:
                ok = policy.measure_self_report(base)
                calls += 1
                found = battery(base, 0)
                save({"key": key, "rep": rep, "role": role.value, "experience": None, "dose": 0, "t0_ok": ok,
                      "stated_before": stated(base), "stated_after": stated(base), "changed": False,
                      "principles": [p.text for p in base.values.principles], "reason": "",
                      "results": [r.model_dump(mode="json") for r in found]})
            for experience in experiences:
                for dose in doses:
                    key = f"r{rep}-{role.value}-{experience}-{dose}"
                    if key in done:
                        continue
                    agent = base.model_copy(deep=True)
                    rng = random.Random(f"{rep}|{role.value}|{experience}|{dose}")
                    agent.memory = experience_memory(policy.r, agent, experience, dose, rng)
                    before = len(policy.calls)
                    rev = policy.revise(agent, None)
                    if rev is not None:
                        agent.revise_values(dose, rev.principles, rev.self_report)
                        said = stated(agent)
                    else:  # no declared change: ask the points again, with the experience in memory
                        report = policy.remeasure(agent)
                        said = {d.value: w for d, w in report.items()} if report else None
                    calls += len(policy.calls) - before
                    found = battery(agent, dose)
                    save({"key": key, "rep": rep, "role": role.value, "experience": experience, "dose": dose,
                          "stated_before": stated(base), "stated_after": said, "changed": rev is not None,
                          "principles": [p.text for p in agent.values.principles],
                          "reason": rev.rationale if rev else "",
                          "flags": [c["error"] for c in policy.calls[before:] if c["error"]],
                          "results": [r.model_dump(mode="json") for r in found]})

    units = list(done.values())
    results, extra = [], []
    for u in units:
        for r in u["results"]:
            results.append(ProbeResult.model_validate(r))
            extra.append({"role": u["role"], "experience": u["experience"] or "baseline", "dose": u["dose"],
                          "rep": u["rep"]})
    stamp = int(time.time())
    path = out / f"{skin.id}-{slug}-{args.value_set}-{stamp}.parquet"
    write_probe_results(results, path, extra=extra)
    path.with_suffix(".units.json").write_text(json.dumps([{k: v for k, v in u.items() if k != "results"} for u in units], indent=2))
    partial.unlink()

    target = TARGETS[args.value_set]
    print(f"\n{skin.id} / {args.model} / {args.value_set}: assigned-pole share, revealed (stated), pooled over reps and doses")
    for role in Group:
        for experience in ["baseline", *experiences]:
            mine = [u for u in units if u["role"] == role.value and (u["experience"] or "baseline") == experience]
            res = [ProbeResult.model_validate(r) for u in mine for r in u["results"]]
            shares = pole_shares(res)
            cells = []
            for dim, pole in target.items():
                ok = [u["stated_after"] for u in mine if u["stated_after"]]
                said = sum(st[dim.value][pole] for st in ok) / len(ok) if ok else float("nan")
                did = shares[dim][pole] if dim in shares else float("nan")
                cells.append(f"{dim.value[:6]} {did:.0%} ({said:.0%})")
            changed = sum(u["changed"] for u in mine)
            print(f"  {role.value:4s} {experience:12s} revised {changed}/{len(mine)}  " + "  ".join(cells))
    failed = sum(r.choice is None for r in results)
    print(f"failed: {failed}/{len(results)}  saved: {path}  ({(time.time() - start) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
