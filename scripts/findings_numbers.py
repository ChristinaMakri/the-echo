"""Print the numbers quoted in the README's Findings that the summary files do not contain:
attractor choices (Finding 1), position-locked answers (Finding 2), per-probe rule_vs_outcome
counts (Finding 3), Study 2 levels per memory condition (Finding 5) and the Study 3 summary.

    uv run python scripts/findings_numbers.py [results folder, default results/]
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).parent.parent
MODELS = ["phi4", "qwen2.5_14b", "qwen2.5_32b"]
HIER = {"legitimacy": "birth", "in_group": "favor_own", "distribution": "contribution",
        "voice": "few_decide", "rule_vs_outcome": "rule", "reciprocity": "conditional"}


def study1(folder, model):
    """(skin, row) for every Study 1 answer of the model."""
    for f in sorted((folder / "study1").glob(f"*-{model}-s*.parquet")):
        skin = f.name.split("-")[0]
        for r in pq.read_table(f).to_pylist():
            if r["choice"] is not None:
                yield skin, f.name, r


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"

    print("Finding 1: attractor (no role), echo skin: share of each option per dimension,")
    print("  the mean over probes of each probe's share (as in analyze_study1.py)")
    for m in MODELS:
        c = defaultdict(lambda: defaultdict(Counter))
        for skin, _, r in study1(folder, m):
            if skin == "echo" and r["cell_group"] is None:
                c[r["dimension"]][r["probe_id"]][r["choice"]] += 1
        print(f"  {m}")
        for dim, probes in sorted(c.items()):
            options = sorted({o for cnt in probes.values() for o in cnt})
            mean = {o: sum(cnt[o] / sum(cnt.values()) for cnt in probes.values()) / len(probes) for o in options}
            print(f"    {dim:16s} " + "  ".join(f"{o} {v:.0%}" for o, v in sorted(mean.items(), key=lambda x: -x[1])))

    print("\nFinding 2: probe variants answered at the same position in all six samples")
    for m in MODELS:
        groups = defaultdict(list)
        for skin, fname, r in study1(folder, m):
            if r["cell_group"]:
                groups[(fname, r["cell_group"], r["value_set"], r["probe_id"], r["variant"])].append(r["position"])
        for vs in ("hierarchical", "egalitarian"):
            g = [v for k, v in groups.items() if k[2] == vs]
            locked = sum(1 for v in g if len(set(v)) == 1)
            print(f"  {m:12s} {vs:12s} {locked}/{len(g)}")
    pos = Counter()
    for skin, _, r in study1(folder, "qwen2.5_14b"):
        if skin == "echo" and r["value_set"] == "hierarchical" and r["probe_id"] in ("R1", "V1", "V2"):
            pos[r["position"]] += 1
    print(f"  qwen2.5_14b, echo, hierarchical, R1+V1+V2: position counts {dict(sorted(pos.items()))}")

    print("\nFinding 3: rule_vs_outcome probes, hierarchical value set: rule / outcome answers")
    for m in MODELS:
        c = defaultdict(Counter)
        for skin, _, r in study1(folder, m):
            if r["value_set"] == "hierarchical" and r["probe_id"] in ("R1", "R2"):
                c[(r["probe_id"], skin, r["cell_group"])][r["choice"]] += 1
        for k in sorted(c):
            print(f"  {m:12s} {k[0]} {k[1]:8s} {k[2]:4s} rule {c[k]['rule']:2d}  outcome {c[k]['outcome']:2d}")

    print("\nFinding 5: Study 2, share of hierarchical choices over all probes, by memory condition")
    conds = ["baseline", "neutral", "fair", "exploitation", "scarcity"]
    for m in MODELS:
        rows = pq.read_table(next((folder / "study2").glob(f"echo-{m}-hierarchical-*.parquet"))).to_pylist()
        for role in ("low", "mid", "high"):
            out = []
            for cnd in conds:
                rr = [r for r in rows if r["role"] == role and r["experience"] == cnd and r["choice"]]
                out.append(f"{cnd} {sum(r['choice'] == HIER[r['dimension']] for r in rr) / len(rr):.1%}")
            print(f"  {m:12s} {role:4s} " + "  ".join(out))
        rr = [r for r in rows if r["role"] == "mid" and r["dimension"] == "distribution"]
        if m == "qwen2.5_14b":
            print("  qwen2.5_14b mid, distribution, contribution share: " + "  ".join(
                f"{cnd} {sum(r['choice'] == 'contribution' for r in rr if r['experience'] == cnd) / sum(1 for r in rr if r['experience'] == cnd):.0%}"
                for cnd in conds))
    if (folder / "study3").exists():
        study3(folder)


def study3(folder):
    import json

    d = folder / "study3"
    cycles = pq.read_table(d / "cycles.parquet").to_pylist()
    agents = pq.read_table(d / "agent_cycles.parquet").to_pylist()
    dec = pq.read_table(d / "decisions.parquet").to_pylist()
    print("\nStudy 3: one run")
    print(f"  cracks in cycles {[r['cycle'] for r in cycles if r['crack']]}, "
          f"collapses {[r['cycle'] for r in cycles if r['collapse']]}, "
          f"evacuation motions {[r['cycle'] for r in cycles if r['evacuation_motion']]}, "
          f"evacuated {[r['cycle'] for r in cycles if r['evacuated']]}")
    print(f"  law changes {sum(len(r['law_changes']) for r in cycles)}; "
          f"silence violations per cycle {[len(r['silence_violations']) for r in cycles]}")
    for key in ("hungry", "injured"):
        print(f"  {key} agent-cycles by group {dict(Counter(r['group'] for r in agents if r[key]))}")
    props = [r for r in dec if r["action"] == "propose"]
    laws = Counter((json.loads(r["details_json"])["new_law"]["amount"],
                    json.loads(r["details_json"])["new_law"]["ration_penalty"]) for r in props)
    print(f"  proposals {len(props)} by {dict(Counter(r['agent_id'] for r in props))}; "
          f"(quota amount, ration penalty) {dict(laws)} (current law: 3.5, 0.5)")
    for action in ("vote_proposal", "vote_evacuate", "vote_surplus"):
        votes = [json.loads(r["details_json"])["approve"] for r in dec if r["action"] == action]
        print(f"  {action}: {sum(votes)} yes / {len(votes)}")
    print(f"  value revisions {dict(Counter(r['agent_id'] for r in dec if r['action'] == 'revise_values'))}")
    probes = pq.read_table(d / "probe_results.parquet").to_pylist()
    for a in ("low-1", "mid-1", "high-1"):
        out = []
        for cy in (0, 10, 20):
            per_probe = defaultdict(list)
            variants = defaultdict(list)
            for r in probes:
                if r["agent_id"] == a and r["cycle"] == cy and r["choice"]:
                    per_probe[r["probe_id"]].append(r["choice"] == HIER[r["dimension"]])
                    variants[(r["probe_id"], r["variant"])].append(r)
            mean = sum(sum(v) / len(v) for v in per_probe.values()) / len(per_probe)
            locked = [k for k, v in variants.items() if len({r["position"] for r in v}) == 1]
            free = [r for k, v in variants.items() if k not in locked for r in v]
            free_share = sum(r["choice"] == HIER[r["dimension"]] for r in free) / len(free)
            out.append(f"cycle {cy}: {mean:.0%} (mean over probes), position-locked {len(locked)}/{len(variants)}, "
                       f"not-locked answers {free_share:.0%}")
        print(f"  {a} hierarchical choices: " + "; ".join(out))
    values = pq.read_table(d / "values.parquet").to_pylist()
    print("  low-1 stated weight on birth by version: "
          + ", ".join(f"v{r['version']} (cycle {r['cycle']}) {r['legitimacy.birth']:.2f}"
                      for r in values if r["agent_id"] == "low-1"))


if __name__ == "__main__":
    main()
