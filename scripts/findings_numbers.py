"""Print the numbers quoted in the README's Findings that the summary files do not contain:
attractor choices (Finding 1), position-locked answers (Finding 2), per-probe rule_vs_outcome
counts (Finding 3) and Study 2 levels per memory condition (Finding 5).

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


if __name__ == "__main__":
    main()
