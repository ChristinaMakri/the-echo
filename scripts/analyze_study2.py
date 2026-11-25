"""Study 2: what each experience does to stated and revealed values, against neutral memory.

    uv run python scripts/analyze_study2.py runs/study2/<file>.parquet [more.parquet ...]

Several files (same model and skin) are read together, e.g. a wording check run on its own
and compared with the main run's neutral units.

Neutral cycles are the control: any memory at all moves answers, so an experience's effect
is its difference from neutral at the same role and dose. Revealed = share of probe choices
on the assigned (hierarchical) pole; stated = points on that pole. Differences are pooled
over doses, with a 95% bootstrap interval that resamples replicates. Low and Mid have only 3
members, so replicates 4 and 5 reuse agents and the interval is too narrow. Also writes <first file>.summary.csv.
"""

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from echo_sim.models import Dimension
from echo_sim.study1 import TARGETS

TARGET = {d.value: p for d, p in TARGETS["hierarchical"].items()}
BOOT = 2000


def revealed(rows, dim):
    """Mean over probes of the probe's share of the assigned pole."""
    by_probe = defaultdict(Counter)
    for r in rows:
        if r["dimension"] == dim and r["choice"] is not None:
            by_probe[r["probe_id"]][r["choice"] == TARGET[dim]] += 1
    per = [c[True] / sum(c.values()) for c in by_probe.values()]
    return sum(per) / len(per) if per else float("nan")


def main():
    paths = [Path(a) for a in sys.argv[1:]]
    path = paths[0]
    rows = [r for p in paths for r in pq.read_table(p).to_pylist()]
    units = [u for p in paths for u in json.loads(p.with_suffix(".units.json").read_text())]
    by_unit = defaultdict(list)  # (rep, role, experience, dose) -> probe rows
    for r in rows:
        by_unit[(r["rep"], r["role"], r["experience"], r["dose"])].append(r)
    stated = {(u["rep"], u["role"], u["experience"] or "baseline", u["dose"]): u["stated_after"] for u in units}

    def effect(reps, role, exp, dim):
        """Mean over reps and doses of (experience - neutral), revealed and stated."""
        rev, said = [], []
        for rep in reps:
            for dose in doses:
                a, b = by_unit[(rep, role, exp, dose)], by_unit[(rep, role, "neutral", dose)]
                rev.append(revealed(a, dim) - revealed(b, dim))
                sa, sb = stated.get((rep, role, exp, dose)), stated.get((rep, role, "neutral", dose))
                if sa and sb:
                    said.append(sa[dim][TARGET[dim]] - sb[dim][TARGET[dim]])
        return sum(rev) / len(rev), (sum(said) / len(said) if said else float("nan"))

    reps = sorted({u["rep"] for u in units})
    doses = sorted({u["dose"] for u in units if u["experience"]})
    roles = list(dict.fromkeys(u["role"] for u in units))
    exps = [e for e in dict.fromkeys(u["experience"] for u in units if u["experience"]) if e != "neutral"]
    rng = random.Random(0)
    out = []
    print(f"{', '.join(p.name for p in paths)}: experience minus neutral, pooled over doses {doses}, {len(reps)} reps")
    print("revealed (stated) change in the hierarchical pole, [95% CI of revealed]\n")
    for role in roles:
        for exp in exps:
            cells = []
            for dim in Dimension:
                d = dim.value
                rev, said = effect(reps, role, exp, d)
                boot = sorted(effect([rng.choice(reps) for _ in reps], role, exp, d)[0] for _ in range(BOOT))
                lo, hi = boot[int(0.025 * BOOT)], boot[int(0.975 * BOOT) - 1]
                out.append({"role": role, "experience": exp, "dimension": d, "revealed": rev, "lo": lo, "hi": hi,
                            "stated": said})
                mark = "*" if lo > 0 or hi < 0 else " "
                cells.append(f"{d[:6]} {rev:+.0%}{mark}({said:+.2f})")
            print(f"  {role:4s} {exp:12s} " + "  ".join(cells))
    revised = sum(u["changed"] for u in units)
    print(f"\n* interval excludes 0.  declared revisions: {revised}/{sum(1 for u in units if u['experience'])}")
    with path.with_suffix(".summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    main()
