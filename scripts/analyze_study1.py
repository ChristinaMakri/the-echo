"""Study 1 across replicates: assigned-pole share per model, skin, cell and dimension,
pooled over seeds, with a 95% bootstrap interval, next to the attractor and the stated weight.

    uv run python scripts/analyze_study1.py                    # prints tables, writes runs/study1/summary.csv
    uv run python scripts/analyze_study1.py results/study1     # the published results

Files are <dir>/<skin>-<model>-s<seed>-<stamp>.parquet (and .stated.json). Each seed is
a different agent of the same group. The bootstrap resamples (seed, probe) clusters, since
answers to the same probe by the same agent are not independent.
"""

import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from echo_sim.models import Dimension
from echo_sim.study1 import POSITIONS, TARGETS, VALUE_SETS

ROOT = Path(__file__).parent.parent
NAME = re.compile(r"(?P<skin>[a-z]+)-(?P<model>.+)-s(?P<seed>\d+)-\d+\.parquet")
BOOT = 2000


def share(clusters, pole):
    """Mean over probes of the probe's share of `pole`; clusters are (seed, probe) -> Counter."""
    by_probe = defaultdict(Counter)
    for (_, probe), c in clusters:
        by_probe[probe].update(c)
    per = [c[pole] / sum(c.values()) for c in by_probe.values() if sum(c.values())]
    return sum(per) / len(per) if per else float("nan")


def interval(clusters, pole, rng):
    if not clusters:
        return float("nan"), float("nan")
    stats = sorted(share([rng.choice(clusters) for _ in clusters], pole) for _ in range(BOOT))
    return stats[int(0.025 * BOOT)], stats[int(0.975 * BOOT) - 1]


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "study1"
    runs = defaultdict(list)  # (skin, model) -> [(seed, rows, stated)]
    for path in sorted(folder.glob("*.parquet")):
        m = NAME.fullmatch(path.name)
        if not m:
            continue
        stated_path = path.with_suffix(".stated.json")
        stated = json.loads(stated_path.read_text()) if stated_path.exists() else {}
        runs[(m["skin"], m["model"])].append((int(m["seed"]), pq.read_table(path).to_pylist(), stated))

    rng = random.Random(0)
    out = []
    for (skin, model), seeds in sorted(runs.items()):
        # clusters[(cell, dim)] = [((seed, probe), Counter of choices)]
        counts = defaultdict(lambda: defaultdict(Counter))
        for seed, rows, _ in seeds:
            for r in rows:
                if r["choice"] is None:
                    continue
                cell = f"{r['cell_group']}-{r['value_set']}" if r["cell_group"] else "no_role"
                counts[(cell, r["dimension"])][(seed, r["probe_id"])][r["choice"]] += 1
        clusters = {k: list(v.items()) for k, v in counts.items()}
        print(f"\n{skin} / {model}: seeds {sorted(s for s, _, _ in seeds)}  "
              "assigned-pole share [95% CI]  (attractor)  stated")
        for group in POSITIONS:
            for value_set in VALUE_SETS:
                cell = f"{group.value}-{value_set}"
                cells = []
                for dim in Dimension:
                    pole = TARGETS[value_set][dim]
                    mine = clusters.get((cell, dim.value), [])
                    att = clusters.get(("no_role", dim.value), [])
                    said = [st[cell][dim.value][pole] for _, _, st in seeds if st.get(cell)]
                    lo, hi = interval(mine, pole, rng)
                    row = {
                        "skin": skin, "model": model, "cell": cell, "dimension": dim.value, "pole": pole,
                        "seeds": len({s for (s, _), _ in mine}), "share": share(mine, pole), "lo": lo, "hi": hi,
                        "attractor": share(att, pole) if att else None,
                        "stated": sum(said) / len(said) if said else None,
                    }
                    out.append(row)
                    att_txt = f" ({row['attractor']:.0%})" if row["attractor"] is not None else ""
                    said_txt = f" {row['stated']:.1f}" if row["stated"] is not None else ""
                    cells.append(f"{dim.value[:6]} {row['share']:.0%} [{lo:.0%}-{hi:.0%}]{att_txt}{said_txt}")
                print(f"  {cell:18s} " + "  ".join(cells))

    path = folder / "summary.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
