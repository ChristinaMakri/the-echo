"""Write one simulation run to disk as Parquet tables plus a meta.json.

Layout of runs/<run_id>/:
  meta.json              run settings, full world config, final laws
  events.parquet         the seeded event schedule
  cycles.parquet         one row per cycle (CycleRecord scalars and id lists)
  agent_cycles.parquet   one row per agent per cycle (long format, for analysis)
  decisions.parquet      every policy call, details as JSON text
  values.parquet         every value version, self-report flattened to columns
  llm_calls.parquet      LLM policies only: one row per call (tokens, stop reason, failures)
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from echo_sim.engine import Simulation
from echo_sim.models import POLES


def write_run(sim: Simulation, out_dir: str | Path, run_id: str, meta: dict, overwrite: bool = False) -> Path:
    """Never overwrites a run, except a checkpoint written with overwrite=True."""
    path = Path(out_dir) / run_id
    path.mkdir(parents=True, exist_ok=overwrite)

    (path / "meta.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "seed": sim.seed,
                "adaptive": sim.adaptive,
                "cycles_run": sim.cycle,
                "created_at": datetime.now(timezone.utc).isoformat(),
                **meta,
                "world": sim.world.model_dump(mode="json"),
                "final_laws": [law.model_dump(mode="json") for law in sim.laws.values()],
            },
            indent=2,
        )
    )

    _write(path / "events.parquet", [e.model_dump(mode="json") for e in sim.events.values()])

    snapshot_fields = {"mined", "rations", "store", "food_stock", "injured"}
    cycles = []
    for r in sim.records:
        row = r.model_dump(mode="json", exclude=snapshot_fields)
        row["store_energy"] = r.store.get("energy", 0.0)
        row["store_food"] = r.store.get("food", 0.0)
        cycles.append(row)
    _write(path / "cycles.parquet", cycles)

    group = {aid: a.group.value for aid, a in sim.agents.items()}
    agent_cycles = [
        {
            "cycle": r.cycle,
            "agent_id": aid,
            "group": group[aid],
            "mined": r.mined.get(aid),  # None for non-producers
            "ration": r.rations.get(aid, 0.0),
            "food_stock": r.food_stock.get(aid, 0.0),
            "quota_missed": aid in r.quota_missed,
            "hungry": aid in r.hungry,
            "injured": aid in r.injured,
            "heard_crack": aid in r.detectors or aid in r.false_alarms,
            "silence_violation": aid in r.silence_violations,
            "unfounded_crack_report": aid in r.unfounded_crack_reports,
        }
        for r in sim.records
        for aid in sim.agents
    ]
    _write(path / "agent_cycles.parquet", agent_cycles)

    decisions = []
    for d in sim.decisions:
        row = d.model_dump(mode="json", exclude={"details"})
        row["details_json"] = json.dumps(d.details, default=str)
        decisions.append(row)
    _write(path / "decisions.parquet", decisions)

    values = []
    for agent in sim.agents.values():
        for v in agent.value_history:
            row = {
                "agent_id": agent.id,
                "group": agent.group.value,
                "version": v.version,
                "cycle": v.cycle,
                "principles_json": json.dumps([p.model_dump(mode="json") for p in v.principles]),
                "self_report_source": v.self_report_source,
            }
            for dim, poles in POLES.items():
                for pole in poles:
                    row[f"{dim.value}.{pole}"] = v.self_report[dim][pole]
            values.append(row)
    _write(path / "values.parquet", values)

    calls = getattr(sim.policy, "calls", None)
    if calls:
        _write(path / "llm_calls.parquet", calls)

    return path


def _write(path: Path, rows: list[dict]) -> None:
    table = pa.Table.from_pylist(rows)
    # A column that is empty in this run infers as null; pin it to string so
    # every run has the same schema and runs can be read together.
    fields = []
    for f in table.schema:
        if pa.types.is_null(f.type):
            f = f.with_type(pa.string())
        elif pa.types.is_list(f.type) and pa.types.is_null(f.type.value_type):
            f = f.with_type(pa.list_(pa.string()))
        fields.append(f)
    pq.write_table(table.cast(pa.schema(fields)), path)


def write_probe_results(results, path: str | Path, overwrite: bool = False, extra: list[dict] | None = None) -> Path:
    """Probe results as one Parquet table; never overwrites unless asked (checkpoints).
    `extra`, one dict per result, adds columns (e.g. a study's cell labels)."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.model_dump(mode="json") for r in results]
    if extra is not None:
        rows = [{**row, **more} for row, more in zip(rows, extra, strict=True)]
    _write(path, rows)
    return path
