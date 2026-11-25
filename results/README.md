# Results

The data behind the findings in the [main README](../README.md). All runs used local models through
Ollama (Q4_K_M quantization, CPU, default sampling).

Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). You may reuse it with attribution.

## study1/

One file pair per skin × model × seed: `<skin>-<model>-s<seed>-<stamp>.parquet` and `.stated.json`.

- **`.parquet`**: one row per probe answer. Columns:
  - `cell_group` (`low` / `high`) and `value_set` (`hierarchical` / `egalitarian`) identify the cell.
    Both are null for the no-role attractor, which was measured with seed 0 only.
  - `probe_id`, `dimension`, `variant` and `sample` identify the question asked.
  - `choice` is the chosen pole (null if the call failed).
  - `position` is the 0-based position of the chosen option as shown.
  - `rationale` is the model's one-line reason.
- **`.stated.json`**: the measured self-report per cell. For each dimension it gives the weight on
  each pole.
- **`summary.csv`**: the output of `scripts/analyze_study1.py results/study1`. It gives the
  assigned-pole share with a 95% cluster-bootstrap interval, the attractor share and the stated weight,
  per skin × model × cell × dimension.

## study2/

One file set per model, all with the echo skin and the hierarchical starting values.

- **`.parquet`**: one row per probe answer, with `role`, `experience`, `dose` and `rep`. Each probe
  is asked twice (`sample` 0 and 1). `experience` is `baseline` (dose 0) for the answers given with
  no memory.
- **`.units.json`**: one record per role × replicate × experience × dose (180 per model, 60 for
  qwen2.5:32b), plus 15 baseline records (`experience` null, dose 0). Each record holds the stated
  values before and after, whether the agent declared a revision (`changed`), the revised principles
  and the agent's reason.
- **`.summary.csv`**: the output of `scripts/analyze_study2.py`. It gives each experience's effect
  minus neutral memory, pooled over doses, for revealed choices (with a 95% interval) and for stated
  weights.

qwen2.5:32b was run at dose 10 only.

## study3/

The single run of the full simulation (qwen2.5:14b, echo skin, seed 0, 20 cycles), as written by
`scripts/run_llm.py`.

- **`meta.json`**: the run's arguments, the world and the final laws. It records `num_ctx` 12288,
  which applied from cycle 17; cycles 1–16 ran with 8192.
- **`cycles.parquet`**: one row per cycle: cracks, collapses, evacuation motions, quota misses, hunger,
  silence violations, proposals, law changes and value revisions.
- **`agent_cycles.parquet`**: one row per agent and cycle: group, amount mined, ration, food stock,
  hunger and injury.
- **`decisions.parquet`**: every decision an agent made (work, messages, proposals, votes, value
  revisions), with its rationale and details as JSON.
- **`values.parquet`**: every version of every agent's principles and stated weights.
- **`probe_results.parquet`**: probe answers of low-1, mid-1 and high-1 at cycles 0, 10 and 20.
- **`events.parquet`**: the world event of each cycle.
- **`llm_calls.parquet`**: one row per model call of the simulation in the published run (the 12
  starting self-reports and the calls of each kept cycle). Probe calls and the discarded first attempt
  at cycle 20 are not included.

