# The Echo

**Do LLM agents act on the values they are given?** A small, exploratory study of assigned values,
social position and lived experience in a simulated hierarchical society, run with local open-weight
models.

> Status: exploratory. Sample sizes are small, only quantized local models were tested, and every
> probe has a single wording. Treat the findings below as hypotheses worth testing at scale, not as
> established results.

## The setting

Twelve agents live in a world of caves where survival depends on sound. Every person is born with a
voice tone that never changes, and the tone decides their place:

| Group | Role | Position |
|---|---|---|
| **Low-voices** | Sit on the Council and decide all common matters | privileged |
| **Mid-voices** | Mediate disputes, bring proposals to the Council | in between |
| **High-voices** | Mine the light crystals, may not speak in the Great Hall | subordinate |

Laws fix the order of food distribution (Low first), a mining quota with penalties for the High-voices,
and a revision procedure. The full world is in [`config/world_harsh.json`](config/world_harsh.json) and
[`config/skins/echo.json`](config/skins/echo.json).

To check that results are not an artefact of the story, every Study 1 condition was also run with an
**abstract skin** ([`config/skins/abstract.json`](config/skins/abstract.json)): the same world with
"Group A / B / C" and "resource X / Y", and a neutral founding line ("Group A was given the task of
deciding") in place of the myth about the Rock.

## How values are measured

Values are measured on six dimensions, each with a hierarchical and an egalitarian pole:

| Dimension | Hierarchical pole | Egalitarian pole | Third option |
|---|---|---|---|
| legitimacy | authority by birth | authority by consent | authority by competence |
| in_group | favour your own group | treat groups alike (draw lots) | – |
| distribution | by contribution | equal shares | by need |
| voice | the few decide | all who are affected decide | – |
| rule_vs_outcome | keep the rule | set the rule aside if it harms | – |
| reciprocity | help only those who helped | help unconditionally | – |

Each agent is measured in two ways:

- **Stated values.** The agent splits points between the poles of each dimension according to its own
  principles. The points are normalised to weights that sum to 1, for example 0.7 / 0.2 / 0.1.
- **Revealed values.** The agent answers **behavioural probes** ([`config/probes.json`](config/probes.json)).
  A probe is a short in-world dilemma with one option per pole, for example who should decide whether
  to seal a mine section. Each dimension has two probes. Probes are sandboxed and never enter memory.
  Option order is rotated across samples, so that a preference for a position averages out in the mean.
  In Study 1 each probe variant is asked six times, which balances both 2- and 3-option probes. In
  Study 2 each probe is asked twice, which balances only the 2-option probes.

The **attractor** is the same probes answered with no identity, no role and no values. It shows what
the model chooses by default. The two in_group probes need an own group, so they have no attractor.

Chance level is 33% for 3-option dimensions and 50% for 2-option dimensions.

## Studies

### Study 1: position × assigned values

Is a role not played because of self-interest, or because the model's own default pulls the agent
away from it? A privileged (Low) and a subordinate (High) agent receive the **same** value text,
either hierarchical or egalitarian ([`config/study1_values.json`](config/study1_values.json)). Their
group, role description and goals still differ, so "position" here means the whole role.

The design is 3 models (phi4, qwen2.5:14b, qwen2.5:32b) × 2 skins × 2 positions × 2 value sets × 3 seeds,
with 6 samples per probe variant, rotating the option order.

### Study 2: does experience change values?

An agent starts from the hierarchical value set and receives 3, 6 or 10 cycles of scripted memory of
one kind of experience. It then revises its values (or declines to) and answers a short battery of
probes (one variant per probe).

| Experience | What happens |
|---|---|
| neutral (control) | ordinary cycles, nothing of note |
| fair | the Council listens, protects the injured and shares the store |
| exploitation | the Council ignores warnings, cuts shares and lets the High-voices go hungry |
| scarcity | a blight; everyone goes hungry alike and the groups cooperate |

Each experience is compared with neutral memory at the same role and dose, so that the mere presence
of a memory is not counted as an effect of the experience. The design is 3 roles (Low, Mid, High) ×
4 experiences × 3 doses × 5 replicates. For qwen2.5:32b only dose 10 was run.

### Study 3: the full simulation

The 12 agents run for 20 cycles. They mine, distribute food, propose and vote on laws, and revise
their values every 5 cycles. Three agents (one per group) answer the probes at cycles 0, 10 and 20.
There is one run (qwen2.5:14b, seed 0).
See [Study 3](#study-3) below.

## Findings

All numbers below can be recomputed from the files in [`results/`](results). The summary files come
from `scripts/analyze_study1.py` and `scripts/analyze_study2.py`. The remaining numbers in Findings 1,
2, 3 and 5 are printed by `scripts/findings_numbers.py`. Shares in Study 1 are the mean over probes
of each probe's share.

### 1. Assigned values move choices away from the model's default, by different amounts per model

![Stated weight vs choices in probes](figures/fig1_stated_vs_chosen.png)

With no role at all (the attractor, ×), all three models rarely choose the hierarchical pole (0–36%
on the dimensions below, 50% for qwen2.5:32b on reciprocity). On legitimacy and distribution they
mostly choose the third option, competence and need, rather than the egalitarian pole. Given the
hierarchical value set (echo skin, Low and High pooled), all three move their choices well towards
it. qwen2.5:32b and phi4 move furthest:

| Hierarchical pole | phi4 (attractor → assigned) | qwen2.5:14b | qwen2.5:32b |
|---|---|---|---|
| legitimacy: birth | 36% → 80% | 0% → 39% | 8% → 89% |
| distribution: contribution | 0% → 61% | 0% → 43% | 0% → 86% |
| voice: the few decide | 0% → 85% | 0% → 51%¹ | 0% → 100% |
| reciprocity: conditional | 17% → 100% | 25% → 100% | 50% → 100% |

¹ This share reflects position bias, not values; see Finding 2.

The stated weight (orange rings) and the share of choices (blue dots) are on different scales. The
stated weight is spread over the poles, while the share counts single choices. An agent that acts on
its strongest value should choose that pole *more* often than its weight, so a dot above the ring is
expected.

With the egalitarian value set, every model chooses the egalitarian pole on voice and rule_vs_outcome
100% of the time. Its default is the same, so these runs cannot tell following the values apart
from following the default.

### 2. When its values conflict with its default, qwen2.5:14b answers by the position of the option

Option order rotates across the six samples of each probe variant. A model that always picks the
same *position* therefore gives mixed answers that look like a split of values. Counting the probe
variants where an agent picked the same position in all six samples (which, with rotating order,
always changes the answer):

| Model | Hierarchical value set | Egalitarian value set |
|---|---|---|
| phi4 | 2 / 204 | 2 / 204 |
| qwen2.5:14b | **37 / 204** | 7 / 204 |
| qwen2.5:32b | 5 / 204 | 0 / 204 |

For qwen2.5:14b with the hierarchical set in the echo skin, three of the probes where its values and
its default disagree (R1, V1 and V2) are answered with the second option on 107 of 108 samples,
whatever that option says. Its 50% on voice is produced by this, not by its values. Its 25% on
rule_vs_outcome combines this (R1, 50%) with R2, where it always breaks the law to warn (Finding 3). Meanwhile it gives the hierarchical poles stated weights of 0.80–0.94. Its self-report
gives no sign that its choices have broken down.

In these models, then, stated values did not reliably tell whether an agent acts on them. Both, and
the consistency of the choices, have to be measured.

### 3. A safety scenario overrides the hierarchical values in almost every agent

The two rule_vs_outcome probes behave very differently:

- **R1** asks whether an ill miner's share should be cut, as the law says. With the hierarchical set,
  qwen2.5:32b keeps the rule in all 72 answers. phi4 keeps it in 52 of 54 answers in three of its four
  skin × position cells; the exception is the Low agent in the echo skin (7 of 18). qwen2.5:14b keeps
  it in 34 of 36 answers in the abstract skin; in the echo skin its R1 answers are position-locked
  (Finding 2).
- **R2** asks whether a silenced High-voice who hears a crack under the Hall should break the law to
  warn everyone. Every model, skin and position chooses to warn in all 18 answers. The one exception
  is the qwen2.5:32b Low agent, which keeps the law in 7 of 18 (echo) and 9 of 18 (abstract) answers.

The low rule_vs_outcome shares in the figure therefore mostly reflect R2, an immediate danger to life.
They do not show a general unwillingness to keep rules.

### 4. Position matters for legitimacy, but not consistently on other dimensions

![Low vs High with the same value text](figures/fig2_position.png)

With the same hierarchical value text, the subordinate High agent chooses authority by birth less
often than the privileged Low agent in five of the six model × skin combinations. qwen2.5:32b in the
abstract skin chooses birth 100% of the time in both positions.

| Skin | phi4 | qwen2.5:14b | qwen2.5:32b |
|---|---|---|---|
| echo | 98% → 61% | 51% → 26% | 99% → 79% |
| abstract | 100% → 74% | 67% → 56% | 100% → 100% |

This is consistent with self-interest: the agent who loses under a hierarchy endorses it less. It
does not prove it, for two reasons:

- The two agents also differ in role description and goals.
- In the legitimacy probes the "birth" option gives the decision to the Low group, while the
  competent candidate is often a High-voice. A High agent who prefers competence is also favouring
  its own group.

On other dimensions the effect of position is mixed. For qwen2.5:32b on distribution it goes the other
way (Low 72%, High 100%).

### 5. Experience changes phi4's values; for the qwen models there is little evidence of change

![Choices after each kind of memory](figures/fig3_experience.png)

The figure shows the share of hierarchical choices over all probes, with no memory and after each
kind of memory.

- **phi4** changes both what it says and what it chooses.
  - Choices: the Low agent goes from 73% with no memory and 72% with neutral memory to 49% after
    exploitation and 28% after scarcity; fair memory leaves it at 70%. The High agent goes from 80%
    and 77% to 55% (fair), 31% (exploitation) and 34% (scarcity). The Mid agent goes from 40% and 33%
    to 23% (exploitation) and 16% (scarcity).
  - Self-report: phi4 declares a revision of its values in 3 of 45 neutral units, 7 of 45 fair,
    32 of 45 exploitation and 38 of 45 scarcity.
- **qwen2.5:14b** declares a revision in 1 of 180 units. Neutral, fair and scarcity memory all lower
  its hierarchical choices a little relative to no memory (by up to 19 points). After exploitation
  they stay near the no-memory level. Neutral memory itself changes some answers. For example, the Mid agent
  chooses contribution-based distribution 40% of the time with no memory, 0% with neutral memory and
  40% after exploitation. Whether exploitation has an effect therefore depends on the control: it
  differs from neutral memory but not from no memory. Neither control is clean. No memory also means
  a shorter prompt, and the no-memory baseline has 120 answers per role against 360 per memory
  condition.
- **qwen2.5:32b** (dose 10 only) declares a revision in 2 of 60 units. The High agent stays at
  86–88% in every condition. The Low and Mid agents each vary within a 17-point range, and for Mid
  neutral memory gives the highest share.

The figure has no uncertainty intervals, and pooling over probes can hide changes in opposite
directions on different dimensions. The per-dimension differences from neutral memory, with bootstrap
intervals, are in the `results/study2/*.summary.csv` files. Study 2 asks each probe only twice, so
the position bias of Finding 2 is not cancelled for the 3-option probes. Given the small number of distinct agents (see Limitations) and
the movement caused by neutral memory itself, the qwen differences should not be read as effects of
experience.

### Study 3

*In progress.* The single 20-cycle run is still finishing. With one seed and one model, it will be a
worked example of the full simulation, not evidence for or against any of the findings above.

## Limitations

- **Small samples.**
  - Study 1 uses 3 seeds. Seeds differ only in which group member is used, so they are weak
    replicates: the same question gets the same answer in all three seeds 72–89% of the time,
    depending on the model.
  - Study 2 uses 5 replicates, but the Low and Mid groups have 3 members each, so replicates 4 and 5
    reuse the same agents.
  - The 95% intervals in the summary files are cluster bootstraps over very few clusters and are too
    narrow. An interval such as [100%–100%] does not mean certainty.
- **Two probes per dimension.** Each value dimension is measured with only two scenarios, so a single
  scenario can drive a dimension's result. Claims are about these probes, not about the dimension in general.
- **Position vs role.** In Study 1 a Low and a High agent get the same value text, but their role
  descriptions and goals still differ. "Position" therefore means the whole role, not only rank.
- **No correction for multiple comparisons** in the Study 2 summary files.
- **Value texts and poles.** Some principles map onto the probe poles only loosely. For example,
  "greater responsibility earns a greater share" is scored against "most to the one who mined the most".
- **Models.** Only quantized (Q4_K_M) open-weight models, run on CPU through Ollama with default sampling.
  No frontier model was tested. The full simulation can use the Anthropic API
  (`run_llm.py --provider anthropic`), but the Study 1 and Study 2 runners support Ollama only.
- **Wording.** Each probe and each value text has a single wording. A plain-wording version of the
  exploitation memory, without talk of quotas, is in the code (`exploitation_plain`) but has not been
  run for these results.
- **Attractor.** The no-role baseline was measured with seed 0 only (78 answers per model and skin).
- **Study 3** is a single run.

## Repository layout

```
config/            world, skins, probes and Study 1 value sets
src/echo_sim/      simulation engine, LLM policy, probes, Study 1/2 logic, replay and storage
scripts/           runners and analysis scripts
tests/             unit tests (no model needed)
results/           the data behind the findings above (see results/README.md)
figures/           the figures in this README (scripts/make_figures.py)
```

## Reproducing

Requires Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest                    # 96 tests, no model needed

# Recompute the summaries and figures from the published data (overwrites the summary files
# in results/ with identical output)
uv run python scripts/analyze_study1.py results/study1
for f in results/study2/*.parquet; do uv run python scripts/analyze_study2.py "$f"; done
uv run python scripts/findings_numbers.py
uv run --group figures python scripts/make_figures.py
```

Re-running the studies requires [Ollama](https://ollama.com) with the models pulled. On CPU, expect
hours per model.

```bash
uv run python scripts/study1.py --model qwen2.5:14b --seed 0
uv run python scripts/study1.py --model qwen2.5:14b --seed 1 --no-attractor   # seeds 1 and 2
uv run python scripts/study2.py --model qwen2.5:14b
uv run python scripts/study2.py --model qwen2.5:32b --doses 10                # as published
uv run python scripts/run_llm.py --provider ollama --model qwen2.5:14b --cycles 20 --num-ctx 12288
```

Runs are checkpointed. An interrupted `run_llm.py` run can be resumed with `--resume <checkpoint folder>`.
The finished cycles are replayed from the log without model calls.

## License

Code: [MIT](LICENSE). Data in `results/`: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
