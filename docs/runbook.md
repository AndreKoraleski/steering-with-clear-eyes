# Runbook of the proof of concept

The proof of concept evaluates three methods on LLaVA-1.5 and one recipe on Qwen3-VL. Its jobs
are independent and can run on separate machines, or as separate processes on one large card.

Every job reads and writes under one data root and writes its answers to its own results folder.

```bash
export SWCE_DATA=/path/to/shared/data
```

The data root holds the benchmark files, the images and, under `artifacts`, what the recipes of
the methods produce. Sharing it between jobs, or copying it, is what lets one job use a
calibration another job ran.

## 0. Rehearsal, without an accelerator

Before renting anything, run the whole pipeline on the CPU. The rehearsal subject pairs the
real processor of LLaVA-1.5 with a small random model of its architecture, so every prompt, token
lookup, hook, recipe and scorer takes the path it takes on the real model. Its answers are
nonsense and its results mean nothing.

```bash
uv run swce-evaluate --subject swce.models:llava_15_rehearsal swce.baselines:Unmodified methods/vista.py:build methods/vista.py:control methods/revis.py:build methods/revis.py:control methods/vti.py:build methods/vti.py:control methods/nullu.py:control --tasks pope-adversarial repope-adversarial chair mme --limit 2 --lens --results results/rehearsal
```

Since no setting is published for the rehearsal subject, every method runs at the defaults of
its official code. Nullu itself is left out because building its subspaces reads 5000 images.

## 1. Smoke test

One machine, a few minutes. Nothing else is worth starting before this passes.

```bash
uv run swce-evaluate --subject swce.models:llava_15 --tasks pope-adversarial chair mme --limit 16
```

## The main experiment in one command

`experiments/main.py` runs the experiment the sections below spell out, on one model. It
evaluates VTI and REVIS, each with its random control, next to the unmodified model, on POPE
adversarial with its RePOPE labels, on CHAIR and on MME, records the logit lens, measures the
residual shift on fifty images and scores everything. It continues where an earlier run stopped.

```bash
uv run python experiments/main.py --model llava_15 --limit 16   # smoke test
uv run python experiments/main.py --model llava_15
```

Its three jobs can run side by side on one results folder, scored once at the end.

```bash
uv run python experiments/main.py --model llava_15 --jobs baseline --no-score
uv run python experiments/main.py --model llava_15 --jobs vti --no-score
uv run python experiments/main.py --model llava_15 --jobs revis --no-score
uv run python experiments/main.py --model llava_15 --score-only
```

`--model` also takes `qwen3_vl_8b`, `llava_onevision_15_8b` and `llava_15_rehearsal`.

## 2. Jobs

Each line is one job. `--generate-only` writes answers and scores nothing, and `--lens` records
the logit lens of every prompt with its answer, at no cost.

```bash
# A. The unmodified model, which every other job is compared with.
uv run swce-evaluate --subject swce.models:llava_15 --tasks pope-adversarial repope-adversarial chair mme --generate-only --lens --results results/a

# B. VISTA and its random control, which need no recipe.
uv run swce-evaluate --subject swce.models:llava_15 methods/vista.py:build methods/vista.py:control --tasks pope-adversarial repope-adversarial chair mme --generate-only --lens --results results/b

# C. REVIS and its random control. The calibration runs first, by itself.
uv run swce-evaluate --subject swce.models:llava_15 methods/revis.py:build methods/revis.py:control --tasks pope-adversarial repope-adversarial chair mme --generate-only --lens --results results/c

# D. VTI and its random control. Its directions are fitted first, by themselves.
uv run swce-evaluate --subject swce.models:llava_15 methods/vti.py:build methods/vti.py:control --tasks pope-adversarial repope-adversarial chair mme --generate-only --lens --results results/d

# E. REVIS on a model its authors did not cover, with its own unmodified answers.
uv run swce-evaluate --subject swce.models:qwen3_vl_8b swce.baselines:Unmodified methods/revis.py:build --tasks pope-adversarial repope-adversarial mme --generate-only --lens --results results/e
```

Jobs C and D run a method and its control together because the control reuses what the recipe
of the method produced. To run a recipe ahead of time, on a machine of its own, add `--prepare-only` to the
command of the job. The outcome lands under the data root.

A job that stops early keeps the answers it wrote. Run the same command again with `--resume`
and it continues after the last one.

CHAIR is the slow task. A job can be split further by giving `chair` to one process and the
other tasks to another, as long as both use the same data root.

## 3. Scoring

Copy the `per_item` folder of every job into one results folder and score it. No model runs, and
every method is compared with the unmodified answers of job A.

```bash
mkdir -p results/per_item
cp -r results/{a,b,c,d,e}/per_item/. results/per_item/
uv run swce-evaluate --score-only
```

The leaderboard is written to `results/leaderboard.csv`, the gains next to their capability
change to `results/analysis.csv`, the comparison with the shortcuts of the unmodified model to
`results/shortcuts.csv` and the summary of the logit lens to `results/lens.csv`.

Scoring CHAIR needs the COCO annotations, which are fetched into the data root, and the NLTK
resources `punkt_tab`, `averaged_perceptron_tagger_eng` and `wordnet`, which are downloaded on
first use when missing.

## 4. Residual shift

Optional, and independent of the jobs above. It measures, block by block, how far each method
moves the residual stream from that of the unmodified model, on fifty images, and generates no
answer. It can run as one job per method, each with its own results folder, like the jobs above.

```bash
uv run swce-diagnose --subject swce.models:llava_15 methods/vista.py:build methods/revis.py:build methods/vti.py:build --task pope-adversarial --limit 50
```

After copying the `diagnostics` folders of several jobs into one results folder, summarize them.

```bash
uv run swce-diagnose --summarize-only
```

## What to check first

- The unmodified LLaVA-1.5 should land where other papers put it. Four recent papers report
  between 75.6 and 83.6 accuracy on POPE adversarial and between 46.0 and 54.0 CHAIR_S, under
  protocols that differ from this one.
- Every job on LLaVA-1.5 should use the same kind of card. Greedy decoding is repeatable on one
  kind of hardware and not across kinds.
- Qwen3-VL samples by default, so its answers depend on the seed.
