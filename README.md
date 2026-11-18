# steering-with-clear-eyes

A benchmark of activation steering methods against object presence hallucination in vision
language models. Published methods are each evaluated under the protocol of their own paper, with
different images, prompts, lengths and parsers, so their numbers cannot be compared. Here a method
is a black box applied to a frozen model, and every method answers the same items through the same
official scorer.

Benchmarks and methods are used as their authors publish them. Nothing is tuned, split or rescored
by the benchmark. Results are read on two axes, the hallucination a method removes and the general
capability it costs, and never as one ranking.

## Installation

The project is managed with [uv](https://docs.astral.sh/uv/). On Windows and Linux, PyTorch is
installed from the CUDA 13.0 wheel index.

```bash
uv sync
```

This installs the command `swce` together with the shortcuts `swce-evaluate`, `swce-data` and
`swce-registry`.

## Quick start

```bash
export SWCE_DATA=/path/to/data   # defaults to ./data
uv run swce-data tasks
uv run swce-evaluate --tasks "pope-*" chair mme --dry-run
uv run swce-evaluate --subject swce.models:llava_15 --tasks pope-adversarial --limit 16
```

`--dry-run` lists the selected tasks and the files they read. `--limit 16` runs a smoke test on
sixteen images per task, written to `results/smoke/` so that it never mixes with real results.
`--resume` reuses the answers already written for the same model, method setting and task
definition.

## Models

A model is a [`Subject`](src/swce/models/subject.py). It answers an image and a prompt and exposes
the blocks of its decoder and of its vision encoder, so that a method never depends on an
architecture. Three are provided, read through the `transformers` library, pinned by revision and
loaded without quantization.

| Factory | Checkpoint | Notes |
|---|---|---|
| `swce.models:llava_15` | `llava-hf/llava-1.5-7b-hf` | The model every method publishes settings for |
| `swce.models:qwen3_vl_8b` | `Qwen/Qwen3-VL-8B-Instruct` | Images yield a varying number of tokens |
| `swce.models:llava_onevision_15_8b` | `lmms-lab/LLaVA-OneVision-1.5-8B-Instruct` | Ships its own model code, which the library runs when loading it |

Every model generates with the settings shipped with its checkpoint, unchanged. Some checkpoints
sample by default, so the random state is seeded before every generation, which makes an answer
repeatable on one machine.

## Benchmarks

Every benchmark, with its licence, the revision its files are pinned at and its tasks, is listed in
[docs/benchmarks.md](docs/benchmarks.md), generated from the versioned registry
[`registry.json`](src/swce/registry/data/registry.json).

| Task | Measures | Scorer |
|---|---|---|
| `pope-random`, `pope-popular`, `pope-adversarial` | object presence, yes or no | the official POPE script |
| `repope-random`, `repope-popular`, `repope-adversarial` | the same questions with corrected labels | the official POPE script |
| `chair` | objects mentioned but absent in a detailed description | the script distributed with OPERA |
| `mme` | general capability, 14 subtasks | the official MME script |

The registry is meant to grow. A benchmark is added as a card, and an evaluation runs only the
tasks it selects.

Files are downloaded from the hosts of their authors and never redistributed. Every file is pinned
by its size and digest, and an evaluation fails when a host serves other bytes. COCO images are
fetched one by one, so the 6.6 GB archive is not needed.

```bash
uv run swce-data benchmarks   # every benchmark with the local state of its files
uv run swce-data fetch pope   # download pinned files ahead of a run
uv run swce-data verify       # check that every host still serves the pinned files
```

Three choices that the benchmarks themselves leave open are fixed in the registry.

- CHAIR predates vision language models and fixes no images, prompt or length. The task uses the
  500 val2014 images of the greedy LLaVA-1.5 run published with OPERA, the prompt "Please describe
  this image in detail." and 512 new tokens.
- POPE and MME fix no length. Their answers are limited to 64 new tokens, which never truncates the
  first sentence their scripts read.
- RePOPE asks the POPE questions again, so its tasks reuse the answers given to POPE within one
  run and cost no generation.

## Methods and reference conditions

A method subclasses [`Steerer`](src/swce/steering/steerer.py) and modifies the model inside a
context. Methods live in [`methods/`](methods/README.md), one file per method with a `build()`
factory returning one steerer per setting.

- When a paper gives different values per benchmark, each is a setting of its own, and MME is run
  at each of them, so that the capability cost is measured where each gain was obtained.
- Nothing is ever tuned here.
- `fitted_on` declares the data that built the method. A method that declares nothing is marked
  `undeclared` on the leaderboard.

Two reference conditions are provided. `swce.baselines:Unmodified` is the model as released.
`swce.baselines:RandomDirection` adds a direction drawn at random with the norm of the method it
controls for, which separates what a direction carries from the effect of perturbing the model.

Four published methods are ported from their official code. Three rules decide what is run.

1. A value a paper publishes for a model is used as it is.
2. A value the recipe of a method derives, such as a fitted direction or a calibrated threshold,
   is derived by that recipe as the official code implements it. The step runs by itself before
   the method is evaluated whenever its outcome is missing.
3. A value that is neither published for a model nor derived by a recipe is the default argument
   of the official code. Such a setting is named `code-default` and is used only on a model for
   which the method publishes nothing.

| Method | Published for LLaVA-1.5 | Steps of its recipe | `code-default` on other models |
|---|---|---|---|
| [VISTA](methods/vista.py) | Strength 0.17 for CHAIR, 0.01 for POPE | None | Strength 0.1 |
| [VTI](methods/vti.py) | Strengths 0.2 and 0.4 for CHAIR, 0.9 and 0.9 otherwise | `fit` builds its directions | Textual strength 0.8, without the visual branch |
| [Nullu](methods/nullu.py) | Rank 4, blocks 16 to 31 | `fit` builds its subspaces | Rank 8, blocks 16 to 31 |
| [REVIS](methods/revis.py) | Strength 1.1 | `compute_vectors` and `calibrate` | Strength 2.5 |

One command evaluates a method on any model. It picks the published settings or the defaults,
fetches the COCO train2014 images the steps need from the official host and runs the steps.

```bash
uv run swce-evaluate --subject swce.models:qwen3_vl_8b swce.baselines:Unmodified methods/revis.py:build
```

[docs/methods.md](docs/methods.md) records for every method and every model the setting, the
steps and every difference from the official code. A result under `code-default` shows what the
official code gives on a model its authors did not cover, and not how good the method can be.

Every method file also has a `control()` factory, which returns one control per setting of
`build()`. A control is the method with what it fitted replaced by a random draw, in the own form
of the method, so a gain the control also obtains does not come from what was fitted.

| Method | Its control replaces | And keeps |
|---|---|---|
| VISTA | the vector of every image by a random direction | the strength and the logit mixing |
| VTI | the fitted directions by random ones of the same shape | the strengths |
| Nullu | the fitted subspace by a random one of the same rank | the blocks |
| REVIS | the visual vector by a random one of the same norm | the layer, the threshold, the strength and the gate |

A method that cannot be instantiated on a model as its paper describes raises `InapplicableError`.
The evaluation logs it and leaves the cells empty. The visual branch of VTI is the one case so
far, on models whose images yield a varying number of tokens.

The interventions are tested for numerical equivalence with the official code. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## Metrics

- **Yes or no tasks:** accuracy, precision, recall, F1 and yes ratio as the official POPE script
  computes them, plus specificity, balanced accuracy, MCC, the sensitivity and criterion of signal
  detection theory, and AUROC when the model exposes the probabilities of yes and no. Sensitivity
  and AUROC do not move when a method only makes the model say yes more or less often.
- **CHAIR:** CHAIR_S, CHAIR_I, recall and caption length, always together, since a caption that
  mentions fewer objects hallucinates less.
- **MME:** the score of every subtask, perception, cognition and the total, plus the four subtasks
  that overlap with object hallucination and the total without them.

Every metric carries a 95% percentile bootstrap interval over images, with the questions of one
image kept together. When the unmodified model answered the same items, every metric also carries
its paired difference from it on the same resamples, the exact McNemar test and the number of items
fixed and broken.

## Results

Scoring writes five files under the results folder.

- `leaderboard.csv` holds one row per model, method setting, task and metric, and
  `leaderboard.md` summarizes the primary metric of every task and is readable on GitHub.
- `analysis.csv` sets the gain of every method on every hallucination task next to its change on
  MME. The gain is signed so that positive is better, with its interval and its McNemar test
  corrected over the methods and tasks of a model with the procedure of Holm. The capability
  change is read on the MME subtasks that do not overlap with object hallucination and tested for
  non inferiority against a margin, two percent of the unmodified score unless `--margin` says
  otherwise. The gain of the control of the method is subtracted when it was evaluated, and a
  setting is listed as dominated only by settings at least as good on both axes.
- `shortcuts.csv` compares every method with what the unmodified model reaches by a trivial
  change of its own answers. On yes or no questions, that is the score at a threshold giving the
  yes ratio of the method. On descriptions, it is the CHAIR of its captions cut to the recall of
  the method. What a method adds beyond the shortcut is the part of its gain the shortcut does
  not explain.
- `lens.csv` summarizes the logit lens readings recorded with the answers.

`per_item/` holds every answer, verbatim, with the probabilities of yes and no, and the software,
the hardware and the generation settings of the run that wrote it. Every file above is computed
from it without running a model. It stays out of git. An interrupted run keeps the answers it
wrote, and `--resume` continues after the last one.

Results are compared only under identical task definitions, tracked by a hash of the task card and
of its pins. Rows obtained before a definition changed stay in the CSV file and leave the summary.

## Diagnostics

Two readings of the internals of a model are available under any method.

- **Logit lens.** The residual stream after every decoder block is read through the final
  normalization and the unembedding, at the position that predicts the answer. It shows at which
  depth the model comes to prefer yes or no, and where a method changes that.
- **Residual shift.** The stream under a method is compared with the stream of the unmodified
  model on the same input, block by block, which shows where an intervention takes effect and
  whether its effect grows or fades toward the output.

The logit lens can be recorded during an evaluation, at no cost, since it reads the pass over the
prompt that generation runs anyway. `--lens` stores the readings with every answer, and scoring
summarizes them per decoder block in `lens.csv`, next to the leaderboard. The readings describe
exactly the runs that were evaluated and are merged with them.

```bash
uv run swce-evaluate --subject swce.models:llava_15 swce.baselines:Unmodified methods/revis.py:build --lens
```

The residual shift needs the steered and the unmodified stream of the same input, so it is a
command of its own. `swce-diagnose` reads both diagnostics on the items of a task without
generating an answer, and writes one file per method setting under `diagnostics` in the results
folder. Folders written by several jobs are copied into one, and `--summarize-only` summarizes
everything found there.

```bash
uv run swce-diagnose --subject swce.models:llava_15 methods/revis.py:build methods/revis.py:control --task pope-adversarial --limit 50
uv run swce-diagnose --summarize-only
```

## Running on several machines

An evaluation has three stages, which run together by default and apart on request.

| Stage | Flag | Needs a model | Writes |
|---|---|---|---|
| Recipes of the methods | `--prepare-only` | yes | `artifacts/` under the data root |
| Answers | `--generate-only` | yes | `per_item/` under the results folder |
| Scores | `--score-only` | no | the leaderboard of the results folder |

Jobs that generate answers are independent. Each writes to its own results folder, the folders
are copied into one, and a single scoring pass compares every method with the unmodified answers
found there. What a recipe produces lives under the data root, so jobs sharing that folder share
the fitted directions and calibrations. [docs/runbook.md](docs/runbook.md) gives the commands of
the proof of concept.

## Limitations

- The CHAIR port was compared with the script distributed with OPERA. See
  [CONTRIBUTING.md](CONTRIBUTING.md) for how to repeat the comparison.
- COCO publishes no checksum of single images, so they are the only files that are not pinned.
- The official POPE parser reads every answer without `No`, `not` or `no` as yes, including an
  empty one. It is kept, and the yes ratio makes its effect visible.
- MME has between 20 and 200 images per subtask, so its intervals are wide.
- A rehearsal runs the whole evaluation on the CPU with the real processor of LLaVA-1.5
  and a small random model of its architecture, `swce.models:llava_15_rehearsal`. It exercises
  the prompts, the tokens, the hooks, the recipes and the scoring, and says nothing about
  results.
- Only LLaVA-1.5-7B has been run on real weights, with VTI, REVIS and their controls. Qwen3-VL
  and LLaVA-OneVision-1.5 are tested only as small models with random weights, and the latter
  cannot be built without downloading its code. [docs/methods.md](docs/methods.md) lists what has
  run.
- Equivalence with the official code covers the interventions, the fitting of the VTI directions
  and the principal direction and layer selection of REVIS. It does not cover the prompts the
  fitting steps build, nor a whole evaluation, which also depends on the prompt template and on
  the generation settings of the checkpoint.
- The sparse autoencoder analysis of the project is not part of this repository yet.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup and for adding benchmarks,
methods and metrics.

## Citation

If you use this work, please cite it using the metadata in [CITATION.cff](CITATION.cff), together
with the benchmarks and methods you evaluate.

## License

Released under the [MIT License](LICENSE).
