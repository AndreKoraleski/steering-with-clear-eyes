# Contributing

## Development Environment

The project is managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync                    # create the environment and install dependencies
uv run pre-commit install  # install the pre-commit and commit-msg hooks
```

## Quality Checks

All checks run automatically on commit and in continuous integration. They can
also be run manually:

```bash
uv run pre-commit run --all-files  # lint, format and type-check
uv run pytest                      # run the test suite
```

Code is formatted and linted with [Ruff](https://docs.astral.sh/ruff/) and
type-checked with [mypy](https://mypy.readthedocs.io/) in strict mode, which
covers `src`, `tests` and `methods`. All public functions must be fully
annotated. Tests mirror the layout of `src/swce`.

Tests never touch the network, except the rehearsal of `tests/models`, which
downloads the processor of LLaVA-1.5 and runs only when `SWCE_NETWORK` is set.
Other tests never do. They download no benchmark and no weights, so
that they run offline and in continuous integration. Build small synthetic
inputs instead, serve files through `file` addresses as `tests/support.py`
does, and replace a model by the tiny stand in defined there.

Docstrings follow the
[Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
with `Args`, `Returns`, `Yields`, `Raises` and `Attributes` sections, which
Ruff checks, and use single backticks for code rather than Sphinx notation.

## Out of the Box

Benchmarks and methods are used as their authors publish them.

- A benchmark is read from the files of its authors and scored by their
  script, or by a port tested against it. Prompts are asked as written.
- A value a paper publishes for a model is used as it is.
- A value the recipe of a method derives is derived by that recipe, implemented
  as the official code implements it, and the step runs before the method is
  evaluated.
- A value that is neither published nor derived is the default argument of the
  official code, under a setting named `code-default`. Nothing is ever tuned
  here.
- Where a benchmark leaves a choice open, the choice is made once, recorded in
  the registry and described in the README.

## Adding a Benchmark

Only benchmarks that anyone can download from files their authors host are
registered. Never commit images or other benchmark content.

1. Add a card to `src/swce/registry/data/registry.json` with its reference,
   source, pinned revision, licence and terms.
2. Pin every file by its size and digest, at an address that names a commit or
   revision whenever the host has one. Compute both from the bytes the address
   serves, never from a clone, since Git can rewrite line endings on checkout.
   `uv run swce-data verify` compares every pin with its host.
3. Add one task per problem, with its kind, scorer, primary metric and token
   budget. Write a loader in `src/swce/tasks/loaders.py` only when no existing
   one reads the files.
4. Regenerate the documentation.

   ```bash
   uv run swce-registry
   ```

A task definition is never changed silently. Its card and pins enter the task
fingerprint, so a change makes every earlier row of the task leave the
leaderboard summary. Change one only on purpose and say so in the pull request.

## Adding a Method

Methods live in [`methods/`](methods/README.md), one file per method with a
`build()` factory, which [`methods/template.py`](methods/template.py) shows.

```bash
uv run swce-evaluate --subject swce.models:llava_15 methods/my_method.py:build --limit 16
```

- The module docstring records the paper, the official code with its commit,
  its licence, and where every value comes from.
- The files a recipe reads are pinned in the registry, like benchmark files.
- `build()` returns one steerer per published setting, named after the
  benchmark it was chosen for.
- `fitted_on` lists the data that built the method. Declaring nothing marks the
  method on the leaderboard.
- The factory raises when its vectors are missing, so that the unmodified
  model is never evaluated under the name of a method.
- A step the official code runs before inference, such as fitting directions
  or calibrating a threshold, is a function of the method file that works on
  any model and writes one file per model under `artifacts/` in the data
  root, which jobs on several machines share.
- `prepare` runs those steps when their outcome is missing, so that one
  command evaluates the method on any model.
- `build()` also returns the defaults of the official code as a setting named
  `code-default` with `fallback=True`, used on models without a published
  setting.
- Every difference from the official code, on every model, is recorded in
  `docs/methods.md`.
- `control()` returns one control per setting of `build()`. A control replaces
  what the method fitted by a random draw, in the own form of the method.
- Vectors and weights never enter git.
- `published_for` names the models the setting is published for. An evaluation
  skips it on any other model.

A port must be numerically equivalent to the official code. The tests in
`tests/methods/test_equivalence.py` compare the interventions of every method
with the classes of its official repository, on random tensors and on a small
model of the LLaVA architecture. They run when `SWCE_OFFICIAL` names a folder
holding clones of the official repositories at the commits the method files
record, and are skipped otherwise, since the test suite never touches the
network.

```bash
SWCE_OFFICIAL=/path/to/official/clones uv run pytest tests/methods
```

A method moves to `src/swce/baselines/` only once it reproduces its paper and
is useful to everyone who compares against it.

## Adding a Metric or a Scorer

A scorer subclasses `Scorer` in `src/swce/evaluation/scorers.py`. It parses
every record once and aggregates any selection of the parsed records, which
the bootstrap resamples. A port of an official script keeps its behaviour,
quirks included, and is tested against it whenever that needs no network.

The CHAIR port can be compared with the script distributed with OPERA on the
captions published with it. This needs the COCO annotations and the NLTK
resources `punkt_tab`, `averaged_perceptron_tagger_eng` and `wordnet`, so it is
not part of the test suite.

## Publishing Results

`results/leaderboard.csv` and `results/leaderboard.md` hold published results
only.

- Publish full evaluations only, never smoke tests, which write to
  `results/smoke/`.
- Never edit the tables by hand. After merging rows from another machine into
  the CSV file, regenerate the summary with `uv run swce results`.
- Record the GPU and the versions of PyTorch and of the model library with
  every run. Generation is seeded and repeatable on one machine, not across
  machines.
- Commit the two tables together, with a message naming the method added.
- Per item records under `results/per_item/` stay out of git.

## Dependencies

Add dependencies with uv so that `uv.lock` stays consistent:

```bash
uv add <package>          # runtime dependency
uv add --dev <package>    # development dependency
```

## Commit Messages

Commit messages must follow the
[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
specification, which is enforced by the `commit-msg` hook:

```text
<type>(<optional scope>): <description>
```

Common types are `feat`, `fix`, `docs`, `refactor`, `test`, `chore` and `ci`.

## Pull Requests

1. Create a branch from `main`.
2. Keep each pull request focused on a single change.
3. Ensure all checks pass before requesting a review.
4. Ask another member of the team to review, and merge only after their
   approval.
