# Methods

One file per steering method, each with a `build()` factory returning one steerer per published
setting. Copy [`template.py`](template.py), rename it and fill it in.

```bash
uv run swce-evaluate --subject swce.models:llava_15 methods/my_method.py:build --limit 16  # smoke test
uv run swce-evaluate --subject swce.models:llava_15 methods/my_method.py:build
```

Several methods can be evaluated in one run. Add `swce.baselines:Unmodified` to the same run, or
run it before, so that every method is compared with the unmodified model on the same items.

## Rules

- Hyperparameters are those the paper publishes for the model and the benchmark, with their source
  recorded in the module docstring. A value the recipe derives is derived by the recipe, in
  `prepare`. Any other value is the default of the official code, under a `code-default` setting
  with `fallback=True`. Nothing is tuned.
- A paper that publishes different values per benchmark yields one setting per benchmark, named
  after it. MME is run at every setting.
- Names are unique across the team, since they key the leaderboard.
- `fitted_on` lists the data that built the method. A method fitted on evaluation images is
  declared as such and not altered.
- The factory raises when its vectors are missing, so that the unmodified model is never evaluated
  under the name of a method.
- What a recipe produces is written under `artifacts/` in the data root, one file per method
  and model, and never enters git. Record in the docstring how it is built and from which code,
  with its commit and licence.
