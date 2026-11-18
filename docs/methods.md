# Methods on every model

This page records, for every method and every model, which setting is evaluated, which steps of
the recipe run before it, and what differs from the official code. Three rules decide every entry.

1. A value a paper publishes for a model is used as it is.
2. A value the recipe of a method derives, such as a fitted direction or a calibrated threshold,
   is derived by that recipe, implemented as the official code implements it. The step runs by
   itself before the method is evaluated whenever its outcome is missing.
3. A value that is neither published for a model nor derived by a recipe is the default argument
   of the official code. Such a setting is named `code-default` on the leaderboard and is used
   only on a model for which the method publishes nothing.

A method that cannot be instantiated on a model as its paper describes raises
`InapplicableError`. The evaluation logs it and leaves the cells empty.

## Models

| Model | Decoder blocks | Image tokens | Generation settings shipped with the checkpoint |
|---|---|---|---|
| `llava-1.5-7b` | 32 | 576 for every image | Greedy |
| `qwen3-vl-8b` | 36 | Vary with the image | Sampling at temperature 0.7, top p 0.8 and top k 20 |
| `llava-onevision-1.5-8b` | 36 | Vary with the image | Sampling at temperature 0.000001, close to greedy, with a repetition penalty of 1.05 |

Every model generates with these settings, unchanged, and the random state is seeded before every
generation. Answers of Qwen3-VL therefore depend on the seed, which is zero.

## What differs from the official code on every model

- **Benchmarks.** The official scripts ask their own prompts, for instance "Please help me
  describe the image in detail." for VISTA and " Please answer yes or no." after every POPE
  question for REVIS. Here every method answers the prompts of the benchmark, unchanged, like
  every other method.
- **Model code.** VTI, VISTA and Nullu were written for the original LLaVA code base. Here every
  model is read through `transformers`, with its own prompt template.
- **Hooks.** VTI and VISTA wrap the feed forward modules and Nullu rewrites weights. Here all of
  them are hooks on the same modules, removed when the method detaches. The outputs are equal,
  which `tests/methods/test_equivalence.py` checks against the official classes.
- **Seeds.** Where the official code draws at random without a seed, a seed is set.

## VISTA

Official code at commit `efcf499`. The method needs no data and no step.

| | `llava-1.5-7b` | `qwen3-vl-8b` | `llava-onevision-1.5-8b` |
|---|---|---|---|
| Setting | Published. Strength 0.17 for CHAIR, 0.01 for POPE | `code-default`. Strength 0.1 | `code-default`. Strength 0.1 |
| Logit mixing | Weight 0.3 over blocks 25 to 30 | Weight 0.3 over blocks 25 to 30 | Weight 0.3 over blocks 25 to 30 |
| Prompt without the image | `<image>` removed | The vision span removed | The vision span removed |

- The defaults are those of `chair_eval.py`. The script leaves both the steering and the logit
  mixing off unless asked, and both are on here, since they are the method.
- Blocks 25 to 30 are the last blocks but one of a decoder of 32. On the two models of 36 blocks
  they sit further from the output, and the paper gives no rule to move them.
- MME is not covered by an official script. On LLaVA-1.5 it is run at both published settings.

## VTI

Official code at commit `bb5dfee`. One step, `fit`, builds the directions.

| | `llava-1.5-7b` | `qwen3-vl-8b` | `llava-onevision-1.5-8b` |
|---|---|---|---|
| Setting | Published. Visual 0.2 and textual 0.4 for CHAIR, 0.9 and 0.9 otherwise | `code-default`. Textual 0.8, visual off | `code-default`. Textual 0.8, visual off |
| Visual branch | Fitted on 50 images with 50 masks of 99% of the patches | Inapplicable | Inapplicable |
| Textual branch | Fitted on the same 50 demonstrations | Fitted on 50 demonstrations | Fitted on 50 demonstrations |

- The visual branch fits one direction per image token. The two newer models give every image
  its own number of tokens, so the branch cannot be fitted on them. The default of the official
  script happens to switch it off.
- The demonstrations are drawn with seed 42 from a generator of its own. The official script
  draws them from the global random state after loading the model, so the sample of the authors
  cannot be recovered.
- Three behaviours of the official fitting are kept because they change the directions. They are
  listed in `methods/vti.py`.

## Nullu

Official code at commit `dcb1ef4`. One step, `fit`, builds the subspaces from the 5000 pairs of
the official repository.

| | `llava-1.5-7b` | `qwen3-vl-8b` | `llava-onevision-1.5-8b` |
|---|---|---|---|
| Setting | Published. Rank 4, blocks 16 to 31 | `code-default`. Rank 8, blocks 16 to 31 | `code-default`. Rank 8, blocks 16 to 31 |
| Blocks edited | The upper half | Blocks 16 to 31 of 36 | Blocks 16 to 31 of 36 |
| Conversation averaged over | Template, description and end of sequence token | Chat template with the description as the answer | Chat template with the description as the answer |

- The paper chooses the rank and the blocks by a sweep it does not describe, so on a new model
  the defaults of `model_edit.py` are the only values there are. Its range of blocks was written
  for 32 blocks and leaves the last four of the newer models unedited.
- The paper does not state the number of pairs. The official command uses all of them.

## REVIS

Official code at commit `75f4d29`. Two steps, `compute_vectors` and `calibrate`.

| | `llava-1.5-7b` | `qwen3-vl-8b` | `llava-onevision-1.5-8b` |
|---|---|---|---|
| Setting | Published. Strength 1.1 | `code-default`. Strength 2.5 | `code-default`. Strength 2.5 |
| Vectors | Released by the authors, downloaded | Computed on the 100 official pairs | Computed on the 100 official pairs |
| Prompts of the vectors | Not needed | The construction the official code holds for Qwen3 | The construction the official code uses for any other model |
| Layer and threshold | Calibrated | Calibrated | Calibrated |

- The paper names layer 31 in its Appendix C.3 and reports its main numbers at layer 30 in its
  Table 11 for LLaVA-1.5, and gives no threshold for it. Both therefore come from the calibration
  on every model, with the defaults of `layer_select.sh`, the 85th percentile over 300 questions
  down to layer 10. The paper states 0.8 for the percentile and 100 images.
- The official script answers the calibration questions once per layer. They are answered once
  here and every layer is observed in the same pass, which gives the same risks.
- The calibration answers greedily in at most five tokens, as the official script does, whatever
  the generation settings of the checkpoint.
- The entry point of the official code routes every Qwen checkpoint to the prompts written for
  Qwen2.5, although the code also holds prompts written for Qwen3. The latter are used for
  Qwen3-VL.
- LLaVA-OneVision-1.5 is not among the models of the official code.
- For the descriptions without an image, the official code passes a prompt of LLaVA-1.5 that
  still holds its image placeholder through the processor. Recent versions of the library
  refuse that, where the version the official code was written for tokenized the text. The
  tokenizer is called directly here.
- The official calibration script imports a class that current versions of `transformers`
  removed, so it does not run as distributed with the version pinned here.

## Controls

`control()` in every method file returns one control per setting of `build()`, on every model.
Each replaces what the method fitted by a random draw and keeps the rest of the method.

| Method | Control | Needs the recipe of the method |
|---|---|---|
| VISTA | A random direction per decoder block, the same for every image, in place of the vector of the image. The strength and the logit mixing are kept | No |
| VTI | Random directions of the shape of the fitted ones. The strengths are kept | Yes, for the shapes |
| Nullu | A random subspace of the same rank projected out of the same blocks | No |
| REVIS | A random vector of the norm of the visual vector, through the same layer, threshold and gate | Yes, for the norm and the calibration |

The edits of VISTA and of VTI normalize their vectors, so their controls match the direction
only. The gate of REVIS compares the residual stream with its vector, so under a random vector
the gate opens at other positions than under the visual one. The threshold is kept as calibrated.

## Reading the residual stream

VISTA, the logit lens and the residual shift read the output of every decoder block from the
block itself. The hidden states the library returns end with the state after the final
normalization, which is not the output of the last block. The official code of VISTA hooks the
blocks, and the port reads the same states, which a test checks against the official tracer. The
official code of VTI, Nullu and REVIS fits on the hidden states of the library, and so do their
ports.

## What has been run

The main experiment ran on `llava-1.5-7b` with VTI, REVIS and their controls, on POPE
adversarial, RePOPE, CHAIR and MME, whole, with the logit lens and the residual shift on fifty
images. Its scores are in `results/leaderboard.csv`.

- Environment: one NVIDIA A100-SXM4-80GB, Python 3.13.16, PyTorch 2.14.1+cu130, transformers
  5.23.0, half precision, seed zero, the three jobs side by side on one card.
- The unmodified model lands where other papers put it, with an F1 of 0.802 on POPE adversarial
  and a CHAIR_S of 50.2.
- REVIS answers POPE and RePOPE exactly as the unmodified model does, raises CHAIR_S to 60.2 and
  lowers MME by 302, with longer answers on MME. Its random control answers every task exactly as
  the unmodified model does, since its gate never opens at the calibrated threshold.
- Qwen3-VL and LLaVA-OneVision-1.5 have not been run.

The interventions, the fitting of the VTI directions, the principal direction of REVIS, its
selection of a layer and the states VISTA reads are tested against the official code.

A rehearsal ran the whole evaluation with the real processor of LLaVA-1.5 and a small random
model of its architecture, on POPE, RePOPE, CHAIR and MME, at the defaults of the official code.

- VISTA, VTI and the controls of VISTA, VTI and Nullu ran from their recipes to the scores, and
  each changed the answers of the model.
- The vectors of REVIS were computed and its calibration ran. It found no layer that separates
  false positives from correct answers, as expected of random weights, so REVIS was reported as
  inapplicable and left out.
- The fitting of Nullu was not rehearsed, since it reads 5000 images.
- Qwen3-VL was exercised as a small random model without its processor, and LLaVA-OneVision-1.5
  was not exercised at all, since its model code cannot be built without downloading it.
