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

### Loading LLaVA-OneVision-1.5

The checkpoint ships model code written for `transformers` 4.53, which the pinned version of the
library runs unchanged. Its first loads found six differences, each bridged in `swce.models.hf`
so that the code behaves as it did under 4.53. Two of them loaded the model without any error
and left it answering nonsense, so loading now fails unless every weight is read from the
checkpoint and every buffer is finite.

| Found on the first loads | Bridge |
|---|---|
| The code renames the weights of its checkpoint in `_checkpoint_conversion_mapping`, which the library now ignores, so the language model was left at random weights | The same renaming is given as `key_mapping` |
| The library builds the model on the meta device and fills only what the checkpoint holds, so the rotary frequencies, buffers the constructors compute and the checkpoint does not save, were left uninitialized | Each rotary embedding is built again by its own class and its frequencies copied in |
| The processor builds a video processor, which needs `torchvision` | `torchvision` is a dependency |
| The language configuration has no `pad_token_id`, which the code reads | Set to None, its value under 4.53 |
| The code builds its rotary embedding from the entry `default` of the library, which is gone | The computation of 4.53 is registered under that entry while the model is built, and removed afterwards |
| The processor returns `mm_token_type_ids`, which the forward pass does not take | Inputs a forward pass does not name are left out, on any model whose forward pass names its arguments |

The library passes `cache_position`, which the code reads, only to classes it marks as remote
code, and warns that it will stop passing it in a future version. The class is fetched ahead of
loading, to read its renaming, and marked by hand.

On the first 40 questions of POPE adversarial, the model loaded this way gives the answers it gave
under `transformers` 4.53 and PyTorch 2.5, all 40. Long descriptions part after a few dozen words,
each time at a near tie of at most 0.25 logit, the step of bfloat16 at that magnitude.

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
| Setting | Published. Strength 1.1 at the calibrated block, and at block 30 as `table-11` | `code-default`. Strength 2.5 | `code-default`. Strength 2.5 |
| Vectors | Released by the authors, downloaded | Computed on the 100 official pairs | Computed on the 100 official pairs |
| Prompts of the vectors | Not needed | The construction the official code holds for Qwen3 | The construction the official code uses for any other model |
| Layer and threshold | Calibrated, or block 30 with its calibrated threshold | Calibrated | Calibrated |

- The paper names layer 31 in its Appendix C.3 and reports its main numbers at layer 30 in its
  Table 11 for LLaVA-1.5, and gives no threshold for either. When a paper publishes two values
  for one model, both are evaluated as settings of their own. The default setting takes its
  layer from the calibration, which selects block 31, and `table-11` is fixed at block 30.
- The threshold comes from the calibration on every model, with the defaults of
  `layer_select.sh`, the 85th percentile over 300 questions down to layer 10. The paper states
  0.8 for the percentile and 100 images. The calibration records the threshold of every block it
  observes, as the official script computes it for each, so a setting fixed at a block reads the
  threshold of that block.
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
- The unmodified model lands where other papers put it on CHAIR and MME, with a CHAIR_S of 50.2
  and an MME of 1769. On POPE adversarial it reaches an F1 of 0.802 but answers yes to 61.7% of
  the questions, more than in other papers, as the comparison below shows.
- REVIS answers POPE and RePOPE exactly as the unmodified model does, raises CHAIR_S to 60.2 and
  lowers MME by 302, with longer answers on MME. Its random control answers every task exactly as
  the unmodified model does, since its gate never opens at the calibrated threshold. The causes
  are below.
- REVIS at block 30, the `table-11` setting, lowers CHAIR_S to 32.0 and raises MME by 43, and
  lowers the F1 to 0.787 on POPE adversarial and to 0.770 on RePOPE. It ran later on the same
  kind of card, after a calibration that gave block 31 the same threshold to the last digit.
- LLaVA-OneVision-1.5-8B ran the same experiment at the `code-default` settings, below.
- Qwen3-VL has not been run.

### Why REVIS behaves so on LLaVA-1.5-7B

The port follows the official code at every step. The vectors are the ones the authors release,
the calibration selects the layer and threshold as `layer_select.py` does, and the hook is the
official one. The behaviour comes from what the recipe selects and from the protocol.

- **The calibration selects the last block.** It searches downward from the top and stops at the
  first block that separates false positives from correct answers. Block 31 already does, with a
  threshold of 0.195. The paper names the same block in its Appendix C.3, although its own Table
  11 reports that block as failing, with a CHAIR_S of 51.0, a CHAIR_I of 47.84 and an MM-Vet of
  12.29 against 35.92 unsteered. Nothing comes after the
  last block but the final normalization and the output head. Adding the vector there changes only
  the next token at the positions where the gate opens. It reaches no later position, so it acts
  as a bias on the logits.
- **The vector of that block points against the vectors below it.** Block 31 reads the vector at
  hidden state index 32, which the official code computes from the states after the final
  normalization. Each vector is a principal component, and the sign of a principal component is
  arbitrary. Every released visual vector has a positive cosine with the one of the index below,
  above 0.9 from index 18 upward, except at indices 13, 25 and 32. There it is -0.85, -0.94 and
  -0.75, so their sign is flipped. Index 31, which the paper uses at block 30 in its Table 11, is
  on the other side of the last flip.
- **The gate opens on MME and not at the first answer token of POPE.** The calibration measures
  the risk at the last token of questions ending in "Please answer yes or no.", as the official
  scripts ask them. MME questions end the same way, and on 1377 of its 2374 questions the gate
  opens at that token. There it pushes the answer to "no", correcting 111 wrong yeses and breaking
  365 right ones. POPE asks its questions unchanged here, without that sentence. On none of the
  3000 questions does the probability of the first answer token change, so no answer of POPE or
  RePOPE changes.
- **The gate also opens on later tokens and keeps the model talking.** The official scripts stop
  POPE and MME answers at ten tokens. Here every model generates with its shipped settings. Steered
  answers average 40 tokens on POPE against 23 unsteered, and 47 on MME against 3. CHAIR
  descriptions grow from 107 to 134 tokens and diverge after about 30 characters. The longer
  descriptions name more objects that are not in the image, which raises CHAIR_S.
- **The gate of the control cannot open.** For a random vector, the cosine with any state has a
  standard deviation of 1 over the square root of 4096, which is 0.016. The threshold of 0.195 is
  more than twelve deviations away, so the control never adds its vector.

### REVIS at block 30

- **The gate is open almost everywhere.** The calibration gives block 30 a threshold of -0.093,
  so the gate opens wherever the cosine between the state and the vector is below 0.093. It opens
  at the first answer token of every POPE question, for the method and for its control alike, so
  at this block REVIS adds its vector nearly without condition.
- **The method changes answers while keeping the text fluent.** It flips 218 of the 3000 POPE
  answers and loses F1 on both label sets. Its descriptions keep their usual length, 109 tokens
  against 107, and name fewer objects that are not in the image. Fourteen of them run to the
  token limit in a loop of repeated words.
- **The control breaks the text, which flatters its CHAIR.** The random vector of the norm of the
  visual vector reaches a CHAIR_S of 25.4, below the method, but 181 of its 500 descriptions run to
  the token limit and end in a loop of repeated characters or words. A description that names no
  object hallucinates none, so its CHAIR is not comparable with that of the method. On POPE it
  flips 141 answers and on MME it scores as the unmodified model does.

### Comparison with published results

The protocols differ from paper to paper, so the comparison is of direction and size, not of the
last digit. The REVIS paper evaluates CHAIR on 100 images, against 500 here.

| | Here | VTI paper | REVIS paper | Others |
|---|---|---|---|---|
| Unmodified CHAIR_S and CHAIR_I | 50.2 and 14.1 | 51.0 and 15.2 | 54.0 and 15.78 | 50.0 and 15.4 (V-ITI), 48.6 and 13.6 (RUDDER) |
| Unmodified MME | 1769 | Not reported | 1754 | 1865 (V-ITI), 1746 (RUDDER) |
| Unmodified POPE adversarial, accuracy and F1 | 77.9 and 80.2 | 76.03 and 76.45 | 83.20 and 82.32 | 76.03 and 76.26 (V-ITI) |
| VTI CHAIR_S and CHAIR_I | 36.6 and 11.1 | 35.8 and 11.1 | 35.0 and 15.58 | |
| VTI POPE adversarial F1 | 80.7, +0.5 | 82.11, +5.7 | 80.91, -1.4 | |
| VTI MME | 1553 at the POPE setting, -133 | Not reported | 1507, -247 | |
| REVIS block 31 CHAIR_S | 60.2 | | 51.0 | |
| REVIS block 30 CHAIR_S | 32.0 | | 30.0 | |
| REVIS block 30 MME | 1812, +43 | | 1787, +33 | |
| REVIS block 30 POPE adversarial F1 | 78.7, -1.5 | | 83.51, +1.2 | |

The V-ITI paper is arXiv 2512.03542 and the RUDDER paper arXiv 2511.10292. Neither evaluates VTI
or REVIS.

- **CHAIR and MME agree.** The unmodified model, VTI on CHAIR, both blocks of REVIS on CHAIR and
  the change REVIS makes to MME at block 30 all land within the spread of the published values.
- **The unmodified model answers yes more often on POPE.** Here it says yes to 61.7% of the
  adversarial questions, with a recall of 89.6 and a precision of 72.7, where the REVIS paper
  reports a recall of 78.2 for the same model and split. The REVIS and V-ITI papers append
  "Please answer yes or no." to every question, and the questions are asked unchanged here, which
  is a likely cause. The VTI paper also asks them unchanged, so the cause is not established.
- **This bias decides the sign of a POPE change.** REVIS at block 30 pushes answers to yes in both
  evaluations, raising the yes ratio here from 0.617 to 0.689. A model that says no too often
  gains from that, as in the REVIS paper, and one that says yes too often loses, as here. The
  comparison with the shortcuts of the unmodified model finds no gain beyond the shift of the
  threshold, so the gain the paper reports is likely a shift of the threshold as well. A POPE
  result is only comparable across protocols through its threshold-free metrics, the AUROC and
  d', which `results/leaderboard.csv` reports.
- **VTI on POPE matches the independent evaluation and not its own paper.** The VTI paper
  reports a gain of 5.7 F1 points against a baseline its paper decodes with beam search. The REVIS
  paper finds a loss of 1.4, and here the gain of 0.5 is that of the random control.
- **VTI shortens descriptions more here than in its paper.** Its paper reports the recall rising
  from 75.2 to 76.8 with descriptions 8% shorter. Here the recall falls from 80.8 to 75.9 and the
  descriptions shorten by 22%, which is why most of the gain on CHAIR is matched by truncating
  the unmodified descriptions. The cause of the difference is not known, and the finding holds
  for this protocol.

### POPE without the threshold

F1 mixes how well a model tells present from absent objects with how readily it says yes. The
rows of `results/leaderboard.csv` also hold the yes ratio, d' and the criterion c of signal
detection theory, computed from the same answers, and the AUROC of the margin of the first answer
token, which needs no threshold at all. A positive c is a model that says no more readily. Changes
are against the unmodified model, with their 95% bootstrap intervals over images.

| POPE adversarial | F1 | Yes ratio | c | d' | AUROC |
|---|---|---|---|---|---|
| LLaVA-1.5-7B, unmodified | 0.802 | 0.617 | -0.42 | 1.68 | 0.907 |
| VTI, POPE setting | +0.005 [-0.003, +0.013] | -0.037 | +0.13 | +0.02 [-0.06, +0.08] | -0.009 [-0.013, -0.004] |
| VTI control | +0.006 [-0.001, +0.011] | -0.027 | +0.09 | +0.03 [-0.03, +0.08] | -0.000 [-0.002, +0.002] |
| REVIS, block 30 | -0.015 [-0.023, -0.007] | +0.073 | -0.27 | -0.01 [-0.09, +0.07] | -0.000 |
| REVIS control, block 30 | -0.008 [-0.014, -0.001] | +0.047 | -0.18 | +0.00 [-0.06, +0.07] | +0.000 |
| OneVision-1.5-8B, unmodified | 0.866 | 0.462 | +0.18 | 2.30 | 0.942 |
| VTI | -0.003 [-0.009, +0.004] | +0.026 | -0.13 | -0.09 [-0.16, -0.04] | -0.001 [-0.003, +0.001] |
| VTI control | +0.003 [-0.001, +0.008] | +0.012 | -0.06 | -0.01 [-0.05, +0.04] | -0.001 [-0.002, +0.000] |
| REVIS | +0.001 [-0.003, +0.005] | -0.001 | +0.01 | +0.02 [-0.02, +0.05] | +0.000 [-0.001, +0.002] |
| REVIS control | -0.003 [-0.010, +0.003] | -0.020 | +0.11 | +0.05 [-0.01, +0.12] | -0.001 [-0.003, +0.001] |

REVIS at block 31 of LLaVA-1.5-7B and its control answer POPE exactly as the unmodified model
does, so they are left out. Every change of the yes ratio and of c in the table excludes zero
except those of REVIS on OneVision.

- **Every change of F1 on POPE is a move of the threshold.** On both models and both label sets
  no change of d' excludes zero but that of VTI on OneVision POPE, and the changes of F1 follow
  the yes ratio. REVIS at block 30 loses F1 on LLaVA-1.5-7B by saying yes more, and its random
  control makes about two thirds of that shift. VTI gains on LLaVA-1.5-7B by saying no more,
  and its control gains as much.
- **The one change that survives without the threshold is a loss, by VTI on both models.** On
  LLaVA-1.5-7B VTI lowers the AUROC by 0.009 on POPE and 0.011 on RePOPE where its control
  changes it by 0.002 at most, so the margin of its first token ranks present and absent objects
  slightly worse. On OneVision it lowers d' by 0.09, while the AUROC stays where it was, so the
  ranking of the margins holds and the answers it gives separate slightly less.
- **VTI moves the threshold the other way on each model.** It raises c on LLaVA-1.5-7B, which
  says yes too often, and lowers it on OneVision, which says yes to fewer than half of the
  questions, so whether it gains or loses F1 on POPE depends on where the model starts.
- RePOPE gives the same picture. VTI lowers the AUROC by 0.011 on LLaVA-1.5-7B and by 0.001 on
  OneVision, every other change of the AUROC is 0.003 or less, and no change of d' excludes
  zero.

### Linear probes on POPE adversarial

`swce-diagnose --probe` fitted a probe of the expected answer on every block, on the 3000
questions of POPE adversarial, under the unmodified model, VTI at its POPE setting, REVIS at both
blocks and their controls. The probes were cross-validated over images with five folds.

| Block | 0 | 4 | 6 | 8 | 10 | 11 | 13 | 16 | 24 | 31 |
|---|---|---|---|---|---|---|---|---|---|---|
| Probe of the unmodified model, AUROC | 0.639 | 0.675 | 0.749 | 0.853 | 0.898 | 0.905 | 0.908 | 0.906 | 0.905 | 0.905 |
| Logit lens of the unmodified model, AUROC | 0.521 | 0.478 | 0.488 | 0.444 | 0.497 | 0.600 | 0.886 | 0.910 | 0.906 | 0.907 |

- **The answer is readable four blocks before the logit lens shows it.** The probe reaches 0.85
  at block 8 and 0.90 at block 10, where the lens is still at chance. Blocks 6 to 11 are where
  the stream comes to hold the answer, and the lens only reads it once the head can, from block
  12.
- **Nothing is lost after block 11.** The probe stays at 0.90 to 0.91 from block 11 to the
  output, the same as the final answer of the model reaches. The errors of the unmodified model
  on POPE are not information that a later block discards.
- **No method adds what the stream holds about the answer.** A probe fitted under REVIS, at
  either block, or under its control, matches the unmodified probe to the third decimal. Under
  VTI it is lower between blocks 8 and 14, 0.832 against 0.853 at block 8 and 0.888 against 0.905
  at block 11, and the same from block 16 on.
- **VTI moves the early stream far from where the unmodified probe reads it.** The probe of the
  unmodified model, applied under VTI, falls to 0.51 to 0.53 between blocks 4 and 8, and its
  score moves by up to nine standard deviations, with a sign that changes from block to block.
  Its random control does the same, less, falling to 0.52 to 0.60. A probe fitted under VTI still
  finds the answer there, so VTI displaces the early stream rather than erasing what it holds.
- **REVIS at block 30 displaces the stream just as its random control does.** The unmodified
  probe falls to 0.65 at block 30 under REVIS and to 0.67 under the control, while a probe fitted
  under either still reaches 0.90. At block 31, REVIS changes nothing at the last prompt token,
  as its gate stays shut there.

### Linear probes on LLaVA-OneVision-1.5-8B

The same probes on `llava-onevision-1.5-8b`, on the 3000 questions of POPE adversarial, under the
unmodified model, VTI and REVIS at their `code-default` settings and their controls, with
`swce-diagnose --probe` on one RTX 4090. The model and the probes of one fold in double precision
do not fit together on that card, so the model was moved off it once the states were read, which
changes no number.

| Block | 0 | 4 | 6 | 8 | 10 | 12 | 14 | 17 | 18 | 19 | 24 | 35 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Probe of the unmodified model, AUROC | 0.636 | 0.692 | 0.753 | 0.788 | 0.850 | 0.883 | 0.925 | 0.939 | 0.937 | 0.941 | 0.946 | 0.938 |
| Probe of the unmodified model under VTI | 0.576 | 0.597 | 0.649 | 0.577 | 0.626 | 0.628 | 0.745 | 0.810 | 0.858 | 0.923 | 0.896 | 0.840 |
| Under the VTI control | 0.543 | 0.584 | 0.665 | 0.670 | 0.694 | 0.781 | 0.827 | 0.843 | 0.832 | 0.908 | 0.938 | 0.916 |
| Under REVIS | 0.636 | 0.692 | 0.753 | 0.788 | 0.850 | 0.883 | 0.925 | 0.939 | 0.937 | 0.941 | 0.718 | 0.758 |
| Under the REVIS control | 0.636 | 0.692 | 0.753 | 0.788 | 0.850 | 0.883 | 0.925 | 0.939 | 0.937 | 0.941 | 0.649 | 0.855 |

- **The answer is readable about eight blocks before the logit lens shows it.** The probe reaches
  0.85 at block 10 and 0.93 at block 14, where the lens is still near chance, and the lens only
  reads it from block 18. On LLaVA-1.5-7B the gap is about four blocks, 0.85 at block 8 against a
  lens readable from block 12, so the answer forms earlier in the stream of OneVision, relative to
  where the head can read it, than in that of LLaVA-1.5-7B.
- **It forms over the same share of the model.** The probe passes 0.85 at block 10 of 36 on
  OneVision and block 8 of 32 on LLaVA-1.5-7B, and reaches its plateau at block 17 against block
  11. Activation patching puts the answer at its own position from block 20 on OneVision and
  block 13 on LLaVA-1.5-7B, so on both models the probe reads the answer at the last token a few
  blocks before that position is what decides it.
- **Nothing is lost after block 17.** The probe stays at 0.94 to 0.95 from block 17 to the
  output, the AUROC of the final answer of the model, as on LLaVA-1.5-7B.
- **No method adds what the stream holds about the answer.** A probe fitted under REVIS or its
  control matches the unmodified probe to the third decimal up to block 20 and is within 0.007 of
  it after. Under VTI it is lower by up to 0.020 between blocks 11 and 13 and within 0.014
  elsewhere, as on LLaVA-1.5-7B.
- **VTI displaces the stream at every block below 19.** The unmodified probe applied under VTI
  falls to 0.55 to 0.65 between blocks 0 and 13 and recovers to 0.92 at blocks 19 and 20, and its
  score moves by up to 5.6 standard deviations, with a sign that changes from block to block. Its
  control falls to 0.54 to 0.70 below block 11 and recovers sooner. A probe fitted under VTI still
  finds the answer, so VTI moves the stream without erasing the answer, as on LLaVA-1.5-7B, over
  more blocks.
- **REVIS at block 21 displaces the stream as its control does, and less.** Both leave every
  block below 21 as it was. From block 21 the unmodified probe falls to 0.56 to 0.80 under REVIS
  and to 0.52 to 0.86 under the control, and its score moves toward no by up to 2.3 standard
  deviations under REVIS and 3.5 under the control. A probe fitted under either still reaches
  0.94, so the answer moves with the stream and stays as readable. The block 30 setting of
  LLaVA-1.5-7B does the same. It fits the answers: REVIS changes 28 of the 3000 answers of POPE
  and its control 83.

### LLaVA-OneVision-1.5-8B

The main experiment ran on `llava-onevision-1.5-8b` with VTI and REVIS at their `code-default`
settings and their controls, since neither paper publishes a setting for this model. Its scores
are in `results/leaderboard.csv`.

- Environment: one NVIDIA RTX 4090, PyTorch 2.14.1+cu126 and torchvision 0.29.1+cu126, seed
  zero. The driver of that machine, version 550, cannot run the build for CUDA 13.0 that the lock
  file pins, which is the one departure from the locked environment. Every `setting.json` of the
  run records it.
- The unmodified model reaches an F1 of 0.866 on POPE adversarial, answering yes to 46.2% of the
  questions, with an AUROC of 0.942 and a d' of 2.29. It reaches 0.914 on RePOPE, a CHAIR_S of
  33.4 with descriptions of 169 words, and an MME of 2280.
- **No method changes POPE.** No F1 moves by more than 0.006 on either label set, and no change
  is significant after the Holm correction, p of 0.27 or more. The AUROC stays between 0.940 and
  0.942 under every condition. VTI lowers d' from 2.29 to 2.20, by 0.09 [0.04, 0.16], and raises
  the yes ratio by 0.026, so it moves the threshold toward yes and separates the answers slightly
  less.
- **REVIS lowers CHAIR_S by shortening the descriptions.** It reaches 25.6, a change of -7.8
  [-12.2, -3.4] points with a Holm p of 0.012, with descriptions of 118 words against 169. Cut to
  the same recall, the unmodified descriptions reach 27.5, so only 1.9 points of the change are
  beyond that shortcut. Its MME changes by -15 [-39, +3].
- **The REVIS control does not raise CHAIR_S significantly.** Its change is +5.0 [-0.2, +10.6]
  points with a Holm p of 0.73, and its descriptions run to 381 words, more than twice the
  unmodified length. They reach the token limit without looping, as checked below, so its CHAIR
  and the difference of 12.8 points between REVIS and its control stand.
- **VTI does not help.** Its CHAIR_S is 34.4, within the interval of the unmodified model, with
  descriptions of 304 words, and its MME falls by 51 [2, 100]. That CHAIR_S is not comparable,
  since 131 of its descriptions break into loops or single words, and neither is the difference
  of 3.0 points between VTI and its control. Its control changes CHAIR_S by +4.0 points, not
  significantly, at 195 words.
- No method is non-inferior to the unmodified model on MME.
- **The logit lens separates the questions whose answer is yes from those whose answer is no.**
  On POPE adversarial its AUROC stays near 0.5 up to block 17, rises to 0.87 at block 18 and 0.93
  at block 19, and stays near 0.94 to the output, under every method. It reads the expected answer
  and not whether the answer of the model is right.
- **REVIS changes the stream from block 21 on**, the block its calibration selected, and leaves
  the blocks below it as they were. Why its gate opens there without moving POPE is below.

Lengths are in words, as `results/leaderboard.csv` reports them.

### Loops at the token limit on LLaVA-OneVision-1.5-8B

Every CHAIR description of the five conditions was sorted by its last 300 characters. A
description loops when those characters repeat a period of at most 80 characters at 85% of
their positions, or when it ends in a sequence of at most 20 words repeated three times or more.
It is fragmented when its lines hold at most two words, or when letters make up less than 35% of
those characters. CHAIR is then computed again without those descriptions, on each condition
and on the 369 images where no condition loops or fragments.

| Condition | At the limit of 512 tokens | Loops | Fragmented | CHAIR_S, all 500 | CHAIR_S, 369 images | Change from unmodified on the 369 |
|---|---|---|---|---|---|---|
| Unmodified | 0 | 0 | 0 | 33.4 | 33.1 | |
| VTI | 299 | 34 | 97 | 34.4 | 37.9 | +4.8 [-0.5, +10.3] |
| VTI control | 2 | 0 | 0 | 37.4 | 37.4 | +4.3 [-0.5, +9.8] |
| REVIS | 0 | 0 | 0 | 25.6 | 25.5 | -7.6 [-12.7, -2.4] |
| REVIS control | 186 | 0 | 0 | 38.4 | 37.7 | +4.6 [-1.4, +10.8] |

Intervals are bootstrap intervals over the images, paired with the unmodified model.

- **Only VTI loops, and its loops flatter its CHAIR.** Its 34 loops repeat a numbered list or an
  empty bullet, such as "44. A container." up to the limit, and its 97 fragmented descriptions
  put one or two words on each line. Without them its CHAIR_S rises from
  34.4 to 37.9. Its CHAIR row on the leaderboard, and the `beyond_control` of VTI on CHAIR, which
  compares it with its control, are not comparable.
- **The REVIS control reaches the limit without looping.** Its 186 long descriptions keep fluent
  sentences but leave the image and continue as essays, for instance "the image serves as a
  reminder of the power of photography". Its CHAIR_S moves by less than a point without them, so
  it is confirmed, and so is the `beyond_control` of REVIS on CHAIR, 12.8 points. It compares a
  method that shortens its descriptions with a control that lengthens them, which the recall
  shortcut above already discounts.
- **The other rows are confirmed.** The unmodified model, REVIS and the VTI control never loop,
  and two VTI control descriptions reach the limit in ordinary prose.

### Why REVIS behaves so on LLaVA-OneVision-1.5-8B

The port follows the official code at commit `75f4d29`, as on LLaVA-1.5-7B. Here the vectors are
computed by the recipe and not released, and the calibration selects block 21 with a threshold of
-0.073. The gate therefore opens wherever the cosine between the residual stream and the vector is
below 0.073. The readings below come from the output of block 21 before the vector is added, which
the steering cannot change. They were taken from a pass without steering, and for CHAIR from the
descriptions of each condition read again.

- **The gate sits at the edge of the states it reads.** At the last token of a POPE question the
  cosine is 0.066 with a standard deviation of 0.007, so the gate opens on 86% of the questions,
  83% of those whose answer is yes and 90% of those whose answer is no. MME questions end in
  "Please answer yes or no.", which raises the cosine to 0.092, and the gate opens on 2 of their
  2374. In the prompt it opens on 2% of the image tokens and 19% of the other tokens of POPE. In
  CHAIR it opens at the first token of every description and at about 30% of the later ones,
  under every condition.
- **The vector is orthogonal to the answer.** At block 21 its cosine with the difference between
  the mean states of the questions answered yes and answered no is 0.001, and its cosine with the
  difference of the output rows of yes and no is -0.011. At strength 2.5 it has a norm of 89
  against 117 for the stream at the last token, which is the shift of 0.64. Read directly by the
  final normalization and the output head, it would lower the log odds of yes by 2.4. After the
  remaining fourteen blocks the log odds move by 0.21 at the median on the first 50 images. On
  the whole of POPE adversarial, 27 of 3000 answers change, 16 from yes to no and 11 from no to
  yes, 15 to the right answer and 12 to the wrong one, so the yes ratio stays put.
- **A stronger vector breaks the answer before it moves it.** On the first 50 images, strength 10
  flips 64 of 300 first tokens and strength 25 flips 121, mostly to tokens that are neither yes
  nor no, such as "...", "-" and a blank line. Accuracy falls from 0.877 to 0.743 and then to
  0.560. The vector carries no direction that turns a yes into a no, at any strength.
- **REVIS shortens the descriptions it gates.** The descriptions of REVIS run to 123 tokens against
  176, with a gate open at their first token and at a third of the others, and name fewer
  objects. This is the recall shortcut above. Only 1.9 points of CHAIR_S are beyond it.
- **The sign of the vectors is consistent from block 15 upward.** Every computed visual vector has
  a cosine between 0.45 and 0.94 with the one of the index below from hidden state index 16 to 36.
  Below index 16 the sign alternates, which the calibration never reaches. Unlike the vectors
  released for LLaVA-1.5-7B, no flip lies near the selected block. Each vector is the first
  principal component of samples that PCA centers, so it follows the variance of what the image
  adds to a description and not its mean. At index 22 its cosine with that mean is -0.12.
- **The gate of the control is always open.** The cosine of a random vector with any state has a
  standard deviation of 1 over the square root of 4096, which is 0.016, so it is always below
  0.073. The control adds a vector of norm 89 at every position, image tokens included, on every
  task. It changes 77 POPE answers, 66 of them from yes to no, and 97 of the 2374 MME answers.
- **The long descriptions of the control drift and do not loop.** 186 of its 500 descriptions run
  to the token limit of 512, but none of them ends in a repeated word or phrase. They leave the
  image and continue as essays, for instance "the image serves as a reminder of the power of
  photography".

### Where a wrong yes comes from

Three readings of the unmodified LLaVA-1.5-7B on the 3000 questions of POPE adversarial, with
`swce-diagnose --prior --encoder --knockout`, on one A100. A wrong yes is a question whose object
is absent and whose first answer token favours yes. There are 507 of them among the 1500 absent
objects, and the margin of the first token gives a yes ratio of 0.617, as the generated answers
do.

**The language prior.** Every question was asked again with the image span removed.

| Reading | Value |
|---|---|
| Yes ratio with the image, and without it | 0.617, and 0.817 |
| AUROC of the answer without the image, for the expected answer | 0.42 |
| AUROC of the imageless margin for telling wrong yeses from correct noes | 0.71 |
| AUROC of the co-occurrence of the object with those of the image, for the same | 0.63 |
| Wrong yeses, and correct noes, whose imageless margin already favours yes | 96%, and 86% |
| Spearman correlation of the imageless margin and the co-occurrence | 0.01 |

- **The question alone pushes to yes.** Without the image the model says yes to 82% of the
  questions, and it is worse than chance at the answer, since the adversarial split asks about
  objects that fit the scene. The image is what brings the yes ratio down to 62%.
- **The prior picks out the wrong yeses.** Among absent objects, the margin without the image
  tells the wrong yeses from the correct noes with an AUROC of 0.71. The wrong yeses fall on
  the objects the question alone already favours.
- **The scene adds to it, apart.** How often the object co-occurs in COCO train2014 with the
  objects of the image predicts the wrong yeses too, at 0.63, and does not correlate with the
  prior. One depends on the object asked about, the other on what is in the image.

**The vision encoder.** A probe of every COCO category was fitted on the mean of the image
tokens after every block of CLIP and after the projector, on 5000 COCO train2014 images, and each
POPE question was scored by the probe of its object on its own image.

| Block of the encoder | 0 | 6 | 12 | 16 | 18 | 23 | Projector |
|---|---|---|---|---|---|---|---|
| AUROC over the POPE questions | 0.58 | 0.67 | 0.77 | 0.82 | 0.83 | 0.82 | 0.83 |

- **The encoder holds most of the answer, and the language model at least as much.** The probes
  reach 0.82 to 0.83 from block 16 of the encoder to the projector, where the probes of the residual
  stream reach 0.90 by block 10 and the model answers at 0.91. The language model does not lose
  what the encoder sees.
- **The comparison is a lower bound on the encoder.** The probes read the mean over all image
  tokens, which dilutes small objects, and were fitted on other images, while the probes of the
  stream read the position of the answer to the question itself. Pooling over the patches of the
  object, the second reading of issue #33, is needed before saying the encoder holds less.

**Attention knockout.** The attention from one set of positions to another was blocked in
windows of nine consecutive layers, on 200 wrong yeses and 200 correct answers. The change is in
the margin of yes over no, in nats, and the flipped share is that of answers that change side.

| Route blocked | Layers | Mean change, correct | Flipped, correct | Mean change, wrong yes | Flipped, wrong yes |
|---|---|---|---|---|---|
| Answer to the image | any window | -0.2 to +0.1 | 1% or less | -0.2 to +0.1 | 6% or less |
| Text to the image | 5-13 to 10-18 | -1.0 to -1.8 | 12% to 54% | -0.6 to -1.8 | 5% to 52% |
| Answer to the text | 8-16 to 12-20 | +1.8 to +2.1 | 41% | +1.8 to +2.0 | 0% |
| Answer to the text | 0-8 and 1-9 | -1.0 to -1.2 | 4% | -1.2 to -1.3 | 41% |

The text is the question and the template between the image and the answer.

- **The image reaches the answer through the question, not directly.** Blocking the answer from
  the image tokens changes almost nothing in any window. Blocking the question from the image
  between layers 5 and 18 moves the margin toward no by up to 1.8 nats and flips up to half of the
  answers. In the middle layers the image is written into the tokens of the question, and the
  answer reads it there. This is the route found for LLaVA in work on cross-modal information
  flow.
- **The no comes from the middle layers.** Blocking the answer from the text between layers 8 and
  20 moves every margin toward yes by about 2 nats and turns 41% of correct answers into yes. With
  that route closed the model says yes by default, which agrees with the prior above.
- **A wrong yes leans on an early reading of the question.** Blocking the answer from the text in
  the first nine layers flips 41% of the wrong yeses to no, and only 4% of the correct answers.
  The answer position reads the question early, before the image has reached it, and that early
  reading is a push to yes on which the wrong yeses depend far more than the correct answers.
- **The picture fits the probes.** The answer becomes readable in the stream between blocks 8 and
  11, the layers where the image reaches the answer through the question.

A knockout puts the stream where the model never goes on its own, so its changes say which routes
the answer depends on, not how large each contribution is in a normal pass. Patching (#35) tests
the same layers by copying states from a correct run instead.

### Where a wrong yes is decided, and what the image crop hides

Three more readings of the unmodified LLaVA-1.5-7B on POPE adversarial, with `swce-diagnose
--patching --attention` on one A100, and `swce-diagnose --items` on the answers of the main
experiment.

**Activation patching.** Each of 200 wrong yeses was paired with a correct no to the same question
on another image. The states of the correct run were copied into the wrong one at one block and
one group of positions at a time, and the table gives the median share of the gap between the
two margins that the copy closes, one meaning the wrong yes becomes the correct no.

| Block | 0 | 6 | 8 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 20 | 28 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Image tokens | 1.00 | 0.93 | 0.75 | 0.53 | 0.25 | 0.13 | 0.11 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| Tokens of the object word | 0.00 | 0.07 | 0.35 | 0.32 | 0.54 | 0.44 | 0.13 | 0.11 | 0.08 | 0.01 | 0.00 | 0.00 |
| Position of the answer | 0.00 | 0.00 | 0.00 | 0.02 | 0.04 | 0.11 | 0.35 | 0.41 | 0.58 | 0.90 | 0.95 | 1.00 |

- **What decides the answer moves from the image to the object word to the answer.** Up to
  block 10 the image tokens still carry it, and copying them is enough. Between blocks 8 and 12
  it sits in the tokens of the object named by the question, which alone restore up to 54% of
  the gap at block 11. From block 13 it is at the position of the answer, where copying one
  state restores 90% of the gap by block 16 and all of it by block 28.
- **A wrong yes is settled by block 16.** After that the answer position alone holds it, and no
  other position matters.
- **This agrees with the knockout and the probes.** The image reaches the question between
  layers 5 and 18 and the answer reads the question between layers 8 and 20, and the probes read
  the answer from block 8 to 11. Patching names the tokens that carry it at each step.
- Copying image tokens swaps the whole image and not only the object, so the early rows say the
  answer is still recomputed from the image there, not that the object alone decides it.

**Norm-weighted attention.** The attention of the answer position on every token, weighted by the
norm of what each head writes from it (Kobayashi et al., 2020) and summed over heads, was split
into the image, the region of the object asked about and the regions of the other objects, with
the COCO boxes mapped onto the grid of image tokens.

- **The crop of the processor hides objects.** The processor of LLaVA-1.5 on the Hugging Face hub
  resizes the shortest side to 336 pixels and crops the centre, where the original code pads the
  image to a square. Of the 1351 present objects with a COCO box, 88 lie wholly outside the crop
  and 131 keep less than half their box.
- **Most misses are objects the model cannot see or that have no box.** The model answers no to
  39% of the objects outside the crop and 10% of those under half visible, against 2% to 6% of
  the visible ones. Of its 156 misses, 34 are objects outside the crop, 13 under half visible, and
  64 are questions whose object has no COCO box of that category, whose label is doubtful. Only
  45 misses are about objects with a box and at least half visible.
- **A wrong yes looks at the image more, in the middle layers.** Between layers 14 and 21 the
  answer position puts 19% of its weighted attention on the image for a wrong yes against 15%
  for a correct no, which tells them apart with an AUROC of 0.89, and 11% against 8% on the
  regions of the other objects (AUROC 0.64). A wrong yes reads more of the image, and more of the
  objects that are there, which fits the role of co-occurrence above. The share is correlational.
- The first token takes 10% to 40% of the weighted attention at most layers, the sink of LLaVA,
  whatever the answer.

**On models whose images vary in size.** Qwen3-VL and LLaVA-OneVision-1.5 resize every image to
a whole number of patches, keeping its aspect, so images give different numbers of tokens.

- **Patching** pairs a wrong yes with the first correct no to the same question whose sequence
  has the same length, trying up to 20 partners, and records how many it tried. COCO images
  mostly share a few sizes, so most wrong yeses find one.
- **Attention** maps boxes through the `image_grid_thw` the processor returns, the size of the
  resized image in patches, each token of the language model covering a square of merged
  patches. Nothing is cropped, so every object is visible.
- **DeepStack.** Qwen3-VL adds features of its vision encoder to the image tokens again after
  each of its first three decoder blocks. A patch of the image tokens at those blocks is partly
  overwritten by the hallucinating image, so those rows are marked `reinjected`, and readings
  that hook the blocks miss that addition at the image positions there. The position of the
  answer and the question are never injected into.

The crop is a difference from the official LLaVA-1.5, which pads, and it applies to every run of
this model here. It lowers the recall of POPE by hiding a few percent of the present objects, and
it hides the same objects under every method, so it does not bias the comparison of methods.

**The answers item by item.** From the answers saved by the main experiment.

- **The objects most hallucinated are small things that fit many scenes.** Among absent objects
  asked about 20 times or more, the unmodified model says yes to 77% of handbags, 69% of traffic
  lights, 66% of bottles, 64% of backpacks, 62% of trucks and of tvs and 58% of cups, against 26%
  of dining tables. The rate of an object rises with its co-occurrence with the objects of the
  image, with a Spearman correlation of 0.30 across objects and 0.22 across questions.
- **VTI moves answers toward no, and its control does half of it.** At its POPE setting VTI turns
  120 wrong yeses into correct noes and breaks 45 correct noes, and turns 56 correct yeses into
  misses while fixing 19. Its random control fixes 68 and breaks 11 on absent objects and breaks
  28 present ones. REVIS at block 30 only moves the other way, breaking 158 correct noes and
  fixing 60 misses, and its control does the same at 98 and 43.
- **RePOPE removes 316 questions and relabels 172.** Of the 662 errors of the unmodified model,
  88 are on relabelled questions. Removing the ambiguous questions alone raises the F1 from 0.802
  to 0.811, and the new labels bring it to 0.805, so the corrected labels change the level more
  than the ranking of methods.
- **REVIS at block 31 never acts on the first answer token of POPE.** It changes the text of 81% of
  the answers and the probability of the first token of none, so no yes or no changes. On MME,
  whose questions end with "Please answer yes or no.", it moves the first token of 58% of the
  questions and changes 20% of the answers. VTI moves the first token of 91% of the POPE answers.
  A method that acts only after the first token cannot change the answer to a yes or no question.
- **The capability VTI costs at its POPE setting is mostly counting.** Of its loss of 216 MME
  points, 70 are on the count subtask, 23 on position and 10 on existence, while it gains 20 on
  color. Its random control loses 22 on position and 10 on count.

### Where a wrong yes comes from on Qwen3-VL-8B and LLaVA-OneVision-1.5-8B

The same readings of the unmodified `qwen3-vl-8b` and `llava-onevision-1.5-8b` on the 3000
questions of POPE adversarial, with `swce-diagnose --prior --knockout`, `--encoder`, `--patching`
and `--attention`, on one A100. Both language models have 36 layers. A wrong yes is counted as
for LLaVA-1.5-7B, from the margin of the first answer token.

**The language prior.**

| Reading | LLaVA-1.5-7B | Qwen3-VL-8B | OneVision-1.5-8B |
|---|---|---|---|
| Wrong yeses among the 1500 absent objects | 507 | 231 | 136 |
| Yes ratio with the image, and without it | 0.617, 0.817 | 0.521, 0.195 | 0.463, 0.925 |
| AUROC of the imageless margin for telling wrong yeses from correct noes | 0.71 | 0.66 | 0.45 |
| AUROC of the co-occurrence, for the same | 0.63 | 0.66 | 0.61 |
| Wrong yeses, and correct noes, whose imageless margin favours yes | 96%, 86% | 37%, 22% | 93%, 96% |
| Spearman correlation of the imageless margin and the co-occurrence | 0.01 | 0.24 | 0.00 |

- **The prior points the other way on Qwen3-VL.** Without the image it says yes to only 20% of
  the questions, so the question alone pushes it to no, and the image raises the yes ratio. Its
  wrong yeses still fall where the imageless margin is least negative, with an AUROC of 0.66.
- **OneVision says yes to almost everything without the image**, 92% of the questions, so its
  prior is the same for the wrong yeses and the correct noes and tells them apart no better than
  chance. Its wrong yeses come from what the image adds.
- **Co-occurrence predicts the wrong yeses on all three models**, at 0.61 to 0.66. It is the one
  reading of the prior that does not depend on the model, and the one that holds across them.

**The vision encoder.** The probes of LLaVA-1.5-7B, on the mean of the image tokens after every
block of each encoder and after its merger.

| Block of the encoder | 0 | 6 | 12 | 16 | 18 | 19 | 23 | 25 | Merger |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-8B, 27 blocks | 0.59 | 0.70 | 0.79 | 0.83 | 0.85 | 0.86 | 0.87 | 0.87 | 0.88 |
| OneVision-1.5-8B, 24 blocks | 0.60 | 0.70 | 0.80 | 0.86 | 0.87 | 0.88 | 0.87 | | 0.87 |

- **Both encoders hold the object about as well as CLIP, slightly better.** They reach 0.87 to
  0.88 at the merger, against 0.83 for LLaVA-1.5-7B, and OneVision answers at an AUROC of 0.94.
  As for LLaVA, the mean over all tokens is a lower bound on what the encoder holds.

**Attention knockout.** Windows of nine layers, on 200 wrong yeses and 200 correct answers, the
change being in the margin of yes over no, in nats.

| Model | Route blocked | Layers | Mean change, correct | Flipped, correct | Mean change, wrong yes | Flipped, wrong yes |
|---|---|---|---|---|---|---|
| Qwen3-VL | Answer to the image | any window | -1.0 to +0.1 | 0% | -0.3 to +0.1 | 5% or less |
| Qwen3-VL | Text to the image | 7-15 to 10-18 | -1.0 to -3.2 | 16% to 20% | -5.6 to -6.0 | 48% to 60% |
| Qwen3-VL | Answer to the text | 15-23 to 22-30 | +7.2 to +11.2 | 40% to 57% | +4.2 to +8.4 | 0% |
| Qwen3-VL | Answer to the text | 0-8 and 1-9 | +1.0 to +1.9 | 2% or less | +1.4 | 1% or less |
| OneVision | Answer to the image | any window | 0.0 to +0.1 | 2% or less | -0.1 to +0.1 | 3% or less |
| OneVision | Text to the image | 5-13 to 7-15 | -0.5 to -0.8 | 18% to 22% | -1.2 to -1.3 | 40% to 45% |
| OneVision | Text to the image | 16-24 to 18-26 | -0.2 to -0.4 | 5% to 8% | -1.0 to -1.2 | 37% to 48% |
| OneVision | Answer to the text | 15-23 to 19-27 | +1.0 to +2.0 | 24% to 48% | +0.6 to +1.3 | 0% |
| OneVision | Answer to the text | 24-32 to 27-35 | -0.4 to -0.6 | 2% | -0.4 to -0.6 | 22% to 25% |

- **On every model the image reaches the answer through the question.** Blocking the answer from
  the image tokens changes almost nothing, and blocking the question from the image flips about
  half of the wrong yeses: in layers 7 to 18 on Qwen3-VL, and in two bands, 5 to 15 and 16 to 26,
  on OneVision. The wrong yeses depend on that route far more than the correct answers do, which
  says the image is what writes the yes into the question on these models.
- **The no is read later, in layers 15 to 30.** Closing the answer from the text there moves every
  margin toward yes and turns up to 57% of the correct answers on Qwen3-VL and 48% on OneVision
  into yes, as layers 8 to 20 do on LLaVA-1.5-7B. The route is the same, deeper in a deeper model.
- **The early push to yes of LLaVA-1.5-7B is absent.** Blocking the answer from the text in the
  first nine layers flips 41% of the wrong yeses of LLaVA-1.5-7B and none on the other two, whose
  prior gives no such push. On OneVision the last layers play that part instead: blocking the
  answer from the text in layers 24 to 35 flips about a quarter of its wrong yeses and 2% of the
  correct answers.

**Activation patching.** Wrong yeses paired with a correct no to the same question whose sequence
has the same length, 194 pairs of 231 on Qwen3-VL and 108 of 136 on OneVision. The table gives
the median share of the gap closed.

| Block | 0 | 6 | 10 | 13 | 15 | 17 | 18 | 20 | 22 | 23 | 27 | 35 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL, image tokens | 0.92 | 0.99 | 0.94 | 0.83 | 0.61 | 0.48 | 0.23 | 0.07 | 0.00 | 0.00 | 0.00 | 0.00 |
| Qwen3-VL, object word | 0.00 | 0.01 | 0.11 | 0.36 | 0.70 | 0.64 | 0.70 | 0.34 | 0.11 | 0.00 | 0.00 | 0.00 |
| Qwen3-VL, answer | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 | 0.04 | 0.15 | 0.50 | 0.79 | 0.94 | 1.00 |
| OneVision, image tokens | 1.00 | 0.99 | 0.97 | 0.89 | 0.68 | 0.53 | 0.26 | 0.07 | 0.01 | 0.00 | 0.00 | 0.00 |
| OneVision, object word | 0.00 | 0.02 | 0.13 | 0.50 | 0.73 | 0.65 | 0.70 | 0.40 | 0.11 | 0.00 | 0.00 | 0.00 |
| OneVision, answer | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.04 | 0.23 | 0.58 | 0.78 | 0.88 | 1.00 |

- **The same three steps, at the same depth on both models.** The image tokens carry the answer
  up to block 16, the object word from block 13 to 21, with a peak of 0.70 to 0.73 at block 15
  where LLaVA-1.5-7B peaks at 0.54 at block 11, and the answer position from block 20, holding 78% to 79%
  of the gap by block 23 and all of it by the last block.
- **The two models, of different makers and with different encoders, agree to the block.** Both
  language models derive from Qwen3-8B, so the agreement says where that language model decides,
  whatever encoder feeds it.
- **A wrong yes is settled later than on LLaVA-1.5-7B.** The answer position holds 88% to 94% of
  the gap by block 27 of 36, where it holds 90% by block 16 of 32 on LLaVA-1.5-7B, and the object
  word holds more of it on its own.
- **DeepStack shows at block 0.** Copying the image tokens at block 0 closes 92% of the gap on
  Qwen3-VL and all of it on OneVision, which has no DeepStack. The reinjected features of the
  hallucinating image make up the rest.

**Norm-weighted attention.** No image is cropped, so all 1351 present objects with a COCO box
are wholly visible on both models.

- **On Qwen3-VL a wrong yes looks at the image more, later and less than on LLaVA-1.5-7B.**
  Between layers 21 and 28 the answer position puts 13.3% of its weighted attention on the image
  for a wrong yes against 11.2% for a correct no, an AUROC of 0.80, and 5.3% against 4.0% on the
  other objects (AUROC 0.62). Below layer 19 the shares of all answers agree to the third
  decimal.
- **On OneVision the difference is larger and runs to the output.** Between layers 21 and 28 a
  wrong yes puts 16.7% of its weighted attention on the image against 13.1% for a correct no, an
  AUROC of 0.86, and 9.3% against 6.7% on the other objects (AUROC 0.67). Up to layer 20 the
  shares agree, and the image takes up to 45% of the attention of a wrong yes at layer 33.
- **On all three models a wrong yes reads more of the image where the answer is decided**, in
  layers 14 to 21 on LLaVA-1.5-7B and 21 to 28 on the other two, the blocks where patching moves
  the answer to its own position. It also reads more of the objects that are there, which fits
  the role of co-occurrence. The shares are correlational.

All four readings name the same route on the three models: the image is written into the
question, the object word carries the answer for a few blocks, and the answer position reads it.
The language models of Qwen3-VL and OneVision decide about seven blocks later than LLaVA-1.5-7B,
deeper also in the share of their layers, and their wrong yeses do not lean on the early reading
of the question that those of LLaVA-1.5-7B lean on, so a layer chosen on LLaVA-1.5-7B does not
fall on the same step of these two.

### The answers of LLaVA-OneVision-1.5-8B item by item, and the yes or no instruction

The item analysis of LLaVA-1.5-7B, with `swce-diagnose --items`, on the answers of the main
experiment of `llava-onevision-1.5-8b`, and its POPE tasks asked again with "Please answer yes or
no." appended, `pope-adversarial-yn` and `repope-adversarial-yn`, with `swce-evaluate`. The
evaluation ran on one RTX 4090, the card of the main run, with Python 3.13.13, PyTorch
2.14.1+cu126, transformers 5.23.0, bfloat16 and seed zero. Its scores are in
`results/leaderboard.csv`.

**The answers item by item.**

- **The objects most hallucinated are again small things that fit many scenes, at lower rates.**
  Among absent objects asked about 20 times or more, the unmodified model says yes to 33% of
  trucks, 32% of handbags, 23% of backpacks, 17% of bottles and 13% of chairs, against 10% of
  dining tables and 3% of cars. The rate of an object rises with its co-occurrence with the
  objects of the image, with a Spearman correlation of 0.24 across objects and 0.12 across
  questions, against 0.30 and 0.22 on LLaVA-1.5-7B.
- **VTI moves answers toward yes, the other way from LLaVA-1.5-7B.** It turns 48 correct noes into
  wrong yeses and fixes none, and fixes 34 misses while breaking 5. Its random control does the
  same, less, breaking 19 correct noes and fixing 22 misses. On LLaVA-1.5-7B, VTI at its POPE
  setting moves answers toward no.
- **REVIS barely moves POPE, and its control moves it more.** REVIS fixes 16 answers and breaks 12.
  Its control fixes 33 wrong yeses and breaks 39 correct yeses.
- **RePOPE helps OneVision more than LLaVA-1.5-7B.** Of the 388 errors of the unmodified model, 122
  are on relabelled questions. Removing the ambiguous questions raises the F1 from 0.866 to 0.889,
  and the new labels bring it to 0.914. On LLaVA-1.5-7B the new labels lower it, from 0.811 to
  0.805, so OneVision agrees with the corrected labels where LLaVA-1.5-7B does not.
- **Every method moves the first answer token and almost no answer.** VTI moves the probability
  of the first token of 94% of the POPE answers, REVIS of 93% and their controls of 91% and 96%,
  and each changes at most 3% of the answers. REVIS acts at block 21 here, where its gate is open
  at the last prompt token, unlike REVIS at block 31 on LLaVA-1.5-7B.
- **No method costs a subtask of MME about objects.** On existence, count, position and color,
  every method is within 5 points of the unmodified model, where VTI at its POPE setting loses 70
  points on count on LLaVA-1.5-7B.

**The yes or no instruction.**

| F1 | POPE as written | POPE, yes or no | RePOPE as written | RePOPE, yes or no |
|---|---|---|---|---|
| Unmodified | 0.866 | 0.866 | 0.914 | 0.917 |
| VTI | 0.863 | 0.866 | 0.908 | 0.915 |
| VTI control | 0.869 | 0.867 | 0.916 | 0.920 |
| REVIS | 0.867 | 0.865 | 0.916 | 0.916 |
| REVIS control | 0.862 | 0.868 | 0.915 | 0.921 |

- **The instruction leaves the unmodified model where it was.** Its F1 stays at 0.866, its AUROC
  at 0.942, its yes ratio goes from 0.462 to 0.450 and d' from 2.29 to 2.35. It still answers
  with a sentence that begins with yes or no.
- **With the instruction, no method changes POPE.** No F1 moves by more than 0.003 on either label
  set, every McNemar p is 0.26 or more, and no yes ratio moves by more than 0.005.
- **The instruction removes the push of VTI toward yes.** Asked as written, VTI raises the yes
  ratio by 0.026 and lowers d' by 0.09. With the instruction it changes the yes ratio by 0.001
  and d' by -0.01 [-0.05, +0.04], and it fixes 21 answers and breaks 22, with no direction.
- LLaVA-1.5-7B has not yet been asked the variant, which is issue #37, so this part has no
  reading on that model to set against.

**What holds on both models.** Co-occurrence predicts which absent objects are hallucinated, and
the same kinds of object are hallucinated most. No method improves POPE beyond its random
control. VTI moves the first answer token of most questions of POPE while changing few answers.
What differs is the direction VTI pushes the answers, toward no on LLaVA-1.5-7B and toward yes on
OneVision, and that the corrected labels of RePOPE help OneVision and not LLaVA-1.5-7B.

### Sparse autoencoders on four blocks

`swce-sae` trained TopK autoencoders of 32768 latents, with 64 kept per state, on blocks 9, 13,
24 and 30, two seeds per block. Blocks 9 and 13 are where the probes and the lens first read the
answer, and 24 and 30 are a late block and the block of REVIS `table-11`. They trained on 6.1
million states from COCO train2014 only: the 5000 Nullu descriptions, the 5952 REVIS
calibration questions and 256 descriptions the model wrote itself, in three passes, on one
NVIDIA A100-SXM4-80GB with the locked environment. The checks below were fixed before training.
The two readings ran on the 3000 questions of POPE adversarial and the checks on 1000 held-out
sequences.

| Block | 9 | 13 | 24 | 30 |
|---|---|---|---|---|
| Unexplained variance at the end of training | 6.1% | 9.3% | 13.2% | 15.0% |
| Held out, image tokens of the descriptions | 6.4% | 9.6% | 17.4% | 21.9% |
| Held out, description text | 18.5% | 18.0% | 14.2% | 15.1% |
| Latents that never fire on the held-out states | 3.5% | 0.5% | 0% | 0% |
| Next token loss on the descriptions, from 1.220 nats | +0.059 | +0.069 | +0.094 | +0.158 |
| Latents of one seed matched by the other above a cosine of 0.7 | 2.1% | 1.9% | 3.1% | 2.7% |
| Best single latent for a hallucinated yes, AUROC on the held-out half | 0.79 | 0.87 | 0.87 | 0.87 |

Both seeds give the same figures to within a few tenths of a point.

- **Reconstruction passes.** The autoencoders keep most of the variance of every kind of token
  and cost under 0.16 nats of next-token loss when they replace a block. Description text is the
  hardest at the early blocks and image tokens at the late ones.
- **Replication across seeds fails.** Only 2 to 3% of the latents of one seed have a match
  above a cosine of 0.7 in the other, and the median best match is 0.21 to 0.28. Autoencoders of
  this size trained on hundreds of millions of tokens usually share most of their latents. Six
  million are too few for the dictionary to settle. Each latent is one solution among many, and
  none can be named as a feature of the model. The latents the readings below name replicate no
  better. The readings are therefore stated only where both seeds agree, as patterns, and no
  latent is interpreted.
- **The position of the answer is poorly described.** The checks measure the unexplained
  variance across the 3000 questions at that one position, where the states differ little. It is
  57% at block 9 and 21 to 29% after it, so what separates one question from another there is
  only partly in the latents.
- **The steered states fall outside the dictionary.** At the position of the answer, the
  unexplained variance under VTI is 10 times that of the unmodified model at block 9 and 2.8 to 4.4
  times at the later blocks. Under REVIS at block 30 it is 34 to 37 times. Both methods put the stream
  where the model itself never goes. This agrees with the probes, which found the unmodified
  probe failing under both, and limits what any reading through these autoencoders can say about
  them. REVIS at block 31 leaves every state of these blocks as it was, as its gate stays shut at
  that position.
- **VTI's direction is not sparse in the dictionary.** At every block the ten latents closest to
  its direction explain 40 to 47% of it and fifty explain 57 to 72%, and no latent has a cosine
  above 0.50 with it. The ten latents whose activation it changes most carry 19 to 26% of its
  total change. At blocks 9 and 13 they all fall, with decoder rows nearly orthogonal to its
  direction, which is what renormalizing the stream after a shift does to everything else in it.
  At blocks 24 and 30 the latents that rise are the ones aligned with its direction. The
  direction of the block itself explains little of what VTI does to it there, with a cosine of
  0.17 at block 9 against 0.57 to 0.64 at blocks 24 and 30, since earlier blocks are shifted too.
- **The stream separates a hallucinated yes from a correct one.** Among the 1851 questions the
  unmodified model answers yes, 507 hallucinate. Single latents chosen on half the images tell
  them apart with an AUROC of 0.87 on the other half from block 13 on, and 0.79 at block 9, in
  both seeds. Their decoder rows have cosines of 0.3 to 0.6 with the difference of the mean states
  of the two groups, which uses no autoencoder. So at the moment it says yes the model often
  holds the evidence that the object is absent, as the probes suggested.
- **VTI moves the same latents, mostly away from correct answers.** Many of the latents VTI
  changes most are also among those that best separate the two kinds of yes, at every block and
  in both seeds. It mostly lowers the ones that are higher when the yes is correct, which is
  consistent with VTI damping the evidence for yes rather than sharpening it, and with the d'
  it lowers. Most of these latents fire on every question, so they are the directions of largest
  variance at that position, and the overlap may come from that alone.

These readings are correlational. Removing a latent and watching the answers would test whether
it matters, and needs autoencoders that replicate first, so a longer training. The trained
autoencoders, their curves and the tables of these checks are kept with the data and are not
committed.

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
