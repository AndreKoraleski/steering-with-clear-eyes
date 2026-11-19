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
  unmodified length, which issue #20 checks for loops. The difference between REVIS and its
  control rests on that control and is not reported as a gain until then.
- **VTI does not help.** Its CHAIR_S is 34.4, within the interval of the unmodified model, with
  descriptions of 304 words, and its MME falls by 51 [2, 100]. Its control changes CHAIR_S by +4.0
  points, not significantly, at 195 words.
- No method is non-inferior to the unmodified model on MME.
- **The logit lens separates the questions whose answer is yes from those whose answer is no.**
  On POPE adversarial its AUROC stays near 0.5 up to block 17, rises to 0.87 at block 18 and 0.93
  at block 19, and stays near 0.94 to the output, under every method. It reads the expected answer
  and not whether the answer of the model is right.
- **REVIS changes the stream from block 21 on**, the block its calibration selected, and leaves
  the blocks below it as they were. Why its gate opens there without moving any answer of POPE is
  issue #19.

Lengths are in words, as `results/leaderboard.csv` reports them.

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
  reach 0.83 from block 16 of the encoder to the projector, where the probes of the residual
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
