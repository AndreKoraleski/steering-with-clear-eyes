# Benchmarks

Generated from the registry by `swce-registry`. Do not edit by hand.

Registry version 1. Every file is pinned by its size and digest, and
an evaluation fails when a host serves other bytes.

## COCO 2014

Key `coco-2014`. Lin et al., Microsoft COCO: Common Objects in Context, ECCV 2014.

- **Source:** <https://cocodataset.org>
- **Licence:** Annotations CC BY 4.0. Images under the Flickr terms of use, with the licence of each image listed in the annotations.
- **Terms:** Images are fetched one by one from the official host and never redistributed. The host publishes no checksum of single images, so they are not pinned.
- **Images:** fetched one by one from `http://images.cocodataset.org/val2014/{name}`

| File | Bytes | Digest |
|---|---|---|
| `coco/annotations_trainval2014.zip` | 252872794 | md5 `0a379cfc70b0e71301e0f377548639bd` |

## COCO 2014 training images

Key `coco-train2014`. Lin et al., Microsoft COCO: Common Objects in Context, ECCV 2014.

- **Source:** <https://cocodataset.org>
- **Licence:** Images under the Flickr terms of use.
- **Terms:** No task reads these images. Steering methods fit their directions and calibrate their thresholds on them, and each image is fetched from the official host when a method names it.
- **Images:** fetched one by one from `http://images.cocodataset.org/train2014/{name}`

## POPE

Key `pope`. Li et al., Evaluating Object Hallucination in Large Vision-Language Models, EMNLP 2023.

- **Source:** <https://github.com/RUCAIBox/POPE>
- **Revision:** `08d957b917e5a378a2f99d35b6293c536a66298b`
- **Licence:** MIT
- **Terms:** Question files are downloaded from the official repository. Copies in other repositories differ and are not used.

| File | Bytes | Digest |
|---|---|---|
| `pope/coco_pope_random.json` | 368166 | sha256 `ac25245170b975a5bdf9080b23fd431dfe6be458bc038259c1f4f09a6bef7994` |
| `pope/coco_pope_popular.json` | 370190 | sha256 `72c1a8ad45d0c13514f5f22598261df41d3b533854d29682e924db50ed8aa753` |
| `pope/coco_pope_adversarial.json` | 370401 | sha256 `420b3407db1fa9f1187a805dca41cb7b97fd91504e6c2179706188c107fb8ef8` |

| Task | Kind | Primary metric | Measures |
|---|---|---|---|
| `pope-random` | yes_no | `f1` | Object presence questions on 500 COCO val2014 images, with absent objects drawn at random. |
| `pope-popular` | yes_no | `f1` | Object presence questions on 500 COCO val2014 images, with absent objects drawn among the most frequent ones. |
| `pope-adversarial` | yes_no | `f1` | Object presence questions on 500 COCO val2014 images, with absent objects that co-occur most with the present ones. |
| `pope-adversarial-yn` | yes_no | `f1` | Object presence questions on 500 COCO val2014 images, with absent objects that co-occur most with the present ones. Every question ends with "Please answer yes or no.", as the REVIS and V-ITI scripts ask it. |

## RePOPE

Key `repope`. Neuhaus and Hein, RePOPE: Impact of Annotation Errors on the POPE Benchmark, arXiv 2504.15707.

- **Source:** <https://github.com/YanNeu/RePOPE>
- **Revision:** `bcf22e5690d587cce34e1728af3efdcfaf747d29`
- **Licence:** None declared
- **Terms:** Corrected labels of the POPE questions, with ambiguous questions removed. Downloaded from the official repository and never redistributed.

| File | Bytes | Digest |
|---|---|---|
| `repope/coco_repope_random.json` | 340368 | sha256 `f464fcaf60b24ce2f280cf33f122e000dbdd35c8e88f77c4141c9c2cba864611` |
| `repope/coco_repope_popular.json` | 333773 | sha256 `57832b055bae5fd6176278a4f104c98ddd84811f7d26dfcb93d481f6c9da09a5` |
| `repope/coco_repope_adversarial.json` | 328764 | sha256 `8c346dbb060bf422ff92e769fa68e5ab3e8a777cb28a26e973387441e05c9ccf` |

| Task | Kind | Primary metric | Measures |
|---|---|---|---|
| `repope-random` | yes_no | `f1` | The POPE random questions with corrected labels, which reuse the answers given to POPE. |
| `repope-popular` | yes_no | `f1` | The POPE popular questions with corrected labels, which reuse the answers given to POPE. |
| `repope-adversarial` | yes_no | `f1` | The POPE adversarial questions with corrected labels, which reuse the answers given to POPE. |
| `repope-adversarial-yn` | yes_no | `f1` | The POPE adversarial questions with corrected labels, which reuse the answers given to POPE. Every question ends with "Please answer yes or no.", as the REVIS and V-ITI scripts ask it. |

## CHAIR

Key `chair`. Rohrbach et al., Object Hallucination in Image Captioning, EMNLP 2018.

- **Source:** <https://github.com/shikiw/OPERA>
- **Revision:** `e1bb7632508bdfaa5cdaf5b69b64419688107da8`
- **Licence:** MIT (OPERA repository, whose script and image list are used)
- **Terms:** The metric predates vision language models and fixes no images, prompt or length. The 500 val2014 images are those of the greedy LLaVA-1.5 run published with OPERA, read from its log for their identifiers only.

| File | Bytes | Digest |
|---|---|---|
| `chair/opera_llava15_greedy.jsonl` | 259599 | sha256 `3d70bbefa95f1894a09e97e83f2a40531be39b12db50f4e2386c5bfc99842fdd` |

| Task | Kind | Primary metric | Measures |
|---|---|---|---|
| `chair` | caption | `chair_s` | Detailed descriptions of 500 COCO val2014 images, scored for objects that are mentioned but absent. |

## MME

Key `mme`. Fu et al., MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models, arXiv 2306.13394.

- **Source:** <https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation>
- **Revision:** `7056f44dac19de35e510b62d734bb2f7a6f64739`
- **Licence:** Apache 2.0 tag on the hosting repository. The images come from third parties.
- **Terms:** The release archive is downloaded from the address the authors link and never redistributed.

| File | Bytes | Digest |
|---|---|---|
| `mme/MME_Benchmark_release_version.zip` | 199848448 | sha256 `1c4b79b494eae354df8fc0728c64fe4b89e46b4c609d36003be15dace6d8e3d7` |

| Task | Kind | Primary metric | Measures |
|---|---|---|---|
| `mme` | yes_no | `total` | Two yes or no questions on each of 1187 images in 14 perception and cognition subtasks, used to measure general capability. |

## VTI demonstrations

Key `vti-data`. Liu et al., Reducing Hallucinations in Large Vision-Language Models via Latent Space Steering, ICLR 2025.

- **Source:** <https://github.com/shengliu66/VTI>
- **Revision:** `bb5dfee33a9e555f22af9faa7b3495c0ac3df69e`
- **Licence:** None declared
- **Terms:** Faithful and hallucinated descriptions of COCO train2014 images, on which VTI fits its directions. Downloaded from the official repository and never redistributed. No task reads them.

| File | Bytes | Digest |
|---|---|---|
| `methods/vti/hallucination_vti_demos.jsonl` | 145506 | sha256 `5ce467817fc2a29735715fd6793b4927268cfff030e798f9f3a60a1fcb575d0d` |

## REVIS pairs, calibration questions and vector

Key `revis-data`. Wu et al., REVIS: Sparse Latent Steering to Mitigate Object Hallucination in Large Vision-Language Models, ICML 2026.

- **Source:** <https://github.com/antgroup/Revis>
- **Revision:** `75f4d29d402d4c1c4dbb2c69e88e3aa50cd0c9cc`
- **Licence:** Apache 2.0
- **Terms:** Pairs of descriptions and object presence questions on COCO train2014 images, on which REVIS computes its vectors and calibrates its gate, and the vectors its authors release for LLaVA-1.5. Downloaded from the official repository. No task reads them.

| File | Bytes | Digest |
|---|---|---|
| `methods/revis/OH_sampled_data_100.json` | 131358 | sha256 `b134685685a6ad29bfbf4769cac5e24337ef16126b22fb9ac34061be37b5762c` |
| `methods/revis/coco_pope_calibration_merged.jsonl` | 737993 | sha256 `2b3e8c4ffd554e3b5afd199568b0248dc51ffbb411873c8215b6f950f477fdfe` |
| `methods/revis/llava15_none_image.pt` | 1650809 | sha256 `c130a713856860fdb1f8f1819dff253f84445c4629172502670bd3395905fd4b` |

## Nullu pairs

Key `nullu-data`. Yang et al., Nullu: Mitigating Object Hallucinations in Large Vision-Language Models via HalluSpace Projection, CVPR 2025.

- **Source:** <https://github.com/Ziwei-Zheng/Nullu>
- **Revision:** `dcb1ef4ea44e659a769d72caa499ddde23eab877`
- **Licence:** MIT
- **Terms:** Truthful and hallucinated descriptions of 5000 COCO train2014 images, from LURE, on which Nullu builds its subspaces. Downloaded from the official repository and never redistributed. No task reads them.

| File | Bytes | Digest |
|---|---|---|
| `methods/nullu/hallucination5k_train.jsonl` | 7147910 | sha256 `46fd3acf8c3f07bd84d1092e8cee1faf432888c510bd0b3f6209db206feb4df5` |
