# Leaderboard

Generated from `leaderboard.csv`. Do not edit by hand. Cells hold the primary metric
of the task with its 95% bootstrap interval over images.

## llava-onevision-1.5-8b

| Method | Setting | Fitted on | pope-adversarial | repope-adversarial | chair | mme |
|---|---|---|---|---|---|---|
| revis | code-default | lure-pairs, coco-train2014 | 0.8668 [0.854, 0.8795] | 0.9157 [0.9036, 0.9255] | 0.256 [0.22, 0.296] | 2266 [2188, 2343] |
| revis-random | code-default | nothing | 0.8623 [0.8497, 0.8743] | 0.9155 [0.9038, 0.9258] | 0.384 [0.344, 0.43] | 2285 [2206, 2368] |
| unmodified | default | nothing | 0.8656 [0.8526, 0.878] | 0.9137 [0.9023, 0.9242] | 0.334 [0.292, 0.374] | 2280 [2202, 2356] |
| vti | code-default | coco-train2014 | 0.8626 [0.8504, 0.8748] | 0.908 [0.8954, 0.9193] | 0.344 [0.302, 0.388] | 2229 [2151, 2308] |
| vti-random | code-default | nothing | 0.8686 [0.8559, 0.8814] | 0.9158 [0.9044, 0.926] | 0.374 [0.328, 0.416] | 2262 [2183, 2334] |

Metrics: `pope-adversarial` f1, `repope-adversarial` f1, `chair` chair_s, `mme` total
