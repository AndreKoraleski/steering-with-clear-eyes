# Leaderboard

Generated from `leaderboard.csv`. Do not edit by hand. Cells hold the primary metric
of the task with its 95% bootstrap interval over images.

## llava-1.5-7b

| Method | Setting | Fitted on | pope-adversarial | repope-adversarial | chair | mme |
|---|---|---|---|---|---|---|
| revis | default | lure-pairs, coco-train2014 | 0.8024 [0.7911, 0.8133] | 0.8053 [0.7913, 0.8183] | 0.602 [0.56, 0.648] | 1466 [1402, 1531] |
| revis | table-11 | lure-pairs, coco-train2014 | 0.787 [0.7776, 0.796] | 0.7702 [0.7572, 0.7832] | 0.32 [0.28, 0.36] | 1812 [1734, 1889] |
| revis-random | default | nothing | 0.8024 [0.7911, 0.8133] | 0.8053 [0.7913, 0.8183] | 0.502 [0.46, 0.548] | 1769 [1693, 1844] |
| revis-random | table-11 | nothing | 0.7946 [0.7848, 0.8048] | 0.7835 [0.7704, 0.7963] | 0.254 [0.22, 0.292] | 1768 [1694, 1844] |
| unmodified | default | nothing | 0.8024 [0.7911, 0.8133] | 0.8053 [0.7913, 0.8183] | 0.502 [0.46, 0.548] | 1769 [1693, 1844] |
| vti | chair | coco-train2014 | n/a | n/a | 0.366 [0.324, 0.41] | 1757 [1683, 1832] |
| vti | pope | coco-train2014 | 0.8073 [0.7967, 0.8182] | 0.8167 [0.8028, 0.8299] | n/a | 1553 [1483, 1619] |
| vti-random | chair | nothing | n/a | n/a | 0.436 [0.394, 0.484] | 1758 [1686, 1827] |
| vti-random | pope | nothing | 0.808 [0.7963, 0.8191] | 0.8137 [0.7992, 0.8266] | n/a | 1684 [1610, 1753] |

Metrics: `pope-adversarial` f1, `repope-adversarial` f1, `chair` chair_s, `mme` total

## llava-onevision-1.5-8b

| Method | Setting | Fitted on | pope-adversarial | pope-adversarial-yn | repope-adversarial | repope-adversarial-yn | chair | mme |
|---|---|---|---|---|---|---|---|---|
| revis | code-default | lure-pairs, coco-train2014 | 0.8668 [0.854, 0.8795] | 0.8652 [0.8522, 0.8777] | 0.9157 [0.9036, 0.9255] | 0.9162 [0.9039, 0.9267] | 0.256 [0.22, 0.296] | 2266 [2188, 2343] |
| revis-random | code-default | nothing | 0.8623 [0.8497, 0.8743] | 0.8679 [0.855, 0.8807] | 0.9155 [0.9038, 0.9258] | 0.9205 [0.9088, 0.9305] | 0.384 [0.344, 0.43] | 2285 [2206, 2368] |
| unmodified | default | nothing | 0.8656 [0.8526, 0.878] | 0.866 [0.8531, 0.8782] | 0.9137 [0.9023, 0.9242] | 0.9175 [0.9052, 0.9282] | 0.334 [0.292, 0.374] | 2280 [2202, 2356] |
| vti | code-default | coco-train2014 | 0.8626 [0.8504, 0.8748] | 0.8658 [0.8531, 0.879] | 0.908 [0.8954, 0.9193] | 0.9146 [0.9022, 0.9253] | 0.344 [0.302, 0.388] | 2229 [2151, 2308] |
| vti-random | code-default | nothing | 0.8686 [0.8559, 0.8814] | 0.8674 [0.8545, 0.8806] | 0.9158 [0.9044, 0.926] | 0.92 [0.9083, 0.9303] | 0.374 [0.328, 0.416] | 2262 [2183, 2334] |

Metrics: `pope-adversarial` f1, `pope-adversarial-yn` f1, `repope-adversarial` f1, `repope-adversarial-yn` f1, `chair` chair_s, `mme` total

## qwen3-vl-8b

| Method | Setting | Fitted on | pope-adversarial | repope-adversarial | chair | mme |
|---|---|---|---|---|---|---|
| revis | code-default | lure-pairs, coco-train2014 | 0.8694 [0.8558, 0.8813] | 0.9103 [0.8982, 0.9207] | 0.03 [0.016, 0.046] | 2365 [2294, 2438] |
| revis-random | code-default | nothing | 0.7059 [0.6976, 0.7139] | 0.682 [0.6694, 0.6945] | 0.56 [0.516, 0.604] | 2396 [2324, 2470] |
| unmodified | default | nothing | 0.8712 [0.8586, 0.8835] | 0.9116 [0.8992, 0.9224] | 0.532 [0.486, 0.576] | 2403 [2336, 2474] |
| vti | code-default | coco-train2014 | 0.8479 [0.8355, 0.8592] | 0.8631 [0.8487, 0.876] | 0.094 [0.068, 0.118] | 2382 [2310, 2454] |
| vti-random | code-default | nothing | 0.8632 [0.8501, 0.875] | 0.9006 [0.8872, 0.9122] | 0.496 [0.452, 0.54] | 2390 [2315, 2460] |

Metrics: `pope-adversarial` f1, `repope-adversarial` f1, `chair` chair_s, `mme` total
