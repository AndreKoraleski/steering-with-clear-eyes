# Leaderboard

Generated from `leaderboard.csv`. Do not edit by hand. Cells hold the primary metric
of the task with its 95% bootstrap interval over images.

## llava-1.5-7b

| Method | Setting | Fitted on | pope-adversarial | repope-adversarial | chair | mme |
|---|---|---|---|---|---|---|
| revis | default | lure-pairs, coco-train2014 | 0.8024 [0.7911, 0.8133] | 0.8053 [0.7913, 0.8183] | 0.602 [0.56, 0.648] | 1466 [1402, 1531] |
| revis-random | default | nothing | 0.8024 [0.7911, 0.8133] | 0.8053 [0.7913, 0.8183] | 0.502 [0.46, 0.548] | 1769 [1693, 1844] |
| unmodified | default | nothing | 0.8024 [0.7911, 0.8133] | 0.8053 [0.7913, 0.8183] | 0.502 [0.46, 0.548] | 1769 [1693, 1844] |
| vti | chair | coco-train2014 | n/a | n/a | 0.366 [0.324, 0.41] | 1757 [1683, 1832] |
| vti | pope | coco-train2014 | 0.8073 [0.7967, 0.8182] | 0.8167 [0.8028, 0.8299] | n/a | 1553 [1483, 1619] |
| vti-random | chair | nothing | n/a | n/a | 0.436 [0.394, 0.484] | 1758 [1686, 1827] |
| vti-random | pope | nothing | 0.808 [0.7963, 0.8191] | 0.8137 [0.7992, 0.8266] | n/a | 1684 [1610, 1753] |

Metrics: `pope-adversarial` f1, `repope-adversarial` f1, `chair` chair_s, `mme` total
