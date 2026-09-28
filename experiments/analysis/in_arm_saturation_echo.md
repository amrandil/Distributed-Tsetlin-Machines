There's a mix-up in the numbers: the 10% isn't a share of runs, it's part of the test applied to each run. Here's what the 883 means.

For every run I looked at the last 100 generations and measured two things:
- **Arm changes:** the share of cell-steps where the automaton switched arm.
- **Depth changes:** the share of cell-steps where the automaton stayed on its arm but moved to a different depth, which is what shows up as a change in saturation.

A run counts as showing the saturation effect when arm changes are **below 1%** (the colours are frozen) and depth changes are **at least 10%** (the saturation keeps moving). **883 of the 1,659 majority runs** pass this test, and **12 of the 1,659 minority runs** do. So the 883 are the runs themselves, and the 10% is the threshold each of them passes. 37 vs 45 is one of the 883, with 27% of its cell-steps changing depth.

The full list is in the attached CSV: 895 rows, one per run, with feedback, rules, class pair, Hamming label, settle, domination, and both shares.

**The 883 majority runs:**
- By label: 333 aperiodic, 315 cycle, 229 near_cycle, 6 drift.
- By domination: 859 coexist, 10 complete domination, 7 dominate, 7 neutral.
- By class pair: 469 II×III, 177 II×IV, 141 I×II, 43 I×III, 39 III×IV, 14 I×IV.
- 483 of them are in the selected subset from Section 5.
- Median depth-change share: 0.34, with the 10th–90th percentile at 0.15–0.51.

One caveat: the test detects the mechanism, not how striking it looks. In a `cycle` run the saturation texture is just as periodic as the B&W plot, so it matches but isn't eye-catching. The clearest cases are aperiodic runs like 37 vs 45, where the B&W plot has recognisable shapes to echo.

**The 12 minority runs:**

| Rules | Classes | Label | Settle | Domination | Depth changes |
|---|---|---|---|---|---:|
| 40 vs 57 | I×II | drift | early | neutral | 18% |
| 56 vs 22 | II×III | near_cycle | late | complete | 60% |
| 56 vs 60 | II×III | near_cycle | late | complete | 29% |
| 56 vs 122 | II×III | near_cycle | never | coexist | 25% |
| 57 vs 122 | II×III | near_cycle | late | dominate | 34% |
| 184 vs 22 | II×III | near_cycle | late | complete | 56% |
| 35 vs 106 | II×IV | drift | early | neutral | 70% |
| 57 vs 41 | II×IV | drift | early | neutral | 58% |
| 57 vs 54 | II×IV | near_cycle | tail_only | complete | 58% |
| 57 vs 106 | II×IV | near_cycle | tail_only | dominate | 43% |
| 58 vs 41 | II×IV | drift | early | neutral | 70% |
| 184 vs 41 | II×IV | drift | early | neutral | 30% |

These are the exceptions to minority's usual arm-switching. Under minority feedback an automaton only stays committed if its cell is mostly *not* in the local majority. That happens in striped or travelling patterns (all five drift runs) or when one rule has taken over and produced such a pattern (the complete and dominate runs). Only one of the 12, 56 vs 122, is a genuinely coexisting run.The CSV with all 895 runs is attached.