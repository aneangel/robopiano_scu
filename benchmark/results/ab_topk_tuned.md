# A/B Comparison: tuned_quality, topk2_tuned over N=3 trials each

**MIDI:** /private/tmp/dense_test.mid (16.8s duration, 132 notes, dense bimanual)
**Active window:** 15.0s

| Variant | Wall time (median +/- IQR) | IK success frac | Residual mean | Residual p95 | Static F1 |
|---|---|---|---|---|---|
| tuned_quality | 29.14s +/- 0.41s | 0.023 | 0.1315 | 0.1918 | 0.449 |
| topk2_tuned | 27.59s +/- 0.14s | 0.026 | 0.1156 | 0.1504 | 0.584 |

**Speedups vs baseline (median wall):**
- topk2_tuned: 1.06x faster (-5.3%)

**Quality regressions:**
- topk2_tuned: +0.135 F1

**Verdict:**
Faster than baseline: topk2_tuned. Quality preserved: topk2_tuned.
