# A/B Comparison: production_all, tuned_quality over N=3 trials each

**MIDI:** /private/tmp/dense_test.mid (16.8s duration, 132 notes, dense bimanual)
**Active window:** 15.0s

| Variant | Wall time (median +/- IQR) | IK success frac | Residual mean | Residual p95 | Static F1 |
|---|---|---|---|---|---|
| production_all | 13.52s +/- 0.43s | 0.026 | 0.1205 | 0.1846 | 0.412 |
| tuned_quality | 28.30s +/- 0.19s | 0.023 | 0.1315 | 0.1918 | 0.449 |

**Speedups vs baseline (median wall):**
- tuned_quality: 0.48x slower (+109.3%)

**Quality regressions:**
- tuned_quality: +0.037 F1

**Verdict:**
No variant beat baseline on median wall time. Quality preserved: tuned_quality.
