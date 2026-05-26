# A/B Comparison: baseline, cache_exact_only over N=3 trials each

**MIDI:** /private/tmp/dense_test.mid (16.8s duration, 132 notes, dense bimanual)
**Active window:** 15.0s

| Variant | Wall time (median +/- IQR) | IK success frac | Residual mean | Residual p95 | Static F1 |
|---|---|---|---|---|---|
| baseline | 118.66s +/- 0.66s | 0.016 | 0.1172 | 0.1964 | NaN |
| cache_exact_only | 76.92s +/- 0.27s | 0.020 | 0.1388 | 0.2206 | NaN |

**Speedups vs baseline (median wall):**
- cache_exact_only: 1.54x faster (-35.2%)

**Quality regressions:**
- cache_exact_only: F1 unavailable

**Verdict:**
Faster than baseline: cache_exact_only.
