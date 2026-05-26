# A/B Comparison: baseline, cache_exact_only, cache_warm_start over N=3 trials each

**MIDI:** /private/tmp/dense_test.mid (16.8s duration, 132 notes, dense bimanual)
**Active window:** 15.0s

| Variant | Wall time (median +/- IQR) | IK success frac | Residual mean | Residual p95 | Static F1 |
|---|---|---|---|---|---|
| baseline | 118.57s +/- 0.38s | 0.016 | 0.1172 | 0.1964 | NaN |
| cache_exact_only | 77.18s +/- 0.08s | 0.020 | 0.1388 | 0.2206 | NaN |
| cache_warm_start | 71.99s +/- 0.42s | 0.016 | 0.1398 | 0.2220 | NaN |

**Speedups vs baseline (median wall):**
- cache_exact_only: 1.54x faster (-34.9%)
- cache_warm_start: 1.65x faster (-39.3%)

**Quality regressions:**
- cache_exact_only: F1 unavailable
- cache_warm_start: F1 unavailable

**Verdict:**
Faster than baseline: cache_exact_only, cache_warm_start.
