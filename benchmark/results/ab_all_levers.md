# A/B Comparison: baseline, cache_warm_only, l3_only, l1l3, l1l2l3, production_legacy, production_all over N=3 trials each

**MIDI:** /private/tmp/dense_test.mid (16.8s duration, 132 notes, dense bimanual)
**Active window:** 15.0s

| Variant | Wall time (median +/- IQR) | IK success frac | Residual mean | Residual p95 | Static F1 |
|---|---|---|---|---|---|
| baseline | 101.36s +/- 1.22s | 0.016 | 0.1172 | 0.1964 | 0.369 |
| cache_warm_only | 56.45s +/- 0.16s | 0.016 | 0.1398 | 0.2220 | 0.420 |
| l3_only | 62.03s +/- 0.09s | 0.007 | 0.1334 | 0.2488 | 0.444 |
| l1l3 | 61.76s +/- 0.09s | 0.007 | 0.1334 | 0.2488 | 0.444 |
| l1l2l3 | 10.94s +/- 0.29s | 0.020 | 0.1332 | 0.2038 | 0.356 |
| production_legacy | 73.58s +/- 0.19s | 0.013 | 0.1249 | 0.1853 | 0.461 |
| production_all | 12.83s +/- 0.14s | 0.026 | 0.1205 | 0.1846 | 0.412 |

**Speedups vs baseline (median wall):**
- cache_warm_only: 1.80x faster (-44.3%)
- l3_only: 1.63x faster (-38.8%)
- l1l3: 1.64x faster (-39.1%)
- l1l2l3: 9.27x faster (-89.2%)
- production_legacy: 1.38x faster (-27.4%)
- production_all: 7.90x faster (-87.3%)

**Quality regressions:**
- cache_warm_only: +0.050 F1
- l3_only: +0.074 F1
- l1l3: +0.074 F1
- l1l2l3: -0.013 F1
- production_legacy: +0.092 F1
- production_all: +0.042 F1

**Verdict:**
Faster than baseline: cache_warm_only, l3_only, l1l3, l1l2l3, production_legacy, production_all. Quality preserved: cache_warm_only, l3_only, l1l3, l1l2l3, production_legacy, production_all.
