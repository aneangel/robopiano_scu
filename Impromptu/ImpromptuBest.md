# Impromptu Best Trajectory Model

Date documented: 2026-05-24

This document describes the current best Impromptu model by the calculations it performs. It intentionally does not use the internal piece-name labels for the component scripts. The model is a simulator-gated trajectory selector over hand-state trajectories, with an optional learned residual inverse-kinematics repair for small low-recall songs.

## Current Confirmed Result

The best confirmed selector was validated on the first 10 completed unseen Maestro-v3 rows from the random30 validation set.

| Run | Selected event F1 |
| --- | ---: |
| maestro30_01 | 0.6153846154 |
| maestro30_02 | 0.7861271676 |
| maestro30_03 | 0.6083916084 |
| maestro30_04 | 0.7080291971 |
| maestro30_05 | 0.7928571429 |
| maestro30_06 | 0.7503692762 |
| maestro30_07 | 0.6459627329 |
| maestro30_08 | 0.7063020214 |
| maestro30_09 | 0.5925925926 |
| maestro30_10 | 0.6908212560 |

Mean event F1 over these 10 rows:

```text
0.6896837611
```

This is still below the target of 0.8, but it is the strongest confirmed Impromptu selector so far.

## State And Target Representation

For each MIDI song, Impromptu converts the score into a target key activation matrix:

```text
Y in {0, 1}^{T x 88}
```

where `Y[t, k] = 1` means piano key `k` should be depressed at control step `t`.

The trajectory planner produces a dense hand-state rollout:

```text
Q in R^{L x 46}
```

where each row `Q[l]` is a 46-dimensional hand joint state, and `L = T * s` for dense substep factor `s`. The dense target matrix is:

```text
Y_dense[l, k] = Y[floor(l / s), k]
```

The simulator evaluates a hand trajectory by rolling out `Q` in the RoboPianist environment and measuring the resulting played-key activation matrix:

```text
P = Sim(Q, Y_dense) in [0, 1]^{L x 88}
```

The same fixed simulator scoring is used for all candidate selection. No candidate is accepted from static geometry alone.

## Event Scoring

A target press event is a rising edge in the target key matrix:

```text
E_Y = {(l, k) : Y_dense[l, k] = 1 and Y_dense[l - 1, k] = 0}
```

A played press event is similarly extracted from simulator output after thresholding:

```text
P_bin[l, k] = 1[P[l, k] > theta]
E_P = {(l, k) : P_bin[l, k] = 1 and P_bin[l - 1, k] = 0}
```

The current threshold is:

```text
theta = 0.5
```

A target event `(l, k)` and played event `(l', k')` can match only when:

```text
k = k'
abs(l - l') * dt <= 0.15 seconds
```

The evaluator forms one-to-one matches under this timing tolerance. Let:

```text
M = number of matched press events
N = number of target press events = |E_Y|
K = number of played press events = |E_P|
```

Then:

```text
precision = M / K
recall    = M / N
F1        = 2 * precision * recall / (precision + recall)
```

Equivalently:

```text
F1 = 2M / (N + K)
```

Mispresses are:

```text
mispresses = K - M
```

The selector optimizes event F1 first, then matched count, then fewer mispresses, then frame-level F1.

## Candidate Trajectory Families

For each song the system builds two main dense hand-state trajectories:

```text
Q_A, Q_B in R^{L x 46}
```

They are produced by different planning biases:

1. `Q_A` is the higher-recall, more aggressive hand trajectory. It tends to hit more intended notes but often creates adjacent-key or duplicate false presses.
2. `Q_B` is the more conservative precision trajectory. It tends to produce fewer false presses but can miss intended target events.

Both trajectories are scored by the same simulator:

```text
score(Q_A) = SimScore(Q_A, Y_dense)
score(Q_B) = SimScore(Q_B, Y_dense)
```

The starting candidate set includes the better of these two under simulator event F1:

```text
Q_base = argmax_{Q in {Q_A, Q_B}} F1(Q)
```

## Local Window Grafting

Most of the current improvement comes from local hand-state window grafting. The model does not blend entire trajectories. Instead, it edits short windows around events and accepts an edit only if a full simulator rollout improves the score.

For an event at dense frame `l`, define a window:

```text
W(l) = [l - a, l + b]
```

with current default:

```text
a = 2 dense frames
b = 10 dense frames
```

Given a source trajectory `Q_src`, a donor trajectory `Q_don`, a window `W`, and a joint-index subset `J`, the grafted candidate is:

```text
Q_trial[r, j] =
    Q_don[r, j] if r in W and j in J
    Q_src[r, j] otherwise
```

The tested joint subsets are:

```text
J = target-hand joints
J = all hand joints
```

The target hand is selected from the event key:

```text
key < split_key  -> left hand joints
key >= split_key -> right hand joints
```

with:

```text
split_key = 48
```

## Recall-Gain Window Pass

The first window pass starts from the conservative precision trajectory and tries to import short windows from the higher-recall trajectory around missed target events.

Let the current trajectory be `Q_cur`. For each missed event `(l, k)` from `score(Q_cur)`, the model evaluates:

```text
Q_trial = graft(Q_cur, Q_high_recall, W(l), J)
```

for each candidate joint subset `J`.

The candidate is accepted if:

```text
F1(Q_trial) > F1(Q_cur)
```

or, with a small tolerance, if it increases the number of matched events:

```text
M(Q_trial) > M(Q_cur)
and
F1(Q_trial) >= F1(Q_cur) - epsilon
```

where:

```text
epsilon = 0.002 for window grafting
```

When accepted:

```text
Q_cur <- Q_trial
```

The process is greedy and simulator-gated; every accepted edit is evaluated in the context of all previously accepted edits.

This pass is useful when the conservative trajectory has high precision but poor recall.

## False-Press Reduction Window Pass

The second window pass runs in the opposite direction. It starts from the higher-recall trajectory and imports short windows from the conservative trajectory around mispress events.

For each false press `(l, k)` in `score(Q_cur)`, it evaluates:

```text
Q_trial = graft(Q_cur, Q_conservative, W(l), J)
```

The candidate is accepted if:

```text
F1(Q_trial) > F1(Q_cur)
```

or if it reduces mispresses without lowering event F1:

```text
F1(Q_trial) >= F1(Q_cur)
and
mispresses(Q_trial) < mispresses(Q_cur)
```

It can also be accepted if matched events increase within the same small F1 tolerance:

```text
M(Q_trial) > M(Q_cur)
and
F1(Q_trial) >= F1(Q_cur) - epsilon
```

This pass is useful when the aggressive trajectory has good recall but too many adjacent-key or duplicate false presses.

## Learned Residual IK Repair For Small Low-Recall Rows

For songs with small target-event counts and low event F1, the selector also tests a learned inverse-kinematics repair.

A key press target gives a desired fingertip position:

```text
x_i^* in R^3
```

for active finger `i`. For inactive fingers, the target is masked out. The IK input is:

```text
X^* in R^{10 x 3}
m in {0, 1}^{10}
q_prev in R^{46}
```

where `m_i = 1` only for constrained fingertips.

The learned policy works in normalized joint space. Let:

```text
z_q(q) = (q - mu_q) / sigma_q
z_x(X) = (X - mu_x) / sigma_x
```

The policy forms features:

```text
phi = concat(flatten(z_x(X^*) * m), m, z_q(q_prev))
```

It predicts a bounded residual:

```text
delta = alpha * tanh(W phi + f_theta(phi))
z_q(q_pred) = z_q(q_prev) + delta
```

where `f_theta` is a multilayer residual network and `alpha` is the action scale.

The policy is trained from RP1M examples with known hand states and fingertip states. It uses:

1. A differentiable forward model:

```text
F_psi(z_q(q)) -> z_x(X)
```

2. Imitation loss against known RP1M hand states:

```text
L_BC = mean(||z_q(q_pred) - z_q(q_expert)||^2)
```

3. Reward-weighted policy updates using fingertip error and smoothness:

```text
e_mean = sum_i m_i ||F_psi(q_pred)_i - x_i^*||_2 / max(1, sum_i m_i)
e_max  = max_i m_i ||F_psi(q_pred)_i - x_i^*||_2
smooth = mean((z_q(q_pred) - z_q(q_prev))^2)
R = -(w_mean * e_mean + w_max * e_max + w_smooth * smooth)
```

At rollout time, the learned IK state is not trusted by itself. It is inserted only as another simulator-gated candidate. For each missed event, the model tries candidate fingers and joint scopes:

```text
J = one finger
J = one hand
J = full hand
```

Each candidate is simulated. It is accepted only under the same event-F1, matched-event, and mispress rules above.

## Final Trajectory Selection

For each song, the selector builds a candidate set:

```text
C = {
    Q_base,
    Q_recall_window,
    Q_false_press_window,
    Q_learned_ik_repair if applicable
}
```

The selected final trajectory is:

```text
Q_final = argmax_{Q in C} (
    F1(Q),
    M(Q),
    -mispresses(Q),
    frame_F1(Q)
)
```

The output `Q_final` is the dense hand-state trajectory used for online rollout. Velocities are recomputed from finite differences after the selected dense joint sequence is finalized.

## Why The Current Model Improved

The older selector averaged:

```text
0.6500640620 event F1
```

on its original 10-song validation.

On the first 10 completed random30 validation rows, the current selector averages:

```text
0.6896837611 event F1
```

The main improvement is that the model no longer commits to one global trajectory style. It uses simulator-scored local edits:

```text
high recall -> lower false positives
high precision -> recover missed events
```

The biggest confirmed gain is on row 05:

```text
F1:        0.7412353923 -> 0.7928571429
matched:   222 / 255   -> 222 / 255
played:    344         -> 305
mispresses 122         -> 83
```

This shows the reverse window pass can reduce false presses while preserving the same number of matched target events.

## Known Failure Modes

The model still does not meet the target of 0.8 event F1.

The remaining errors fall into three categories:

1. Low-recall small songs: there are too few target events, so one missed event has a large effect on F1.
2. Nearby-key false contacts: a fingertip can be close enough to a neighboring key to trigger an extra event even when the intended target is hit.
3. Destructive local edits: a candidate window can recover one key while erasing an already-correct simultaneous or sustained key. The simulator gate rejects many of these, but it also means recall improvements are hard.

The learned IK repair helps some small low-recall rows, but it can also add nearby mispresses. It remains a candidate source, not the dominant planner.

## Queued 30-Song Validation

The 30-song validation has been queued for the current selector.

Branch-generation dependency:

```text
job 1454220
```

Best-selector validation:

```text
job 1454989
script /WAVE/projects/ECEN-524-Wi26/robopiano/run_impromptu_best_selector30_array_cmp.slurm
output /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/impromptu_best_bidirectional_barcarolle_rhapsody_30_20260524
```

Aggregate job:

```text
job 1454990
script /WAVE/projects/ECEN-524-Wi26/robopiano/run_impromptu_best_selector30_aggregate_cmp.slurm
```

The selector array waits for the remaining branch-generation jobs to finish. The aggregate job then computes the 30-song mean event F1, mean frame F1, total matched events, total target events, total played events, and total mispresses.
