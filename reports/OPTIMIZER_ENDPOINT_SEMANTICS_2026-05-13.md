# Optimizer Endpoint Semantics Note, 2026-05-13

## Purpose

Record how to interpret `adam`, `adamw`, and `adam_coupled_decoupled` before starting the first real matched-protocol sweep.

This note exists because the 2 epoch M1B smoke metrics showed that PyTorch `AdamW` and `adam_coupled_decoupled` with `coupled_ratio=0.0`, and PyTorch `Adam` and `adam_coupled_decoupled` with `coupled_ratio=1.0`, are semantically matched endpoints but are not guaranteed to produce bitwise-identical full training trajectories.

## Contract

The project-level interpolation contract is:

```text
total_weight_decay = 5e-4
coupled_ratio = r
wd_coupled = r * total_weight_decay
wd_decoupled = (1 - r) * total_weight_decay

r = 0.0 -> AdamW-style decoupled weight decay
r = 1.0 -> Adam-style coupled weight decay
```

The custom optimizer update uses:

```text
g_eff = grad + wd_coupled * p_old
m = beta1 * m + (1 - beta1) * g_eff
v = beta2 * v + (1 - beta2) * g_eff^2
p_decayed = p_old * (1 - lr * wd_decoupled)
p_new = p_decayed - lr * m_hat / (sqrt(v_hat) + eps)
```

## Confirmed Checks

`code/tests/smoke_checks.py --check optimizer-endpoints` confirms the endpoint semantics on a one-step toy parameter-group setup:

- `coupled_ratio=0.0` matches PyTorch `AdamW`.
- `coupled_ratio=1.0` matches PyTorch `Adam`.

A short ResNet-18 diagnostic check showed:

- r=0 vs PyTorch `AdamW`: step 1 had max parameter difference `0.0`; small differences appeared after subsequent steps.
- r=1 vs PyTorch `Adam`: step 1 had max parameter difference `0.0`; small differences appeared after subsequent steps.

This is consistent with the endpoints having the same intended update semantics while using different implementation paths.

## Interpretation Policy

Use option 2:

- Treat `adam_coupled_decoupled` as the primary controlled interpolation axis.
- Do not require its endpoint runs to be bitwise-identical to PyTorch `AdamW` or `Adam` over full training.
- State that `r=0.0` is AdamW-style and `r=1.0` is Adam-style, not that their full training trajectories are exact clones of PyTorch `AdamW` and `Adam`.

This is the cleaner research axis because every interpolation point, including endpoints, is executed through one optimizer implementation and one weight-decay split contract.

## Reporting Rule

When reporting full experiments:

- `adam` means PyTorch `torch.optim.Adam` with coupled weight decay.
- `adamw` means PyTorch `torch.optim.AdamW` with decoupled weight decay.
- `adam_coupled_decoupled_r0` means the project custom interpolation optimizer at `coupled_ratio=0.0`.
- `adam_coupled_decoupled_r1` means the project custom interpolation optimizer at `coupled_ratio=1.0`.

Do not claim:

```text
adam_coupled_decoupled_r0 == PyTorch AdamW full trajectory
adam_coupled_decoupled_r1 == PyTorch Adam full trajectory
```

Instead claim:

```text
adam_coupled_decoupled_r0 is the AdamW-style endpoint of the controlled interpolation optimizer.
adam_coupled_decoupled_r1 is the Adam-style endpoint of the controlled interpolation optimizer.
```

## Implication For The First 200 Epoch Sweep

For the first real sweep, include both:

- named PyTorch baselines: `adam`, `adamw`
- controlled interpolation points: `adam_coupled_decoupled` with `r in {0.0, 0.25, 0.5, 0.75, 1.0}`

Interpretation should focus on the controlled trend over `r`, while PyTorch `adam` and `adamw` remain external baseline anchors.

## Risk

If endpoint equality becomes important for a table or appendix, add a separate diagnostic that compares:

- one-step equality,
- short fixed-batch multi-step drift,
- full deterministic training drift under identical data order.

Until that diagnostic exists, full-training endpoint equality should be treated as unverified.
