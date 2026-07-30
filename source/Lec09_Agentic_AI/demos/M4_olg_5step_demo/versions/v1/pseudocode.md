# V1 Pseudo-code — Stage 4 Contract

The structure follows V0; the cohort dimension is now vectorised.

```
# === one-time setup ===
init_cloud(M):
    z_idx ~ Uniform{0, 1} of shape (M,)
    a     ~ shape (M, N-1) — log-spaced base × U[0.75, 1.25] noise per cohort
    return (z_idx, a)

prices(Z, K):
    KL = K / L
    r = α · Z · KL^(α-1) - δ
    w = (1-α) · Z · KL^α
    return r, w

cohort_decisions(Z, a, s):
    """
    Z : shape (M,)              aggregate TFP
    a : shape (M, N-1)          wealth of cohorts age 1..N-1
    s : shape (M, N-1)          savings rates of cohorts age 0..N-2
    """
    K     = a.sum(axis=-1)
    r, w  = prices(Z, K)
    income[..., 0]    = w · ε[0]
    income[..., 1:N]  = w · ε[1:N] + (1+r) · a                    # asset income for cohorts 1..N-1
    a_next            = s · income[..., :N-1]                     # next-period wealth a^1..a^{N-1}
    c[..., :N-1]      = (1 - s) · income[..., :N-1]               # cohorts age 0..N-2
    c[..., N-1]       = income[..., N-1]                          # cohort age N-1 (retired) consumes all
    return c, a_next, r, w, K

PolicyNet(Z, a):
    x = concat([Z, a])                    # input dim = 1 + (N-1) = N
    h = MLP_3x128_Mish(x)
    s = sigmoid(h)                        # output dim = N-1
    return clamp(s, eps, 1-eps)

euler_residuals(z_idx, a, π):
    Z = Z_VALS[z_idx]
    s = π(Z, a)
    today = cohort_decisions(Z, a, s)
    rhs = zeros_like(s)
    for jz in 0..n_z-1:                              # n_z = 2 at V1
        Zp = Z_VALS[jz]
        a_p = today.a_next                            # shape (M, N-1)
        s_p = π(Zp, a_p)
        out_p = cohort_decisions(Zp, a_p, s_p)
        # cohort age j today saves to a^{j+1}_{t+1}; tomorrow they consume c^{j+1}_{t+1}
        c_next = out_p.c[..., 1:N]
        prob = Π[z_idx, jz]
        rhs += prob[..., None] · (1 + out_p.r[..., None]) · c_next^(-γ)
    R = 1 - β · rhs / today.c[..., :N-1]^(-γ)
    return R                                          # shape (M, N-1)

# === pretraining (~600 steps) ===
for step in 1..600:
    (z_idx, a) = init_cloud(batch_size)
    s = π(Z_VALS[z_idx], a)
    loss = mean((s - 0.40)^2)
    Adam.step(loss)

# === main training loop (6,000 steps) ===
(z_idx, a) = init_cloud(N_cloud)
for step in 1..6_000:
    (z_idx, a) = step_cloud(z_idx, a, π)               # advance cloud one period
    if step % 100 == 0:
        replace 5% of cloud with init_cloud(0.05·N_cloud)
    sample minibatch of 256 from cloud
    R = euler_residuals(minibatch, π)                  # shape (256, N-1)
    loss = mean(R^2)
    Adam.step(loss); scheduler.step()

# === diagnostics (Stage 5 verify) ===
- log-scale loss curve
- ergodic simulation: aggregate K path, conditional moments by Z regime
- lifecycle bar charts: mean consumption by age, mean wealth by age, mean savings rate by age
- pointwise Euler-residual histogram on a fresh ergodic cloud
- shock-off sanity check: re-train with Z_lo = Z_hi = 1, verify K_t collapses to a point
```

## Hyperparameters

| | V0 | V1 |
|---|---|---|
| MLP hidden | 3 × 64 | 3 × 128 |
| Cloud size | 512 | 512 |
| Minibatch | 256 | 256 |
| Pretraining | 400, target $(0.30, 0.50)$ | 600, target $0.40$ |
| Main training | 5,000 | 6,000 |

## Stage 4 exit criterion

The cohort-dimension vectorisation must be airtight: an off-by-one in `c[..., 1:N]` versus `c[..., :N-1]` silently solves the wrong Euler equation. The `domain-reviewer` sub-agent must trace at least one cohort's residual end-to-end against `delta_spec.md` §2 before we open the editor.
