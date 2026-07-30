# V0 Pseudo-code — Stage 4 Contract

```
# === one-time setup ===
init_cloud(N) → returns (z_idx, a_m, a_o)
                z_idx ~ Uniform{0, 1}
                a_m   ~ U[0.05, 0.35]
                a_o   ~ U[0.10, 0.60]

prices(Z, K):
    KL = K / L
    r = α·Z·KL^(α-1) - δ
    w = (1-α)·Z·KL^α
    return r, w

cohort_decisions(Z, a_m, a_o, s_y, s_m):
    K = a_m + a_o
    r, w = prices(Z, K)
    a_m_next = s_y · w · ε_y
    c_y      = (1 - s_y) · w · ε_y
    inc_m    = w · ε_m + (1+r) · a_m
    a_o_next = s_m · inc_m
    c_m      = (1 - s_m) · inc_m
    c_o      = (1+r) · a_o
    return (c_y, c_m, c_o, a_m_next, a_o_next, r, w)

PolicyNet(Z, a_m, a_o):
    x = stack([Z, a_m, a_o])
    h = MLP_3x64_Mish(x)
    s = sigmoid(h)              # ∈ (0, 1)^2
    return s_y, s_m   (clamped to [eps, 1-eps])

euler_residuals(z_idx, a_m, a_o, π):
    Z = Z_VALS[z_idx]
    s_y, s_m = π(Z, a_m, a_o)
    today    = cohort_decisions(Z, a_m, a_o, s_y, s_m)
    rhs_y, rhs_m = 0, 0
    for jz in range(n_z):              # n_z = 2 at V0
        Zp     = Z_VALS[jz]
        Kp     = today.a_m_next + today.a_o_next
        rp, wp = prices(Zp, Kp)
        c_o_p  = (1+rp) · today.a_o_next                              # old at t+1
        s_y_p, s_m_p = π(Zp, today.a_m_next, today.a_o_next)
        inc_m_p = wp · ε_m + (1+rp) · today.a_m_next
        c_m_p   = (1 - s_m_p) · inc_m_p                               # middle at t+1
        prob    = Π[z_idx, jz]
        rhs_y  += prob · (1+rp) · u'(c_m_p)
        rhs_m  += prob · (1+rp) · u'(c_o_p)
    R_y = 1 - β · rhs_y / u'(today.c_y)
    R_m = 1 - β · rhs_m / u'(today.c_m)
    return R_y, R_m

# === pretraining (~400 steps) ===
for step in 1..400:
    (z_idx, a_m, a_o) = init_cloud(batch_size)
    s_y, s_m = π(Z_VALS[z_idx], a_m, a_o)
    loss = mean((s_y - 0.30)^2 + (s_m - 0.50)^2)
    Adam.step(loss)

# === main training loop (5,000 steps) ===
(z_idx, a_m, a_o) = init_cloud(N_cloud)
for step in 1..5_000:
    # 1. advance cloud one period under current π (no grad)
    (z_idx, a_m, a_o) = step_cloud(z_idx, a_m, a_o, π)

    # 2. anti-collapse: replace 5% of cloud with random restarts
    if step % 100 == 0:
        replace 5% of (z_idx, a_m, a_o) with init_cloud(0.05·N_cloud)

    # 3. minibatch Euler-residual update
    sample minibatch of 256 from cloud
    R_y, R_m = euler_residuals(minibatch, π)
    loss = mean(R_y² + R_m²)
    Adam.step(loss)
    scheduler.step()                    # exponential decay 0.9998 per step

# === diagnostics (Stage 5 verify) ===
- log-scale loss curve
- ergodic simulation (5,000 periods, 500-period burn-in)
- lifecycle consumption profile bar chart
- policy-function slices s_y(K), s_m(K) at each Z
- pointwise Euler-residual histogram on a fresh ergodic cloud
- shock-off sanity check: re-train with Z_lo = Z_hi = 1, verify K_t collapses to a point
```

## Hyperparameters

| | Value |
|---|---|
| MLP hidden | 3 layers × 64 units, Mish, sigmoid head |
| Cloud size | 512 |
| Minibatch | 256 |
| Learning rate | $10^{-3}$ |
| LR decay | $0.9998$ per step |
| Pretraining | 400 steps, target $(s_y, s_m) = (0.30, 0.50)$ |
| Main training | 5,000 steps |
| Random restart | 5% of cloud every 100 steps |
| Optimiser | Adam (default $\beta_1, \beta_2$) |

## Stage 4 exit criterion

Read the pseudo-code aloud and defend every choice. If anything is unclear, edit `model_spec.md` *before* changing source files — the spec is the contract.
