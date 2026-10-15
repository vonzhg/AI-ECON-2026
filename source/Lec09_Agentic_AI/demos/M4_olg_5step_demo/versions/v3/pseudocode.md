# V3 Pseudo-code — Stage 4 Contract

```
# === policy network ===
PolicyNet(Z, k, b):
    x = concat([Z, k, b])                          # input dim 1 + 2(N-1)
    h = MLP_3x192_Mish(x)
    s_K_raw, b_raw, p_raw = split(h)               # sizes (N-1, N-1, 1)
    s_K  = sigmoid(s_K_raw).clamp(eps, 1-eps)
    b̃    = tanh(b_raw) · b_scale
    b_next = b̃ - mean(b̃, axis=-1)                # market clearing: sum=0 by construction
    p_b  = p_b_min + (p_b_max - p_b_min) · sigmoid(p_raw)
    return s_K, b_next, p_b

# === cohort decisions (period-t bookkeeping) ===
cohort_decisions(Z, k, b, s_K, b_next, p_b, bonds_off):
    K     = k.sum(axis=-1)
    r, w  = prices(Z, K)
    inc[..., 0]   = w · ε[0]
    inc[..., 1:N] = w · ε[1:N] + (1+r) · k + (b if not bonds_off else 0)

    bond_cost = (p_b · b_next) if not bonds_off else 0
    k_next    = s_K · inc[..., :N-1]
    c[..., :N-1] = (1 - s_K) · inc[..., :N-1] - bond_cost
    c[..., N-1]  = inc[..., N-1]                                   # retired consumes all
    return c, k_next, b_next, r, w, K

# === Euler residuals ===
euler_residuals(z_idx, k, b, π, bonds_off):
    Z = Z_VALS[z_idx]
    s_K, b_next, p_b = π(Z, k, b)
    today = cohort_decisions(Z, k, b, s_K, b_next, p_b, bonds_off)

    rhs_K = zeros_like(s_K)
    rhs_B = zeros_like(s_K)
    for jz in 0..n_tfp-1:
        Zp = Z_VALS[jz]
        s_K_p, b_next_p, p_b_p = π(Zp, today.k_next, today.b_next)
        out_p = cohort_decisions(Zp, today.k_next, today.b_next, s_K_p, b_next_p, p_b_p, bonds_off)
        c_next = out_p.c[..., 1:N]
        prob = Π[z_idx, jz]
        mu_next = c_next^(-γ)
        rhs_K += prob · (1 + out_p.r) · mu_next
        rhs_B += prob · mu_next
    R_K = 1 - β · rhs_K / today.c[..., :N-1]^(-γ)
    R_B = 1 - β · rhs_B / (p_b · today.c[..., :N-1]^(-γ))      # bond Euler

    # Fischer–Burmeister: borrowing limit b_next ≥ b_min
    slack       = b_next - b_min
    multiplier  = |R_B| + 1e-4
    FB          = slack + multiplier - sqrt(slack² + multiplier² + 1e-12)
    return R_K, R_B, FB

# === pretraining (~800 steps) ===
for step in 1..800:
    (z_idx, k, b) = init_cloud(batch_size, b_init=0)
    s_K, b_next, p_b = π(Z_VALS[z_idx], k, b)
    loss = mean((s_K - 0.40)²)
         + mean((p_b - 0.80)²)
         + 0.1 · mean(b_next²)                 # bonds start near zero
    Adam.step(loss)

# === main training loop (8,000 steps) ===
(z_idx, k, b) = init_cloud(N_cloud, b_init=0)
for step in 1..8_000:
    (z_idx, k, b) = step_cloud(z_idx, k, b, π, bonds_off)
    if step % 100 == 0:
        replace 5% with init_cloud(0.05·N_cloud)
    sample minibatch of 256
    R_K, R_B, FB = euler_residuals(minibatch, π, bonds_off)
    loss = w_K·mean(R_K²) + w_B·mean(R_B²) + w_FB·mean(FB²)
    Adam.step(loss); scheduler.step()
```

## Loss weights

| | Default V3 | reduce-to-V2 |
|---|---|---|
| $w_K$ | 1.0 | 1.0 |
| $w_B$ | 1.0 | 0.0 |
| $w_{FB}$ | 0.5 | 0.0 |
| `bonds_off` flag | False | True |

## Stage 4 exit criterion

The market-clearing layer is the most error-prone piece. The `domain-reviewer` sub-agent must verify in writing:

- `b_next.sum(axis=-1)` is zero up to floating-point precision after every forward pass.
- `bonds_off=True` in `cohort_decisions` causes `bond_cost` to drop out of the budget and `b_next` to be returned as a zero tensor.
- Setting `w_B = w_FB = 0` and `bonds_off=True` recovers V2's loss expression exactly.
