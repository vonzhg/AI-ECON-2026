# V2 Pseudo-code — Stage 4 Contract

Identical to V1 except for the TFP discretisation and the (now general) categorical-sampling step.

## TFP discretisation — Rouwenhorst (n=4)

```
rouwenhorst(n, ρ, σ_ε):
    # Recursive transition matrix.
    p = (1 + ρ) / 2
    P = [[p, 1-p], [1-p, p]]
    for k in 3..n:
        P = augment_rouwenhorst(P, p)         # 4 sub-blocks weighted by p, 1-p
    # Grid: equally spaced in [-h, h] of log z.
    σ_z = σ_ε / sqrt(1 - ρ²)
    h   = σ_z · sqrt(n - 1)
    grid_log = linspace(-h, h, n)
    return grid_log, P

aggregate_ar1(ρ_y, σ_y, τ):
    ρ_τ   = ρ_y^τ
    var_ε = σ_y² · (1 - ρ_y^{2τ}) / (1 - ρ_y²)
    return ρ_τ, sqrt(var_ε)

ρ_τ, σ_τ = aggregate_ar1(ρ_y=0.85, σ_y=0.03, τ=72/7)
grid_log, P_MAT = rouwenhorst(n=4, ρ=ρ_τ, σ=σ_τ)
Z_VALS = exp(grid_log)
```

## Cloud step (general categorical)

```
step_cloud(z_idx, a, π):
    Z   = Z_VALS[z_idx]
    s   = π(Z, a)
    out = cohort_decisions(Z, a, s)
    probs = P_MAT[z_idx]                              # shape (M, n_tfp)
    cdf   = cumsum(probs, dim=-1)
    u     = random_uniform(M, 1)
    z_idx_next = (u > cdf).sum(dim=-1).clamp(max=n_tfp - 1)
    return z_idx_next, out.a_next
```

## Euler residual loop

```
euler_residuals(z_idx, a, π):
    Z = Z_VALS[z_idx]
    s = π(Z, a)
    today = cohort_decisions(Z, a, s)
    rhs = zeros_like(s)
    for jz in 0..n_tfp-1:                             # n_tfp = 4 at V2
        Zp     = Z_VALS[jz]
        a_p    = today.a_next
        s_p    = π(Zp, a_p)
        out_p  = cohort_decisions(Zp, a_p, s_p)
        c_next = out_p.c[..., 1:N]
        prob   = Π[z_idx, jz]
        rhs   += prob[..., None] · (1 + out_p.r[..., None]) · c_next^(-γ)
    R = 1 - β · rhs / today.c[..., :N-1]^(-γ)
    return R
```

Everything else (pretraining, main loop, diagnostics) is identical to V1.

## Stage 4 exit criterion

The same vectorisation discipline as V1: an off-by-one in the cohort dimension silently solves the wrong Euler equation. The `domain-reviewer` sub-agent must additionally confirm:

- `P_MAT` has row sums equal to one to floating-point tolerance;
- `Z_VALS` is symmetric in log space around zero;
- the unconditional mean of the Markov chain matches the analytical $\frac{1}{n}\sum z_j$ (exact for symmetric Rouwenhorst).
