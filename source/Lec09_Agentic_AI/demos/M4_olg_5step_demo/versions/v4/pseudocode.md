# V4 Pseudo-code — Stage 4 Contract

Identical to V3 except for the adjustment-cost pieces.

## Cohort decisions

```
cohort_decisions(Z, k, b, s_K, b_next, p_b, bonds_off, ψ_K):
    K     = k.sum(axis=-1)
    r, w  = prices(Z, K)
    inc[..., 0]   = w · ε[0]
    inc[..., 1:N] = w · ε[1:N] + (1+r) · k + (b if not bonds_off else 0)

    bond_cost = (p_b · b_next) if not bonds_off else 0
    k_next    = s_K · inc[..., :N-1]

    # Adjustment cost: cohort age j today (saving into k^{j+1}_{t+1}) compares
    # against their own entering wealth k^j_t (zero for cohort age 0).
    k_aligned = concat([0_vector(M, 1), k[..., :N-2]], axis=-1)
    Δk        = k_next - k_aligned                                     # (M, N-1)
    adj_cost  = 0.5 · ψ_K · Δk²

    c[..., :N-1] = (1 - s_K) · inc[..., :N-1] - bond_cost - adj_cost
    c[..., N-1]  = inc[..., N-1]
    return c, k_next, b_next, r, w, K, Δk
```

## Euler residuals

```
euler_residuals(z_idx, k, b, π, bonds_off, ψ_K):
    Z = Z_VALS[z_idx]
    s_K, b_next, p_b = π(Z, k, b)
    today = cohort_decisions(Z, k, b, s_K, b_next, p_b, bonds_off, ψ_K)

    rhs_K = zeros_like(s_K); rhs_B = zeros_like(s_K)
    for jz in 0..n_tfp-1:
        Zp = Z_VALS[jz]
        s_K_p, b_next_p, p_b_p = π(Zp, today.k_next, today.b_next)
        out_p = cohort_decisions(Zp, today.k_next, today.b_next, s_K_p, b_next_p, p_b_p,
                                 bonds_off, ψ_K)
        c_next  = out_p.c[..., 1:N]
        prob    = Π[z_idx, jz]
        mu_next = c_next^(-γ)
        rhs_K  += prob · (1 + out_p.r) · mu_next
        rhs_B  += prob · mu_next

    mu_today = today.c[..., :N-1]^(-γ)
    marg_cost = 1 + ψ_K · today.Δk                          # < V4 addition
    R_K = marg_cost - β · rhs_K / mu_today                  # capital Euler with adj cost
    R_B = 1 - β · rhs_B / (p_b · mu_today)                  # bond Euler unchanged
    FB  = fb_residual(b_next - b_min, |R_B| + 1e-4)
    return R_K, R_B, FB
```

## Reduce-to-V3 lever

```
hp = {..., psi_K: 0.0}     # restores V3 capital Euler exactly
```

When `psi_K=0`, `marg_cost ≡ 1` and `adj_cost ≡ 0`; the model behaves identically to V3.

## Stage 4 exit criterion

The `domain-reviewer` sub-agent must verify:

- `delta_k[..., 0]` equals `k_next[..., 0]` (cohort age 0 inherits zero capital).
- The capital Euler reduces to V3's $1 - \beta\,\text{rhs}/u'(c)$ when `psi_K = 0`.
- The adjustment cost subtracts from consumption only at the saver cohorts (`c[..., :N-1]`), not at the retired cohort.
