# V5 Pseudo-code — Stage 4 Contract

```
homotopy_run():
    net = PolicyNet(in=1+2(N-1), out=2(N-1)+1, hidden=192)
    pretrain(net, 800 steps, targets=(s_K=0.40, p_b=0.80, b_next≈0))

    opt   = Adam(net.parameters(), lr=Γ)
    sched = ExponentialLR(opt, γ=0.9998)
    cloud = init_cloud(N_cloud, b_init=0)

    # ---- Phase 1: capital-only (bonds_off=True) ----
    cloud, rec1 = _train_block(net, opt, sched, cloud,
        weights=(1.0, 0.0, 0.0), bonds_off=True, n=2000)

    # ---- Phase 2: bond pretraining (low bond weight) ----
    cloud, rec2 = _train_block(net, opt, sched, cloud,
        weights=(1.0, 0.1, 0.0), bonds_off=False, n=1500)

    # ---- Phase 3: bond homotopy (linear ramp) ----
    schedule(t) = (1.0, 0.1 + 0.9·t/1500, 0.5·t/1500)
    cloud, rec3 = _train_block(net, opt, sched, cloud,
        weight_schedule=schedule, bonds_off=False, n=1500)

    # ---- Phase 4: fine-tuning ----
    for g in opt.param_groups:  g.lr *= 0.10
    cloud, rec4 = _train_block(net, opt, sched, cloud,
        weights=(1.0, 1.0, 0.5), bonds_off=False, n=1500)

    return net, history(phases=[rec1, rec2, rec3, rec4])

_train_block(net, opt, sched, cloud, weights, bonds_off, n, weight_schedule=None):
    rec = []
    for step in 1..n:
        cloud = step_cloud(cloud, net, bonds_off, ψ_K)
        if step % 100 == 0:
            cloud = replace_random(cloud, frac=0.05)
        sample minibatch
        R_K, R_B, FB = euler_residuals(minibatch, net, bonds_off, ψ_K)
        if weight_schedule is None:
            w_K, w_B, w_FB = weights
        else:
            w_K, w_B, w_FB = weight_schedule(step)
        loss = w_K·mean(R_K²) + w_B·mean(R_B²) + w_FB·mean(FB²)
        Adam.step(loss); sched.step()
        rec.append((mean(R_K²), mean(R_B²), mean(FB²), total))
    return cloud, rec
```

## Reduce-to-V4 lever

Set `phase1_steps = phase2_steps = phase3_steps = 0` and `phase4_steps = 8000`, and remove the LR cut. This collapses the homotopy to a single-block run with the V4 loss weights — equivalent to V4's `train.run(...)`.

## Stage 4 exit criterion

The `domain-reviewer` sub-agent must confirm:

- The `bonds_off=True` branch in Phase 1 forces `b_next` to zero in `cohort_decisions`, so the bond portion of the budget identity is identically zero (otherwise gradients leak through unused outputs).
- `weight_schedule` returns weights that are *continuous* across phase boundaries — Phase 2's exit weights $(1, 0.1, 0)$ match Phase 3's entry $(1, 0.1, 0)$.
- Per-phase residuals are recorded as `(R_K_mse, R_B_mse, FB_mse, total)` for the notebook's homotopy plot.
