# Book code for Session 3 (vendored snapshot)

Source: Miao & Feng, *AI for Economists* (面向经济学家的人工智能), code for chapter 11 (`ch11/`) and
chapter 12 (`ch12/`). Frozen 2026-09-20. `MANIFEST.json` lists every file with the SHA-256 of the book's
original and of this copy. The files are byte-identical to the book's except two one-line import-path
fixes (`ch12/growth_env.py` line 50, `ch12/running_example.py` line 42), both listed in the manifest.

The two `README.md` files are the book's own and are vendored unedited. They still use the chapter
numbers from before the book was renumbered (they say "chapter 12/13" and `code/ch12`, `code/ch13`), so
their commands do not run as written. Working commands, from this folder:

    python3 -B ch11/growth_mdp.py            # environment smoke test and the DP solution
    python3 -B ch11/mc_prediction.py         # first-visit Monte Carlo evaluation
    python3 -B ch11/mc_control.py            # MC control: exploring starts, epsilon-greedy   (~2.5 min)
    python3 -B ch11/running_example.py       # every number in ch. 11's examples, 7x2 grid    (~3 min)
    python3 -B ch12/growth_env.py            # shared environment and the DP benchmark lines
    python3 -B ch12/tabular_td_control.py    # SARSA and Q-learning, 20x2 grid                (~1.5 min)
    python3 -B ch12/linear_td_approx.py      # linear TD and LSTD
    python3 -B ch12/dqn_growth.py            # a minimal DQN                                  (~2 min)
    python3 -B ch12/policy_gradient.py       # REINFORCE and actor-critic                     (~1.5 min)
    python3 -B ch12/portfolio_ac.py          # continuous action: Beta vs sigmoid-Gaussian    (~1.5 min)
    python3 -B ch12/running_example.py       # every number in ch. 12's examples, 7x2 grid    (~4.5 min)

The scripts print in Chinese. `../book_recorded_results.json` holds the full output of each script from one
run on 2026-09-20 (one CPU thread), with hashes; `session03_env.recorded_digest` reads the headline
numbers out of it in English. Dependencies: numpy, and CPU PyTorch for the last four ch. 12 scripts.
