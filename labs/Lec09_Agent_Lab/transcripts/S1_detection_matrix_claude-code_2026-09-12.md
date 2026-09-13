# S1 detection matrix — reference run

Recorded 2026-09-12 with `scripts/detection_matrix.py` on Claude Code 2.1.269–2.1.270, each agent on the model its brief names (`sonnet`), three fresh headless runs per agent, each run in a folder holding only `aiyagari_solver.py` and its `run_log.txt`. Scored against `scenarios/S1_aiyagari_audit/scenario.json` (convergence lines 31, 44, 58, 64, 72; domain lines 34, 35, 51, 52; tolerance 1).

| | tooling-agent | math-agent | econ-agent |
|---|---|---|---|
| the loop never tests a convergence criterion | **3/3** | 3/3 | 2/3 |
| a utility function evaluated outside its domain | 3/3 | **3/3** | 0/3 |
| *control:* `EV = P @ V.T`, which is correct | 0/3 | 0/3 | 0/3 |
| *control:* partial equilibrium, stated in the docstring | 0/3 | 0/3 | 0/3 |

Bold: the agent the scenario expects to catch that defect.

**What the runs show**

- The two specialists caught both planted defects every time and never flagged a correct line or the stated scope.
- `econ-agent` did the job its brief gives it: it asked economics questions (is the asset-grid ceiling of 20 high enough when β(1+r) = 0.998? is r = 4% too close to 1/β − 1?) rather than hunting code bugs, and it never reported partial equilibrium as a defect. In an earlier single run on the default model it did cite the negative-consumption line — the model named in the brief changes what the agent finds.
- Two of econ-agent's first three runs failed with a transient CLI error; the runner now retries once, and the column above is from a clean re-run.
- One scoring lesson: a finding at line 58 ("discards V_new without ever comparing it to V") is the convergence defect cited where the missing check belongs. The scenario's line list was widened and the runs were re-scored with `--rescore`, not re-run.
