# Pre-registration: does FOMC statement text add anything to a yield forecast?

Session 5 lab, block F. Written 2026-09-20.

This file is the design. It was settled, written out in full, and hashed **before the first
scored run of the test window**; `protocol_forecast.sha256` carries that hash and the moment it
was taken. Nothing below was written after seeing a test-window result. The hash exists so that a
reader can tell the difference, and the whole point of the exercise is that the answer may well
be "text adds nothing" — that outcome is expected, reportable, and passes every gate in §10.

The object is Session 2's one-month-ahead forecast of the 10-year Treasury yield. Session 2 built
it from numbers alone. Block F asks whether the FOMC's own words, available at the moment the
forecast is made, improve on it.

---

## 1. Data, and what is read only

| File | Rows | Read as |
|---|---|---|
| `labs/session02_assets/baseline_predictions.csv` | 79 + header | read-only, hash-asserted |
| `labs/session02_assets/yield_panel.csv` | 439 + header | read-only, hash-asserted |
| `labs/session05_assets/fomc/*.txt` | 245 statements | read-only, hash-asserted via `fomc_manifest.json` |
| `labs/session05_assets/llm_stance/scores.csv` | when present | read-only, reported separately (§8) |

1.1 Both Session 2 files are inputs. Block F never rewrites them, never re-runs Session 2's
training, and never fills a missing value in either.

1.2 Their SHA-256 digests are asserted at load:

```
baseline_predictions.csv  a8e815d45edc48872cedeb37bd12e725b46c7c2b5cc99294837c2bc48c923332
yield_panel.csv           b1687bce6a37f4270a9014d89e7ef1b5351aca982dbfff8c77d36c2e1ca37e64
```

1.3 The panel carries no row for the forecast date **2025-11-30**. That is the only month-end
missing anywhere in the panel, and Session 2's test window is 80 calendar months of which 79
survive. Block F **inherits the gap and never fills it**. Any interpolation would manufacture an
observation that Session 2's published baselines were not scored on, and the two would stop being
comparable.

## 2. Estimation sample, and why it is not Session 2's

2.1 The panel begins in 1989. The first FOMC statement in the corpus is dated **1994-02-04**.
Of Session 2's 300 training rows, **50 predate that statement** and could not carry a text signal
under any rule. The two claims "Session 2's splits are unchanged" and "the text sample runs
1994–2014" cannot both be true.

2.2 The design resolves this by restriction, not by imputation. The estimation sample is every row
with

```
forecast_made >= 1994-02-28
```

which leaves **250 training rows** (target dates through 2014-12-31), **60 validation rows**
(2015–2019) and **79 test rows** (2020 onward), 389 in all.

2.3 Because the comparator is no longer estimated on what Session 2 estimated it on, the
structured models are **re-estimated on those rows**, and every result in block F is reported in
**three columns**:

| Column | What it is |
|---|---|
| **frozen file** | Session 2's published forecasts, scored as shipped in `baseline_predictions.csv` |
| **re-estimated** | the same model forms refitted on rows from 1994-02-28 on, scored on the same 79 test rows |
| **plus text** | the re-estimated model with the text regressors of §6 added |

2.4 The frozen-to-re-estimated gap is the cost of the sample restriction and has nothing to do
with text. Reporting it in its own column is what keeps it from being read as a text effect. A
comparison that quoted only the first and third columns would attribute the sum of two changes to
one of them.

## 3. Session 2's frozen baselines, and the bar to beat

3.1 Scoring the columns of `baseline_predictions.csv` over the 79 test rows must return:

| Model | RMSE | MAE |
|---|---|---|
| random walk | 0.2658 | 0.2097 |
| AR(1) | 0.2647 | 0.2089 |
| OLS-5 | 0.2855 | 0.2255 |

These three are **gates**: they reproduce to four decimals or the run stops.

3.2 The MLP row is a named exception, because Session 2 has two different MLP numbers and they
are not interchangeable:

- Session 2 **publishes** RMSE **0.3078** / MAE **0.2406**. That is the *mean of ten per-seed
  test RMSEs*, with the per-seed RMSE ranging 0.2661 to 0.3588.
- The `mlp_seed_mean` column of `baseline_predictions.csv` is a different object: the *mean
  forecast across the ten seeds*. Scoring that column gives RMSE **0.2956** / MAE **0.2289**.

3.3 Block F records both, each under its own name, and gates neither against the other. Scoring
the column and expecting the published figure would be an error of definition, not a failure to
reproduce.

3.4 The gap between them is forecast combination doing its ordinary work: averaging ten forecasts
(0.2956) beats the average of the ten models (0.3078). When Session 5 quotes "the bar to beat" it
quotes Session 2's **published** definition, 0.3078 / 0.2406.

3.5 Note for the record: on this test window the random walk and AR(1) beat OLS-5, and OLS-5 beats
the MLP. The bar to beat is therefore a very low-variance forecast, and any text model has to
improve on *that*, not on the weakest structured model available.

## 4. The information rule

4.1 For a panel row whose forecast is made at date *t* (the last observation date of month *t*),
the text signal is the **last policy statement dated on or before *t***.

4.2 Seven months in the corpus carry two or three statements — 2001-01, 2007-08 (three),
2008-01, 2008-03, 2008-10, 2019-10, and 2020-03 (three). **The last one on or before *t* wins.**
Earlier statements in the same month are not averaged in: a forecaster standing at the end of the
month has read the latest one, and averaging would blend a superseded stance into the current one.

4.3 No statement dated after *t* may reach any feature of row *t*. This is asserted in code, not
assumed.

## 5. The exclusion set

Not every release in the corpus is a policy statement. The following are **hand-listed** and
removed before the rule of §4 is applied. This list is part of the pre-registration and is hashed
with this file.

| Date | File | Why it is not a policy statement |
|---|---|---|
| 2025-08-22 | `fomc_20250822.txt` | Announces revisions to the *Statement on Longer-Run Goals and Monetary Policy Strategy* — the 2025 framework review. It is a statement about the Committee's framework, and takes no position on the current stance. |
| 2019-10-11 | `fomc_20191011.txt` | *Statement Regarding Monetary Policy Implementation*: reserve-management bill purchases and repo operations. The release says in terms that these are "purely technical measures … and do not represent a change in the stance of monetary policy." |
| 2010-05-09 | `fomc_20100509.txt` | Re-authorisation of dollar liquidity swap lines with four central banks. A balance-sheet action with no rate decision and no stance language. |
| 2008-03-11 | `fomc_20080311.txt` | Announcement of the Term Securities Lending Facility and an increase in swap lines, issued jointly with four other central banks. A facility announcement, not a stance decision. |
| 2007-08-10 | `fomc_20070810.txt` | Notice that the Desk will supply reserves to keep the funds rate near an **unchanged** 5-1/4 percent target. A market-functioning notice, not a policy decision. |

5.1 The corpus was read for this list, not sampled. The 245 files were searched for releases
lacking policy-rate language and for framework, minutes and facility vocabulary; every candidate
was read in full before being listed or kept. Statements that merely *read* unusually — the
1994–95 releases phrased in terms of "the degree of pressure on reserve positions", and the
2007-08-17 and 2020-03-03 inter-meeting moves — are genuine policy decisions and are **kept**.

5.2 Only the 2025-08-22 exclusion changes the test window; the other four fall in the training
window, or are superseded later in their own month.

## 6. Stale months, and how a month without a meeting is handled

6.1 The Committee does not meet every month, so many rows carry a statement from an earlier month.
In the 79 test rows this is **25 rows** under the raw corpus; applying the exclusion set of §5
makes it **26**, because August 2025's only release is the framework-review document and the month
therefore falls back to 2025-07-30. Both counts are recorded.

6.2 Three rules govern such a row:

- **Stance is carried forward.** The last stance is still the prevailing stance; that is what a
  reader of the statement would believe.
- **Novelty is 0.** The novelty index measures how much a *new* statement differs from the one
  before it. In a month with no meeting there is no new statement, so there is nothing to be
  novel. Carrying novelty forward would be wrong by construction: it would report a surprise that
  the market had already had a month or more to absorb.
- **"Months since the last statement" enters as its own regressor**, so that staleness is a
  measured feature of the row rather than a silent distortion of the other two.

## 7. Fitting discipline

7.1 Every fitted object — the vocabulary, the IDF weights, the term co-occurrence matrix, the SVD,
the seed axis, every scaler, and the choice of model and penalty — is estimated on the **training
window** (text dated 1994-01-01 to 2014-12-31; panel rows with target dates through 2014-12-31) or
chosen on the **validation window**, 2015–2019.

7.2 The code asserts that **no document dated after the end of the training window reaches any
fitted object**, and that no panel row outside the training window contributes to any scaler,
coefficient or vocabulary. These are assertions, not comments.

7.3 The penalty of the ridge in §8 is chosen on 2015–2019 **only**, by validation RMSE. The test
window plays no part in any choice.

## 8. The models, named in advance

Named here, before any of them is scored:

- **(a) text alone.** The change in the yield, `y10_next − y10`, on the text regressors: stance,
  novelty, and months since the last statement.
- **(b) OLS-5 plus text.** Session 2's five structured predictors — `y10`, `spread`,
  `fedfunds_lag1`, `unrate_lag1`, `inflation_lag1` — re-estimated on the §2 sample, with the same
  three text regressors added.
- **TF-IDF enters through ridge**, never through unpenalised least squares: the vocabulary is far
  larger than 250 training rows. Its penalty is chosen on 2015–2019 and nowhere else.

8.1 **The LLM column is reported on its own and is never pooled into model selection.** The
stance scores being recorded into `llm_stance/` are a separate measurement of the same statements;
they are read from a tidy `llm_stance/scores.csv` with columns `date,variant,label` when that file
exists. Only variant **A**, the frozen prompt run on every statement, may enter a forecast;
variants B, C and D exist to separate prompt sensitivity from sampling noise and are reported as
measurement diagnostics. If `scores.csv` is absent the LLM column is recorded as unavailable with
a plain note saying so, and the rest of block F runs unchanged. Nothing about the choice between
models (a) and (b), or about the ridge penalty, may depend on the LLM column.

8.2 **Session 1's Machine B is excluded by assertion.** That classifier was trained on all 245
statements, test window included, and anything derived from it would be contaminated end to end.
Block F asserts that no artefact of it is loaded.

## 9. Comparing forecasts

9.1 **Diebold–Mariano with the Harvey–Leybourne–Newbold small-sample correction.** 79
observations is small enough that the uncorrected statistic over-rejects; the corrected statistic
is referred to a *t* distribution with *n* − 1 degrees of freedom.

9.2 **Clark–West alongside it for nested comparisons.** When one model nests the other — and
"OLS-5" inside "OLS-5 plus text" is exactly that — Diebold–Mariano is not correctly sized under
the null, because the larger model estimates parameters that are zero in population and is
therefore *expected* to lose. Clark–West adjusts for that. Both statistics are reported for every
nested pair; neither is reported alone.

9.3 Both are written in numpy. Neither source was read at first hand for this implementation;
both are implemented from their standard statements and a separate check of the references is
under way. Each docstring names its source and repeats that caveat.

9.4 A **cumulative squared-error series** is recorded for every model, so that the comparison can
be read as a path rather than a single number. A difference in RMSE that is produced entirely by
March 2020 is a different finding from one accumulated steadily, and only the path shows which.

## 10. Gates, and what counts as a result

10.1 The gates are on **procedure only**:

1. Both Session 2 files match the hashes in §1.2.
2. The three reproducible frozen baselines of §3.1 match to four decimals.
3. The splits have the sizes stated in §2.2 — 250 / 60 / 79 — and the 2025-11-30 gap is present
   and unfilled.
4. No document dated after the training window reaches any fitted object (§7.2).
5. No artefact of Session 1's Machine B is loaded (§8.2).
6. The test window is scored **once**, in a single pass, after every choice has been made.

10.2 **No gate depends on text helping.** "Text adds nothing" is an acceptable, expected, and
fully reportable outcome. It is the outcome the design is built to be able to state honestly.
There is no gate anywhere in block F whose passing requires a text model to beat a structured one,
and none may be added later.

10.3 The results are reported **as found**, in the three columns of §2.3, with the LLM column
beside them and not inside them.

## 11. Scoring the test window once

11.1 The test window is scored in **one pass**, after §7 and §8 are complete. The numbers from
that pass are written to `recorded_forecast.json` and are not recomputed by any later choice.

11.2 The notebook **recomputes the validation metrics live** — those are cheap and carry no
inferential weight — and then **asserts that the test-window numbers it holds equal the recorded
file**. A mismatch is a failure, not a new result. This is what makes "scored once" checkable by a
reader rather than merely asserted by the author.

11.3 If the design has to change after this file is hashed, the change is recorded as an amendment
with its own date and hash, beneath the original. The original is never edited.
