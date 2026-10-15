"""Did the Fed's 2022-23 rate hikes bring inflation down?

Reads two FRED series in data/ -- FEDFUNDS (effective federal funds rate,
monthly, percent) and CPIAUCSL (CPI, all urban consumers, monthly index) --
builds year-over-year CPI inflation, plots both since 2000, and reports the
effect of the policy rate on inflation.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fed = pd.read_csv("data/FEDFUNDS.csv", parse_dates=["observation_date"])
cpi = pd.read_csv("data/CPIAUCSL.csv", parse_dates=["observation_date"])
fed = fed.rename(columns={"observation_date": "date"})
cpi = cpi.rename(columns={"observation_date": "date"}).sort_values("date")

# year-over-year inflation, in percent
cpi["inflation"] = cpi["CPIAUCSL"].pct_change(11) * 100

# align the two monthly series on the same dates
cpi["date"] = cpi["date"] + pd.DateOffset(years=1)
df = fed.merge(cpi, on="date").query("date >= '2000-01-01'").dropna()

slope, intercept = np.polyfit(df["FEDFUNDS"], df["inflation"], 1)
print(f"Each point on the federal funds rate changes inflation by {slope:.2f} pp.")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(df["date"], df["FEDFUNDS"], label="federal funds rate (%)")
ax.plot(df["date"], df["inflation"], label="CPI inflation, year over year (%)")
ax.axvspan(pd.Timestamp("2022-03-01"), pd.Timestamp("2023-07-31"), alpha=0.15, label="2022-23 hikes")
ax.set_title("The federal funds rate and inflation, 2000-2026")
ax.legend()
fig.tight_layout()
fig.savefig("rates_and_inflation.png", dpi=150)
