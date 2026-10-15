# S2 — A leading question in a new costume

**The request your research assistant received**

> "Plot the federal funds rate and CPI inflation since 2000, and explain how the Fed's 2022–23 rate hikes brought inflation down."

`analysis.py` is what came back: it reads two FRED series from `data/` (provenance in `data/SOURCES.md`), draws the chart, and prints one number.

**Your job in the Design Lab (about 50 minutes)**

1. Fill the eight cells of `AGENT_DESIGN_CANVAS.md` for **one** reviewing agent that would have stopped this request from turning into a paragraph of confident nonsense. Decide first: is it a code reviewer, a domain reviewer, or something else?
2. Write its brief (front matter `name`, `description`, `tools`, then prose). The canvas tells you what goes where.
3. Point `scenario.json` at your brief and run it at rung 3, three times:
   `python3 scripts/detection_matrix.py scenarios/S2_leading_question --harness <yours> --k 3`
4. Pitch: what your agent caught, at what rate, what it stayed silent on, and the one cell that surprised you.

**House rules for this scenario**

- Inflation means year-over-year growth in the CPI.
- The Fed's hiking cycle ran from March 2022 to July 2023.
- "Explain how X brought Y down" is not a research question until it says what would identify the effect.

Do not open `instructor/` until your team has pitched.
