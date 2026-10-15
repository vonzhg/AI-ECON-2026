# AI for Economic Research: Dynamic Models, Language, and Agents

Course site by **Zhigang Feng** — six sessions (32 decks) on what modern AI changes about how
economic research is actually done: an orientation to AI as a tool, an economic object and an agent;
deep learning; reinforcement learning; heterogeneous-agent models; text and large language models;
and retrieval and agents in one research workflow. The organizing premise is that as AI absorbs more
of the implementation, the economist's edge shifts to designing algorithms and validating results.

🔗 **Live site:** https://vonzhg.github.io/AI-ECON-2026/

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/vonzhg/AI-ECON-2026?quickstart=1)

**Prerequisites and where to start.** The course assumes you are comfortable with dynamic
macroeconomic models. Python experience helps but is not required: Session 1's last deck, *The
Workbench*, teaches the Python the labs use and sets up the tools, and the companion course,
[**Quantitative Macroeconomics with AI and Machine
Learning**](https://vonzhg.github.io/Quant_Macro/), teaches it from scratch in [Topic 3, Programming
Basics for Economists](https://vonzhg.github.io/Quant_Macro/syllabus.html#topic-3), and covers the
classical computational methods — dynamic programming, perturbation, projection, parallel
computing — with extensive recordings from previous offerings. The two are designed as one sequence:
classical methods → machine learning → agentic AI research.

The earlier ten-lecture version, taught at Zhejiang University in July 2026, is kept with its decks
at [`archive/summer-2026.html`](archive/summer-2026.html).

> **Unlisted, not secret.** Every page carries `noindex, nofollow` and `robots.txt` disallows
> crawlers, so the site does not show up in search results — but this repository is public, so
> anyone with the link can read it. Treat the slide PDFs accordingly.

## For learners

- **Everything** — syllabus, slides, labs, and project tracks — is linked from the
  [course home page](https://vonzhg.github.io/AI-ECON-2026/).
- **Slides** are free PDF downloads, no password. Each page carries a copyright watermark; please
  don't redistribute or repost them without permission.
- **Labs** run either in **GitHub Codespaces** (click the badge — any GitHub account works, on your
  own free monthly quota; stop idle codespaces) or **locally**:

  ```bash
  git clone https://github.com/vonzhg/AI-ECON-2026.git
  cd AI-ECON-2026
  python3 -m venv .venv                # Windows: py -3.11 -m venv .venv
  source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
  python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
      numpy pandas matplotlib scipy scikit-learn torch jupyter ipykernel
  ```

  Every Session lab runs offline on a laptop CPU with fixed seeds — no API keys, no GPU. Pull before
  each session: the labs are posted to `labs/` as the course goes.

## Structure

```
index.html               Course home (sessions, course map, quick links)
syllabus.html            Full syllabus — six sessions, every deck, resources
slides/                  Session decks and one combined PDF per session (watermarked);
                         SESSIONS.json records each deck's master; Lec*.pdf are the Summer 2026 decks
labs/                    Jupyter notebooks: Session labs as they are posted, and the Summer 2026 labs
capstone.html            Four self-directed project tracks
archive/summer-2026.html Record of the July 2026 offering at Zhejiang University, with its decks
assets/style.css         Shared stylesheet for every page
.devcontainer/           Codespaces environment (Python 3.11 + PyTorch + Jupyter)
requirements.txt          Lab dependencies (CPU PyTorch)
robots.txt               Disallow all crawlers (keep the site unlisted)
.nojekyll                Serve files verbatim (no Jekyll build)
```

Licensing is split: the notebooks and site code fall under the repository `LICENSE`, while the slide
PDFs are **© Zhigang Feng** and shared for personal study only.

## For the instructor — maintain & extend

### Publish

Pages is served from `main` at `/ (root)` (**Settings → Pages**). Push to `main` and the site
refreshes within about a minute. Nothing to build — `.nojekyll` means files are served verbatim.

### Publish or replace Session decks

Decks are **unencrypted but watermarked**. `tools/publish_session_decks.py` stamps the copyright
overlay (`tools/watermark.tex`, the same one the Summer 2026 decks carry) onto every page of each
Session deck, writes it into `slides/` with one combined PDF per session, and records the master in
`slides/SESSIONS.json`. Run it from the instructor's working copy, which holds the deck sources; a
deck whose title changes also needs its row in `slides/index.html` and `syllabus.html`.

### Add a lab

Put the notebook under `labs/` with its asset folder (`labs/sessionNN_assets/`), then change that
session's row in `labs/index.html` from *Released before the session* to links, as the Summer 2026
rows are.

### Run the course again

The live pages carry a **Class Meetings** table for the current offering — one copy in `index.html`
(above the Course Map) and an identical copy in `syllabus.html` (below the Format box). Two steps:

1. Update both tables with the new term's dates, times, rooms, and the `<h2>` term label. Keep the
   two copies byte-identical so the next edit stays a straight copy-paste.
2. When the offering ends, retire it: copy `archive/summer-2026.html` to `archive/<term>.html`, move
   the finished meeting table into it, and add it to the footer link list on the main pages.

Everything else on the live pages is deliberately generic — sessions and decks, no dates — so the
meeting tables are the only place a term shows up.

*Deployment options, the watermark script, and other private notes live in the git-ignored
`DEPLOY_NOTES.local.md`.*
