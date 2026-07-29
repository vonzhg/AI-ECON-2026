# AI for Economic Research: Dynamic Models, Language, and Agents

Course site by **Zhigang Feng** — ten lectures (~24 hours) on solving high-dimensional dynamic
models with deep learning and reinforcement learning, turning economic text into measurement with
language models, and running a research pipeline with agentic AI.

🔗 **Live site:** https://vonzhg.github.io/AI-ECON-2026/

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/vonzhg/AI-ECON-2026?quickstart=1)

The material is organized in **six modules rather than calendar dates**, so it works both as a taught
course and as self-study. The schedule of the July 2026 offering is kept for reference at
[`archive/summer-2026.html`](archive/summer-2026.html).

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
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt   # CPU PyTorch — small wheel
  jupyter lab
  ```

  Every notebook is CPU-only and runs offline once its data is bundled. No API keys, no GPU.

## Structure

```
index.html               Course home (modules, lecture list, quick links)
syllabus.html            Full syllabus — six modules, ten lectures, resources
slides/                  Lecture decks (watermarked PDFs); Lec 6 & 10 are placeholders
labs/                    Jupyter notebooks; Lec 6 and Lec 9–10 labs are placeholders
capstone.html            Four self-directed project tracks
archive/summer-2026.html Record of the July 2026 offering (dates, rooms, how it was graded)
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

### Add or replace a lecture deck

Decks are **unencrypted but watermarked**. Stamp the copyright overlay onto every page of the source
PDF, save it into `slides/`, then add or update the deck's row in `slides/index.html`. The watermark
recipe (a `pikepdf` overlay script) is in the git-ignored `DEPLOY_NOTES.local.md`.

When a lecture moves from placeholder to real:

1. Drop the watermarked PDF into `slides/`.
2. In `slides/index.html`, move the lecture out of the *In preparation* section into its own
   `<h2>` + `res-row` block.
3. In `syllabus.html`, replace the `in preparation` topic-deck link with direct PDF links.
4. In `index.html`, flip that lecture card's badge from `badge-soon` to `badge-ready`.

### Add a lab

Put the notebook under `labs/` (or a `labs/LecNN_*/` folder with its data and helper modules), then
add a `res-row` to `labs/index.html` keyed by lecture, and mention it in the matching module of
`syllabus.html`.

### Run the course again

Nothing on the live pages needs editing — they carry no dates. Record the offering instead: copy
`archive/summer-2026.html` to `archive/<term>.html`, fill in the meeting table, and add it to the
footer link list on the main pages.

*Deployment options, the watermark script, and other private notes live in the git-ignored
`DEPLOY_NOTES.local.md`.*
