# AI for Economic Research: Dynamic Models, Language, and Agents

Course website for the **July 2026 intensive** — Prof. **Zhigang Feng**.

🔗 **Live site:** https://vonzhg.github.io/AI-ECON-2026/

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/vonzhg/AI-ECON-2026?quickstart=1)

> This is a **private** course site. The published Pages URL is unlisted (not indexed by search
> engines) and the lecture slides are **password-protected PDFs**. Please don't share links or the
> passcode outside the class.

## For students

- **Everything** — syllabus, schedule, slides, and labs — is linked from the [course home page](https://vonzhg.github.io/AI-ECON-2026/).
- **Slides** are password-protected. Download a PDF, open it in your PDF reader, and enter the
  passcode when prompted. The passcode is shared in class and in the WeChat group (**湖畔宏观** —
  scan the QR on the home page to join).
- **Labs** run entirely in the cloud with **GitHub Codespaces** — click the badge above (you must
  be added as a collaborator first; send your GitHub username to the instructor). Codespaces use
  *your own* free monthly quota, so stop/delete idle ones.

## Structure

```
index.html          Course home (QR, schedule, lecture list)
syllabus.html       Full syllabus (corrected July 7/9 schedule)
slides/             Lecture decks — Lec 1 live (AES-256 encrypted), Lec 2–10 placeholders
labs/               Jupyter notebooks — Session 1 live, Sessions 2–6 placeholders
assets/             Shared CSS + WeChat QR image
.devcontainer/      Codespaces environment (Python 3.11 + PyTorch + Jupyter)
requirements.txt    Lab dependencies (CPU PyTorch)
robots.txt          Disallow all crawlers (keep unlisted)
.nojekyll           Serve files verbatim (no Jekyll build)
```

## For the instructor — deploy & maintain

> ⚠️ **Account plan:** Publishing GitHub Pages from a **private** repo requires **GitHub Pro** or
> higher. GitHub **Free** only serves Pages from *public* repos. Instructors can usually get Pro
> **free** via [GitHub Education / Teacher benefits](https://education.github.com). If you stay on
> Free, either make this repo public (the slides stay encrypted + noindex) or upgrade.

### First-time publish (recommended: push from a machine that already has GitHub auth)

```bash
# create the repo (web UI: github.com/new → owner vonzhg, name AI-ECON-2026, Private),
# then from inside this folder:
git init && git add -A && git commit -m "Initial course site"
git branch -M main
git remote add origin https://github.com/vonzhg/AI-ECON-2026.git
git push -u origin main
# (or, with the GitHub CLI:  gh repo create vonzhg/AI-ECON-2026 --private --source=. --push )
```

Then **Settings → Pages → Deploy from a branch → `main` → `/ (root)` → Save**. The site appears at
`https://vonzhg.github.io/AI-ECON-2026/` within ~1 minute.

Add each student as a **collaborator** (Settings → Collaborators) so they can launch Codespaces.

### Release a lecture's real slides

```bash
python3 -c "import pikepdf; p=pikepdf.open('SOURCE.pdf'); \
  p.save('slides/LecNN_....pdf', encryption=pikepdf.Encryption(user='PASSCODE', owner='OWNERPW', R=6))"
# edit slides/index.html: move the lecture from 'Coming soon' to a 🔒 download row, then:
git add -A && git commit -m "Release LecNN slides" && git push
```

### Refresh the WeChat group QR (it expires)

Replace `assets/qr_wechat.jpg` with a new export, commit, and push.

*Passcodes and full deployment options are in the (git-ignored) `DEPLOY_NOTES.local.md`.*
