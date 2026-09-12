"""Fill a scenario's detection matrix at rung 3, on any harness, k times per cell.

    python3 scripts/detection_matrix.py scenarios/S1_aiyagari_audit --harness claude --k 3

Every run is one headless process in a fresh folder that holds only the files
under review -- a new session by construction, and nothing to leak the answer
(see instructor/ANSWER_KEY.md, "Two leakage channels").  Each agent's brief is
its Markdown file; the runner asks it to end with line-tied findings, then
scores them against the scenario's planted defects and its control line.

Reading the matrix (TERMINAL_WALKTHROUGH.md, Step 7): the diagonal should fire,
off-diagonal cells may or may not, and the control row must stay silent.  Report
rates over k runs, not the best run -- a reviewer that is right once in three is
a different instrument from one that is right three times in three.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, LAB)
import agent_lab  # noqa: E402

FINDINGS = re.compile(r"<findings>\s*(\[.*?\])\s*</findings>", re.S)
FORMAT = (
    "\n\nEnd your reply with a machine-readable list of the defects you would stand behind, "
    "each tied to one line of the file under review, exactly in this form:\n"
    '<findings>[{"line": 12, "claim": "one sentence"}]</findings>\n'
    "An empty list, <findings>[]</findings>, is a valid and respectable answer."
)


def brief_model(path: str) -> str | None:
    """The `model:` line of an agent file's front matter, if any."""
    text = open(path, encoding="utf-8").read()
    if text.startswith("---"):
        m = re.search(r"^model:\s*(\S+)\s*$", text[: text.find("\n---", 3)], re.M)
        if m:
            return m.group(1)
    return None


def brief_body(path: str) -> str:
    """The prose of an agent file, without its YAML front matter."""
    text = open(path, encoding="utf-8").read()
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4:].strip()
    return text.strip()


def parse_findings(reply: str) -> list[dict]:
    matches = FINDINGS.findall(reply)
    if not matches:
        return []
    try:
        items = json.loads(matches[-1])
    except json.JSONDecodeError:
        return []
    return [i for i in items if isinstance(i, dict) and isinstance(i.get("line"), int)]


def hits(findings: list[dict], lines: list[int], tol: int) -> bool:
    return any(abs(f["line"] - l) <= tol for f in findings for l in lines)


def control_fired(findings: list[dict], control: dict) -> bool:
    """A control is a correct line, or a stated scope, that must not be reported as a defect."""
    if control.get("lines") and hits(findings, control["lines"], 0):
        return True
    pattern = control.get("pattern")
    return bool(pattern) and any(re.search(pattern, f.get("claim", ""), re.I) for f in findings)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", help="scenario folder containing scenario.json")
    ap.add_argument("--harness", default="claude", choices=sorted(agent_lab.HARNESSES))
    ap.add_argument("--k", type=int, default=3, help="fresh runs per agent (default 3)")
    ap.add_argument("--agents", nargs="+", help="subset of the scenario's agents")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    sdir = os.path.abspath(args.scenario)
    spec = json.load(open(os.path.join(sdir, "scenario.json"), encoding="utf-8"))
    agents = {n: a for n, a in spec["agents"].items() if not args.agents or n in args.agents}
    tol = spec.get("tolerance", 1)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(LAB, "results", os.path.basename(sdir), f"{args.harness}-{stamp}")
    os.makedirs(outdir, exist_ok=True)

    with tempfile.TemporaryDirectory() as stage:
        for rel in spec["files"]:
            shutil.copy(os.path.join(sdir, rel), stage)
        pre = spec.get("precompute")
        if pre:  # run the code once, so every reviewer can stay read-only
            proc = subprocess.run(pre["command"], cwd=stage, capture_output=True, text=True,
                                  timeout=pre.get("timeout", 600))
            with open(os.path.join(stage, pre["save_as"]), "w", encoding="utf-8") as fh:
                fh.write(f"$ {' '.join(pre['command'])}\n--- stdout ---\n{proc.stdout}"
                         f"--- stderr ---\n{proc.stderr}--- exit code: {proc.returncode} ---\n")

        table = {n: {"runs": 0, "failed": 0, "defects": {d["id"]: 0 for d in spec["defects"]},
                     "controls": [0] * len(spec["controls"]), "other": 0} for n in agents}
        for name, agent in agents.items():
            brief = os.path.join(sdir, agent["brief"])
            prompt = brief_body(brief) + "\n\n## Task\n\n" + agent["task"] + FORMAT
            model = brief_model(brief) if args.harness == "claude" else None
            for run in range(1, args.k + 1):
                with tempfile.TemporaryDirectory() as work:
                    for f in os.listdir(stage):
                        shutil.copy(os.path.join(stage, f), work)
                    try:
                        reply = agent_lab.harness_ask(args.harness, prompt, cwd=work, timeout=args.timeout,
                                                      read_only=True, model=model)
                    except RuntimeError as exc:
                        table[name]["failed"] += 1
                        print(f"  {name} run {run}: FAILED -- {str(exc)[:160]}")
                        continue
                with open(os.path.join(outdir, f"{name}_run{run}.md"), "w", encoding="utf-8") as fh:
                    fh.write(reply)
                found = parse_findings(reply)
                row = table[name]
                row["runs"] += 1
                matched = set()
                for d in spec["defects"]:
                    if hits(found, d["lines"], tol):
                        row["defects"][d["id"]] += 1
                        matched.update(f["line"] for f in found
                                       if any(abs(f["line"] - l) <= tol for l in d["lines"]))
                for i, control in enumerate(spec["controls"]):
                    if control_fired(found, control):
                        row["controls"][i] += 1
                row["other"] += len([f for f in found if f["line"] not in matched])
                print(f"  {name} run {run}: {len(found)} findings", flush=True)

    summary = {"scenario": spec["name"], "harness": args.harness, "k": args.k,
               "harness_status": agent_lab.HARNESSES[args.harness].status, "when": stamp, "table": table}
    json.dump(summary, open(os.path.join(outdir, "summary.json"), "w"), indent=2)

    names = list(agents)
    width = max(len(d["label"]) + 9 for d in spec["defects"] + spec["controls"]) + 2
    print(f"\n{spec['name']} -- harness {args.harness}, k = {args.k}")
    print(f"{'':{width}}" + "".join(f"{n:>16}" for n in names))
    for d in spec["defects"]:
        cells = ""
        for n in names:
            r = table[n]
            mark = "*" if d.get("owner") == n else " "
            cells += f"{r['defects'][d['id']]:>11}/{r['runs']}{mark:<3}"
        print(f"{d['label']:{width}}" + cells)
    for i, control in enumerate(spec["controls"]):
        print(f"{'control: ' + control['label']:{width}}"
              + "".join(f"{table[n]['controls'][i]:>11}/{table[n]['runs']}   " for n in names))
    print(f"{'other findings (read them)':{width}}"
          + "".join(f"{table[n]['other']:>15} " for n in names))
    print("\n* = the agent the scenario expects to catch that defect.  Replies: " + os.path.relpath(outdir, LAB))
    return 0


if __name__ == "__main__":
    sys.exit(main())
