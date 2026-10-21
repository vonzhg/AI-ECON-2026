"""Session 6 lab -- the agent half: the loop, and the rules that bound it.

Block A  The loop and the stub model, adapted from the Summer 2026 agent lab (labs/Lec09_Agent_Lab/agent_lab.py).
         A ScriptedModel replays a fixed sequence of tool calls, so the control flow -- which is the real thing --
         runs with no key, no network and no GPU. Token accounting is kept per turn and cumulatively, because the
         whole message list is re-sent every turn and that is where an agent's price comes from. trace_rows()
         prints a run in Deck 6.1's shape: your instruction, each tool call as one JSON line, each observation.
         Two real tools, each a function plus a typed JSON schema: retrieval over the FOMC statements, and a
         numerical tool over Session 2's yield panel. The bounded task is one question -- what the Committee
         said about inflation at a given meeting, and what the ten-year yield did over the following month,
         with the statement cited.
Block E  The rules, written as code rather than as advice: a step and token budget, an allow-list of tools, a
         PreToolUse gate, and a verifier that recomputes every number the agent asserts. Each one is shown
         firing, and the gate is also shown NOT firing -- the matcher that silently never runs is the lesson.
         Then reliability over the three scenarios recorded for the Summer 2026 agent lab, and the redaction
         proof for the transcripts shipped beside this file.

The instructor's record -- the function that writes recorded_agent.json, and the audit of a recorded run that
an earlier version of this lab taught -- is kept with the instructor's tools, outside the lab.

Nothing here imports anthropic, openai, langchain or crewai, and nothing reaches the network.
Imports: numpy, pandas and the standard library.
"""
from __future__ import annotations

import csv
import fnmatch
import hashlib
import json
import math
import re
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LABS = HERE.parent
REPO = LABS.parent
TRANSCRIPTS = HERE / "transcripts"
RECORDED = HERE / "recorded_agent.json"
YIELD_PANEL = LABS / "session02_assets" / "yield_panel.csv"
# The Summer 2026 agent lab's recorded run summaries. Its results/ folder is not part of the course site, so the eight
# summary.json files block E reads were copied beside this module on 21 Oct 2026 (same bytes); a clone has them.
LEC09_RESULTS = HERE / "lec09_summaries"

# The corpus is shared, so it is looked for rather than assumed. The first folder that exists
# and holds statements wins; none of them is written to from here.
FOMC_DIRS = (HERE / "fomc", LABS / "session05_assets" / "fomc", LABS / "Lec07_LLM_Lab" / "data" / "fomc")

SEED = 20260920             # for a notebook that wants one; nothing in this module is random
BASIS_POINTS = 100.0         # a percentage point is a hundred basis points


# ======================================================================================
# Block A -- token accounting (A4)
# ======================================================================================
# Four characters to a token, the same crude rule as the Lecture 9 lab. A real tokenizer is
# model-specific; this is enough to make the SHAPE of the cost curve right, which is the only
# claim being made. The recorded transcripts carry the harness's own exact counts for contrast.
CHARS_PER_TOKEN = 4


def count_tokens(obj: Any) -> int:
    """Approximate token count of any JSON-serialisable message structure."""
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, default=str)
    return max(1, len(text) // CHARS_PER_TOKEN)


def conversation_tokens(messages: Sequence[dict]) -> int:
    """Tokens in the WHOLE list -- what gets re-sent on the next turn."""
    return count_tokens(list(messages))


@dataclass
class TurnCost:
    """One turn's place in the ledger: what went up, what came back, what it has cost so far."""
    turn: int
    tokens_sent: int         # the entire message list, re-sent
    tokens_new: int          # what this turn added to it
    tokens_out: int          # the model's reply
    tool_calls: int
    cumulative_sent: int
    cumulative_out: int

    def as_dict(self) -> dict:
        return {"turn": self.turn, "tokens_sent": self.tokens_sent, "tokens_new": self.tokens_new,
                "tokens_out": self.tokens_out, "tool_calls": self.tool_calls,
                "cumulative_sent": self.cumulative_sent, "cumulative_out": self.cumulative_out}


def resend_cost(n_turns: int, new_tokens_per_turn: int) -> List[int]:
    """Cumulative tokens SENT when the whole list is re-sent every turn.

    Turn k sends a list of size k, so the running total grows with the square of the turn
    count even though each turn adds a fixed amount of new text.
    """
    cumulative, total = [], 0
    for k in range(1, n_turns + 1):
        total += new_tokens_per_turn * k
        cumulative.append(total)
    return cumulative


def price(tokens_in: int, tokens_out: int, usd_per_mtok_in: float, usd_per_mtok_out: float) -> float:
    """The ledger in dollars. Rates are an input, not a constant: they are dated facts."""
    return (tokens_in * usd_per_mtok_in + tokens_out * usd_per_mtok_out) / 1e6


# ======================================================================================
# Block A -- tools are code plus a schema (A1)
# ======================================================================================
@dataclass
class Tool:
    """A tool is code plus a typed schema. The model picks the arguments; the code runs."""
    name: str
    description: str
    input_schema: dict
    fn: Callable[..., str]

    def to_api_dict(self) -> dict:
        """Exactly the shape the Messages API expects in tools=[...]."""
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


@dataclass
class Permissions:
    """An allow-list of tool names. Everything not named is refused."""
    allow: Tuple[str, ...]

    def refuse(self, name: str) -> str:
        """The refusal text, or '' when the call is permitted."""
        if name in self.allow:
            return ""
        return (f"Error: {name!r} is not on this agent's allow-list "
                f"{sorted(self.allow)}. The call was refused before the tool ran.")


@dataclass
class Budget:
    """A ceiling on turns and on tokens sent. An agent without one runs until the money stops."""
    max_steps: int = 8
    max_tokens: int = 60_000

    def exceeded(self, turn: int, cumulative_sent: int) -> str:
        if turn > self.max_steps:
            return f"step budget reached: {self.max_steps} turns"
        if cumulative_sent > self.max_tokens:
            return f"token budget reached: {cumulative_sent} > {self.max_tokens} tokens sent"
        return ""


class ToolRegistry:
    """The tools an agent may call, plus the permission check and the gate that guard them."""

    def __init__(self, tools: Sequence[Tool] | None = None,
                 permissions: Permissions | None = None,
                 hooks: Sequence["HookSpec"] | None = None):
        self._tools: Dict[str, Tool] = {t.name: t for t in (tools or [])}
        self.permissions = permissions
        self.hooks: List["HookSpec"] = list(hooks or [])
        self.blocked: List[dict] = []       # every call the gate or the allow-list stopped

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def names(self) -> List[str]:
        return sorted(self._tools)

    def to_api_list(self) -> List[dict]:
        return [self._tools[n].to_api_dict() for n in sorted(self._tools)]

    def run(self, name: str, args: dict) -> Tuple[str, bool]:
        """Execute a tool. Returns (result_text, is_error).

        A refused, blocked or failing call must still come back as a tool_result: dropping it
        would leave a tool_use block unanswered and the API would reject the next request.
        """
        if self.permissions is not None:
            refusal = self.permissions.refuse(name)
            if refusal:
                self.blocked.append({"tool": name, "by": "allow-list", "reason": refusal})
                return (refusal, True)
        for hook in self.hooks:
            if not hook_fires(hook, name, args):
                continue
            reason = hook.check(name, args)
            if reason:
                self.blocked.append({"tool": name, "by": f"hook:{hook.name}", "reason": reason})
                return (f"Blocked by {hook.name} (PreToolUse, exit 2): {reason}", True)
        if name not in self._tools:
            return (f"Error: no tool named {name!r}.", True)
        try:
            return (str(self._tools[name].fn(**args)), False)
        except Exception as exc:                  # noqa: BLE001 - shown to the model, not hidden
            return (f"Error: {type(exc).__name__}: {exc}", True)


# ======================================================================================
# Block A -- the stub model (A2)
# ======================================================================================
@dataclass
class Response:
    """The fields of a real Messages API response that the loop actually reads."""
    content: List[dict]
    stop_reason: str


@dataclass
class ScriptedModel:
    """Replays a fixed script of responses, ignoring the prompt.

    Not a language model. It exists so the surrounding loop -- which IS the real thing -- can
    be run, priced and inspected deterministically, with no key and no network.
    """
    script: List[Response]
    name: str = "scripted"
    calls: int = 0
    seen_message_counts: List[int] = field(default_factory=list)

    def create(self, *, system: str, tools: List[dict], messages: List[dict]) -> Response:
        self.seen_message_counts.append(conversation_tokens(messages))
        if self.calls >= len(self.script):
            self.calls += 1
            return Response(content=[text_block("(script exhausted)")], stop_reason="end_turn")
        resp = self.script[self.calls]
        self.calls += 1
        return resp


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def tool_use_block(block_id: str, name: str, args: dict) -> dict:
    return {"type": "tool_use", "id": block_id, "name": name, "input": args}


def scripted(*turns) -> ScriptedModel:
    """Build a ScriptedModel from (text, [(tool, args), ...]) pairs; a turn with no call ends it."""
    script = []
    for i, (said, calls) in enumerate(turns, start=1):
        blocks: List[dict] = [text_block(said)] if said else []
        for j, (tool, args) in enumerate(calls, start=1):
            blocks.append(tool_use_block(f"call{i}_{j}", tool, args))
        script.append(Response(content=blocks, stop_reason="tool_use" if calls else "end_turn"))
    return ScriptedModel(script=script)


# ======================================================================================
# Block A -- the loop (A2, A3)
# ======================================================================================
def run_agent(model, registry: ToolRegistry, system: str, user_prompt: str,
              budget: Budget | None = None, trace: bool = True) -> dict:
    """The agentic loop, with the budget checked before every request.

    Structurally identical to the Messages API version: only model.create is stubbed. Returns
    the trace the notebook prices and the verifier reads.
    """
    budget = budget or Budget()
    messages: List[dict] = [{"role": "user", "content": user_prompt}]
    ledger: List[TurnCost] = []
    cum_sent = cum_out = 0
    final_text, stop = "", "end_turn"

    for turn in range(1, budget.max_steps + 2):
        tokens_sent = conversation_tokens(messages)
        hit = budget.exceeded(turn, cum_sent + tokens_sent)
        if hit:
            stop = "budget"
            if trace:
                print(f"  stopped: {hit}")
            break

        resp = model.create(system=system, tools=registry.to_api_list(), messages=messages)
        tool_uses = [b for b in resp.content if b.get("type") == "tool_use"]
        tokens_out = count_tokens(resp.content)
        prior = ledger[-1].tokens_sent if ledger else 0
        cum_sent += tokens_sent
        cum_out += tokens_out
        ledger.append(TurnCost(turn=turn, tokens_sent=tokens_sent, tokens_new=tokens_sent - prior,
                               tokens_out=tokens_out, tool_calls=len(tool_uses),
                               cumulative_sent=cum_sent, cumulative_out=cum_out))

        if trace:
            said = " ".join(b["text"] for b in resp.content if b.get("type") == "text")
            print(f"  turn {turn}: sent {tokens_sent:>5} tok (+{tokens_sent - prior:>4} new) | "
                  f"out {tokens_out:>4} | {len(tool_uses)} tool call(s)"
                  + (f' | "{said[:44]}"' if said else ""))

        if resp.stop_reason != "tool_use":
            final_text = " ".join(b["text"] for b in resp.content if b.get("type") == "text")
            stop = resp.stop_reason
            break

        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for b in tool_uses:
            out, is_err = registry.run(b["name"], b["input"])
            if trace:
                flag = " [error]" if is_err else ""
                print(f"       -> {b['name']}({json.dumps(b['input'])[:52]}) = {out[:44]}{flag}")
            results.append({"type": "tool_result", "tool_use_id": b["id"], "content": out,
                            "is_error": is_err})
        messages.append({"role": "user", "content": results})

    return {"messages": messages, "ledger": [t.as_dict() for t in ledger], "final_text": final_text,
            "turns": len(ledger), "stop": stop, "blocked": list(registry.blocked),
            "tokens_sent_total": cum_sent, "tokens_out_total": cum_out}


def trace_rows(run: dict, width: int = 90) -> List[Tuple[str, str]]:
    """A run in Deck 6.1's shape: your instruction, each tool call as one JSON line, each observation.

    The JSON line is the slide's stylised form, {"tool": name, ...arguments}. The message list holds the
    same call as the API carries it: a tool_use block with an id, a name and an input.
    """
    short = lambda text: textwrap.shorten(" ".join(str(text).split()), width, placeholder=" ...")
    rows: List[Tuple[str, str]] = []
    for message in run["messages"]:
        content = message["content"]
        if isinstance(content, str):
            rows.append(("you", short(content)))
            continue
        for block in content:
            if block.get("type") == "tool_use":
                rows.append(("model ->", short(json.dumps({"tool": block["name"], **block["input"]}))))
            elif block.get("type") == "tool_result":
                text = str(block["content"])
                if block.get("is_error") and text.startswith("Error: ") and "allow-list" in text:
                    text = "refused: " + text[len("Error: "):]
                elif block.get("is_error") and text.startswith("Blocked by "):
                    text = "refused by a gate: " + text[len("Blocked by "):]
                rows.append(("harness <-", short(text)))
    rows.append(("model", short("no tool call: " + run["final_text"])))
    return rows


# ======================================================================================
# Replaying a recorded run through the same loop (the instructor's record)
# ======================================================================================
class RecordedTools(ToolRegistry):
    """Answers each call with the result the recorded run got, in order.

    Nothing executes -- no request, no write, no shell. A call that does not match the
    recording comes back as an error, so a replay that drifts says so instead of inventing
    an answer.
    """

    def __init__(self, calls: Sequence[Tuple[str, str, bool]]):
        super().__init__()
        self._calls = list(calls)
        self._next = 0

    def names(self) -> List[str]:
        return sorted({name for name, _, _ in self._calls})

    def to_api_list(self) -> List[dict]:
        return [{"name": n, "description": "recorded", "input_schema": {"type": "object"}}
                for n in self.names()]

    def run(self, name: str, args: dict) -> Tuple[str, bool]:
        if self._next >= len(self._calls):
            return ("Error: the recording has no more tool results.", True)
        want, content, is_error = self._calls[self._next]
        self._next += 1
        if name != want:
            return (f"Error: replay drifted -- recorded call {self._next} was {want!r}, "
                    f"not {name!r}.", True)
        return (content, is_error)


def load_transcript(path) -> Tuple[ScriptedModel, RecordedTools, dict]:
    """Rebuild one of the redacted transcripts as a script plus its tool results.

    The file holds one JSON row per block. Consecutive text and tool_use rows form one model
    response; the tool_result rows after them answer its calls in order. turn_usage rows carry
    the harness's own token counts, one per API request, and are collected into the meta dict
    rather than replayed.
    """
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    script: List[Response] = []
    calls: List[Tuple[str, str, bool]] = []
    blocks: List[dict] = []
    pending: List[str] = []
    usage: List[dict] = []
    prompts: List[str] = []
    meta: dict = {}

    def close_response() -> None:
        if blocks:
            stop = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
            script.append(Response(content=list(blocks), stop_reason=stop))
            blocks.clear()

    for row in rows:
        kind = row.get("type")
        if kind == "prompt":
            prompts.append(row["text"])
        elif kind == "text":
            blocks.append(text_block(row["text"]))
        elif kind == "tool_use":
            blocks.append(tool_use_block(f"call{row['n']}", row["name"], row["input"]))
            pending.append(row["name"])
        elif kind == "tool_result":
            close_response()
            name = pending.pop(0) if pending else "?"
            calls.append((name, str(row.get("content", "")), bool(row.get("is_error"))))
        elif kind == "turn_usage":
            usage.append({k: v for k, v in row.items() if k != "type"})
        elif kind == "result":
            meta = {k: v for k, v in row.items() if k != "type"}
    close_response()
    meta = dict(meta, prompts=prompts, turn_usage=usage)
    return ScriptedModel(script=script, name="recorded"), RecordedTools(calls), meta


def transcript_usage(path) -> dict:
    """The harness's own token ledger for a recorded run: per request, and the totals."""
    _, _, meta = load_transcript(path)
    rows = meta["turn_usage"]
    keys = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
            "output_tokens", "thinking_tokens")
    totals = {k: int(sum(r.get(k, 0) for r in rows)) for k in keys}
    totals["requests"] = len(rows)
    totals["tokens_in_total"] = (totals["input_tokens"] + totals["cache_creation_input_tokens"]
                                 + totals["cache_read_input_tokens"])
    result = {k: meta.get(k) for k in ("duration_ms", "duration_api_ms", "num_turns",
                                       "total_cost_usd", "stop_reason") if meta.get(k) is not None}
    # A stream-json trace carries a partial usage on most of its rows, so the per-request sum
    # is not the run's cost. The closing result row is, and the two are reported side by side
    # rather than reconciled: which record you read decides what number you quote.
    closing = meta.get("usage") or {}
    authoritative = {k: closing.get(k, 0) for k in
                     ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
                      "output_tokens")}
    if closing:
        authoritative["thinking_tokens"] = (closing.get("output_tokens_details") or {}).get(
            "thinking_tokens", 0)
        authoritative["tokens_in_total"] = (authoritative["input_tokens"]
                                            + authoritative["cache_creation_input_tokens"]
                                            + authoritative["cache_read_input_tokens"])
    by_model = {m: {k: v for k, v in u.items()
                    if k in ("inputTokens", "outputTokens", "cacheReadInputTokens",
                             "cacheCreationInputTokens", "thinkingTokens", "costUSD")}
                for m, u in (meta.get("modelUsage") or {}).items()}
    return {"per_request": rows, "totals": totals, "prompts": len(meta["prompts"]),
            "harness_result": result,
            "harness_usage_closing_row": authoritative,
            "harness_usage_by_model": by_model,
            "note": ("per_request sums the usage on each API request row; "
                     "harness_usage_closing_row is the harness's own closing total. They agree "
                     "on a session file and differ on a stream-json trace, whose intermediate "
                     "rows carry a partial usage.")}


# ======================================================================================
# Block A -- the two real tools
# ======================================================================================
def fomc_dir() -> Path:
    """The first FOMC statement folder that exists. Read-only: this file never writes there."""
    for d in FOMC_DIRS:
        if d.is_dir() and any(d.glob("fomc_*.txt")):
            return d
    raise FileNotFoundError(f"no FOMC statement folder among {[str(d) for d in FOMC_DIRS]}")


@lru_cache(maxsize=1)
def _read_corpus() -> pd.DataFrame:
    """Read the corpus off disk once. Callers get a copy, so no notebook can poison the cache."""
    rows = []
    for p in sorted(fomc_dir().glob("fomc_*.txt")):
        stamp = re.search(r"(\d{8})", p.name).group(1)
        raw = p.read_text(encoding="utf-8")
        body = "\n".join(l for l in raw.splitlines() if not l.startswith("#")).strip()
        source = ""
        m = re.search(r"^#\s*source:\s*(\S+)", raw, flags=re.M)
        if m:
            source = m.group(1)
        rows.append({"date": f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}", "file": p.name,
                     "source": source, "text": body})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def load_corpus() -> pd.DataFrame:
    """Every statement as (date, file, source, text), sorted by date."""
    return _read_corpus().copy()


_WORD = re.compile(r"[a-z']+")


def _terms(text: str) -> Counter:
    return Counter(_WORD.findall(text.lower()))


def fomc_search(query: str, start: str = "", end: str = "", k: int = 3) -> str:
    """Rank statements by query-term frequency, with the date window applied first.

    Deliberately the plainest possible retrieval: the lab's subject is the LOOP, and a tool
    whose ranking cannot be inspected would hide which document the answer came from.
    """
    corpus = load_corpus()
    if start:
        corpus = corpus[corpus.date >= start]
    if end:
        corpus = corpus[corpus.date <= end]
    if corpus.empty:
        return json.dumps({"error": f"no statements between {start!r} and {end!r}", "hits": []})
    wanted = [w for w in _WORD.findall(query.lower()) if len(w) > 2]
    hits = []
    for _, row in corpus.iterrows():
        counts = _terms(row.text)
        score = sum(counts[w] for w in wanted)
        if score:
            hits.append({"date": row.date, "score": int(score), "source": row.source,
                         "snippet": _snippet(row.text, wanted)})
    hits.sort(key=lambda h: (-h["score"], h["date"]))
    return json.dumps({"query": query, "window": [start, end], "n_searched": int(len(corpus)),
                       "hits": hits[:k]}, ensure_ascii=False)


def _snippet(text: str, wanted: Sequence[str], width: int = 240) -> str:
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if any(w in sentence.lower() for w in wanted):
            return sentence[:width]
    return text[:width]


def fomc_statement(date: str, topic: str = "") -> str:
    """One statement, whole or reduced to the sentences that mention a topic, with its URL."""
    corpus = load_corpus()
    row = corpus[corpus.date == date]
    if row.empty:
        near = corpus[corpus.date.str[:7] == date[:7]]
        return json.dumps({"error": f"no statement dated {date}",
                           "same_month": near.date.tolist()})
    row = row.iloc[0]
    sentences = re.split(r"(?<=[.!?])\s+", row.text.replace("\n", " "))
    if topic:
        keep = [s for s in sentences if topic.lower() in s.lower()]
    else:
        keep = sentences
    return json.dumps({"date": row.date, "source": row.source, "topic": topic or "(whole statement)",
                       "n_sentences": len(keep), "text": " ".join(keep).strip()}, ensure_ascii=False)


@lru_cache(maxsize=1)
def _read_panel() -> pd.DataFrame:
    return pd.read_csv(YIELD_PANEL)


def load_panel() -> pd.DataFrame:
    """Session 2's monthly yield panel, 439 rows, month-end dates. A copy, not the cache."""
    return _read_panel().copy()


def yield_change(meeting_date: str) -> str:
    """The ten-year yield over the month AFTER the meeting's month, month-end to month-end.

    The panel is monthly, so the window it can report is not "one month from the meeting": it
    is from the end of the meeting's month to the end of the next. The tool returns both dates
    and the lag in days so that the answer cannot quietly be written as though the window
    started on the day the Committee met. That distance is the whole point of the check.
    """
    panel = load_panel()
    month = meeting_date[:7]
    row = panel[panel.date.str.startswith(month)]
    if row.empty:
        return json.dumps({"error": f"no panel row for {month}",
                           "panel_range": [panel.date.iloc[0], panel.date.iloc[-1]]})
    row = row.iloc[0]
    change = float(row.y10_next) - float(row.y10)
    lag = (pd.Timestamp(row.date) - pd.Timestamp(meeting_date)).days
    return json.dumps({"meeting_date": meeting_date, "window_start": row.date,
                       "window_end": row.target_date, "y10_start": float(row.y10),
                       "y10_end": float(row.y10_next),
                       "change_pct_points": round(change, 4),
                       "change_basis_points": round(change * BASIS_POINTS, 1),
                       "days_from_meeting_to_window_start": int(lag),
                       "caveat": ("the window begins at the end of the meeting's month, "
                                  f"{lag} days after the meeting itself")})


def panel_stats(column: str, start: str = "", end: str = "") -> str:
    """Mean, standard deviation and count of one panel column over a date window."""
    panel = load_panel()
    if column not in panel.columns:
        return json.dumps({"error": f"no column {column!r}", "columns": list(panel.columns)})
    sub = panel
    if start:
        sub = sub[sub.date >= start]
    if end:
        sub = sub[sub.date <= end]
    values = pd.to_numeric(sub[column], errors="coerce").dropna()
    if values.empty:
        return json.dumps({"error": f"no rows for {column!r} between {start!r} and {end!r}"})
    return json.dumps({"column": column, "window": [start, end], "n": int(values.size),
                       "mean": round(float(values.mean()), 4),
                       "sd": round(float(values.std(ddof=1)), 4),
                       "min": round(float(values.min()), 4), "max": round(float(values.max()), 4)})


def write_table(file_path: str, content: str) -> str:
    """The write the gate guards. Nothing reaches the disk: the lab keeps the bytes in memory."""
    WRITTEN[file_path] = content
    return json.dumps({"wrote": file_path, "bytes": len(content.encode()),
                       "note": "held in memory for the lab; nothing was written to disk"})


WRITTEN: Dict[str, str] = {}


# --- the schemas, typed, exactly as the API expects them -------------------------------
FOMC_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "words to score statements by"},
        "start": {"type": "string", "description": "earliest meeting date, YYYY-MM-DD"},
        "end": {"type": "string", "description": "latest meeting date, YYYY-MM-DD"},
        "k": {"type": "integer", "minimum": 1, "maximum": 20, "description": "how many hits"},
    },
    "required": ["query"],
}

FOMC_STATEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "date": {"type": "string", "description": "meeting date, YYYY-MM-DD"},
        "topic": {"type": "string", "description": "keep only sentences containing this word"},
    },
    "required": ["date"],
}

YIELD_CHANGE_SCHEMA = {
    "type": "object",
    "properties": {
        "meeting_date": {"type": "string", "description": "meeting date, YYYY-MM-DD"},
    },
    "required": ["meeting_date"],
}

PANEL_STATS_SCHEMA = {
    "type": "object",
    "properties": {
        "column": {"type": "string",
                   "enum": ["y10_next", "y10", "spread", "fedfunds_lag1", "unrate_lag1", "inflation_lag1"]},
        "start": {"type": "string", "description": "earliest panel date, YYYY-MM-DD"},
        "end": {"type": "string", "description": "latest panel date, YYYY-MM-DD"},
    },
    "required": ["column"],
}

WRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string", "description": "where the table goes"},
        "content": {"type": "string", "description": "the file's whole contents"},
    },
    "required": ["file_path", "content"],
}


def build_tools() -> List[Tool]:
    """The four tools the bounded task needs, plus the write the gate is there to catch."""
    return [
        Tool("fomc_search", "Rank FOMC statements by how often they use the query's words.",
             FOMC_SEARCH_SCHEMA, fomc_search),
        Tool("fomc_statement", "Return one FOMC statement, or only its sentences on a topic, with its URL.",
             FOMC_STATEMENT_SCHEMA, fomc_statement),
        Tool("yield_change", "Ten-year Treasury yield over the month after a meeting, from the Session 2 panel.",
             YIELD_CHANGE_SCHEMA, yield_change),
        Tool("panel_stats", "Mean, sd, min and max of one Session 2 panel column over a window.",
             PANEL_STATS_SCHEMA, panel_stats),
        Tool("Write", "Write a file. Guarded by the source-column gate on data/*.csv.",
             WRITE_SCHEMA, write_table),
    ]


# ======================================================================================
# Block E -- the mechanical gate (A5, E2)
# ======================================================================================
# Claude Code's documented semantics, and the trap in them:
#
#   matcher  filters a hook GROUP by TOOL NAME only. "*", "" or none: every tool. Only letters,
#            digits, _, -, spaces, "," and "|": an exact name, or a list of them (Bash, Edit|Write).
#            Anything else: a regular expression that may match anywhere in the name (mcp__.*).
#   if       filters an individual hook by tool name AND arguments, in permission-rule syntax.
#
# So a matcher written Write(data/*.csv) -- permission-rule syntax in the matcher's slot -- is
# read as a regular expression, a valid one, that no tool name can ever match. It raises no
# error, logs nothing, and never fires. That was found on 12 September 2026 and is the reason
# the broken form is kept below beside the working one: a rule that silently never fires is
# worse than no rule, because it is mistaken for one.
# Source: Claude Code hooks reference, the matcher table and the `if` field
# (https://code.claude.com/docs/en/hooks, read 20 October 2026).

GATE_SETTINGS_BROKEN = {
    "hooks": {
        "PreToolUse": [
            {"matcher": "Write(data/*.csv)",
             "hooks": [{"type": "command", "command": "python3 checks/require_source_column.py"}]}
        ]
    }
}

GATE_SETTINGS_WORKING = {
    "hooks": {
        "PreToolUse": [
            {"matcher": "Write",
             "hooks": [{"type": "command", "if": "Write(data/*.csv)",
                        "command": "python3 checks/require_source_column.py"}]}
        ]
    }
}


_PLAIN_MATCHER = re.compile(r"[A-Za-z0-9_\- ,|]*")


def matcher_matches(matcher: str, tool_name: str) -> bool:
    """A hook group's matcher, as the hooks reference defines it: tested against the TOOL NAME only.

    "*", "" or no matcher matches every tool. A matcher of only letters, digits, _, -, spaces, "," and
    "|" is an exact name, or a list of exact names. Anything else -- parentheses, a slash, a dot, a
    star -- is a regular expression that may match anywhere in the name (JavaScript's RegExp.test;
    Python's re.search does the same here). A pattern that does not compile matches nothing, which is
    the other way a hook can be silent -- so the caller is told by gate_report(), not left to guess.
    """
    if matcher in ("", "*"):
        return True
    if _PLAIN_MATCHER.fullmatch(matcher):
        return tool_name in {part.strip() for part in re.split(r"[|,]", matcher) if part.strip()}
    try:
        return re.search(matcher, tool_name) is not None
    except re.error:
        return False


def _glob_segment_match(pattern: str, value: str) -> bool:
    """Permission-rule globbing: * stays inside one path segment, ** crosses them."""
    if "**" in pattern:
        return fnmatch.fnmatchcase(value, pattern.replace("**", "*"))
    if len(pattern.split("/")) != len(value.split("/")):
        return False
    return all(fnmatch.fnmatchcase(v, p) for p, v in zip(pattern.split("/"), value.split("/")))


def rule_matches(rule: str, tool_name: str, tool_input: dict) -> bool:
    """An `if` rule in permission-rule syntax: Tool, or Tool(argument-pattern)."""
    m = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\((.*)\))?\s*", rule)
    if not m:
        return False
    want, pattern = m.group(1), m.group(2)
    if want != tool_name:
        return False
    if pattern is None:
        return True
    argument = str(tool_input.get("file_path") or tool_input.get("path")
                   or tool_input.get("command") or "")
    return _glob_segment_match(pattern.strip(), argument)


@dataclass
class HookSpec:
    """One PreToolUse hook: the group's matcher, the hook's own `if`, and the check it runs."""
    name: str
    matcher: str
    check: Callable[[str, dict], str]        # returns '' to allow, or the reason to block
    if_rule: str = ""

    @classmethod
    def from_settings(cls, settings: dict, name: str, check: Callable[[str, dict], str]) -> "HookSpec":
        """Read one PreToolUse entry out of a settings.json-shaped dict, matcher and `if` alike."""
        entry = settings["hooks"]["PreToolUse"][0]
        inner = (entry.get("hooks") or [{}])[0]
        return cls(name=name, matcher=entry.get("matcher", ""), check=check,
                   if_rule=inner.get("if", ""))


def hook_fires(hook: HookSpec, tool_name: str, tool_input: dict) -> bool:
    """Whether this hook's process would spawn for this call: matcher first, then `if`."""
    if not matcher_matches(hook.matcher, tool_name):
        return False
    if hook.if_rule and not rule_matches(hook.if_rule, tool_name, tool_input):
        return False
    return True


def require_source_column(tool_name: str, tool_input: dict) -> str:
    """The check itself: a CSV of results must carry a source column, and it must be filled.

    The same shape as scenarios/S3_fellows_provenance/checks/require_source_url.py -- read the
    pending call's arguments, return a reason, and let the caller exit 2 on it.
    """
    content = str(tool_input.get("content", ""))
    path = str(tool_input.get("file_path", ""))
    if not path.endswith(".csv"):
        return ""
    rows = list(csv.DictReader(content.splitlines()))
    if not rows:
        return f"{path}: no data rows to check"
    header = list(rows[0])
    if not any(h.lower().startswith("source") for h in header):
        return f"{path}: no source column among {header}"
    column = next(h for h in header if h.lower().startswith("source"))
    empty = [i for i, r in enumerate(rows, start=2) if not (r.get(column) or "").strip()]
    if empty:
        return f"{path}: {column} is empty on line(s) {empty}"
    return ""


# Two candidate calls: the one the rule is meant to catch, and one it is meant to leave alone.
GATE_PROBES = [
    {"label": "the call the rule is meant to catch",
     "tool": "Write", "input": {"file_path": "data/answer.csv",
                                "content": "meeting_date,change_bp\n2022-03-16,57.0\n"}},
    {"label": "the same write, now carrying its source",
     "tool": "Write", "input": {"file_path": "data/answer.csv",
                                "content": "meeting_date,change_bp,source\n2022-03-16,57.0,"
                                           "labs/session02_assets/yield_panel.csv\n"}},
    {"label": "a write outside data/",
     "tool": "Write", "input": {"file_path": "notes/scratch.csv", "content": "a,b\n1,2\n"}},
    {"label": "a different tool entirely",
     "tool": "Bash", "input": {"command": "rm data/answer.csv"}},
]


def gate_report() -> dict:
    """Run both settings against the probes and count what each one would actually stop.

    The broken form is not a near miss: it fires on nothing at all, including the one call it
    was written for, and says nothing while doing so.
    """
    out = {"doc": ("matcher filters a hook group by tool name: a plain name exactly, anything else as "
                   "a regular expression; the separate `if` field filters an individual hook by tool "
                   "name and arguments, in permission-rule syntax (Claude Code hooks reference)"),
           "found": "2026-09-12: a Write(data/*.csv) matcher never fired",
           "settings": {}}
    for label, settings in (("broken", GATE_SETTINGS_BROKEN), ("working", GATE_SETTINGS_WORKING)):
        hook = HookSpec.from_settings(settings, f"require_source_column[{label}]", require_source_column)
        probes = []
        for probe in GATE_PROBES:
            fires = hook_fires(hook, probe["tool"], probe["input"])
            reason = hook.check(probe["tool"], probe["input"]) if fires else ""
            probes.append({"label": probe["label"], "tool": probe["tool"],
                           "file_path": probe["input"].get("file_path", probe["input"].get("command", "")),
                           "hook_fires": fires, "blocked": bool(reason), "reason": reason})
        out["settings"][label] = {
            "settings_json": settings,
            "matcher": hook.matcher,
            "if": hook.if_rule,
            "matcher_is_valid_regex": _compiles(hook.matcher),
            "matcher_matches_any_tool_name": any(matcher_matches(hook.matcher, t["tool"]) for t in GATE_PROBES),
            "fired": sum(p["hook_fires"] for p in probes),
            "blocked": sum(p["blocked"] for p in probes),
            "probes": probes,
        }
    broken, working = out["settings"]["broken"], out["settings"]["working"]
    out["verdict"] = (f"broken settings: the hook fired {broken['fired']} of {len(GATE_PROBES)} times "
                      f"and blocked {broken['blocked']}; working settings: fired {working['fired']} "
                      f"and blocked {working['blocked']}. The broken matcher is a valid regular "
                      f"expression ({broken['matcher_is_valid_regex']}) that matches no tool name, "
                      f"so it raises nothing and is mistaken for a rule that is in force.")
    return out


def _compiles(pattern: str) -> bool:
    try:
        re.compile(pattern)
        return True
    except re.error:
        return False


def working_hook() -> HookSpec:
    """The gate the agent actually runs behind, built from the settings that work."""
    return HookSpec.from_settings(GATE_SETTINGS_WORKING, "require_source_column", require_source_column)


# ======================================================================================
# Block E -- the verifier (E1)
# ======================================================================================
CLAIM_PATTERNS = {
    "change_basis_points": re.compile(r"([-+]?\d+(?:\.\d+)?)\s*basis\s*points", re.I),
    "y10_start": re.compile(r"from\s+([-+]?\d+\.\d+)\s*(?:%|percent)", re.I),
    "y10_end": re.compile(r"to\s+([-+]?\d+\.\d+)\s*(?:%|percent)", re.I),
}


def verify_claims(text: str, meeting_date: str) -> dict:
    """Recompute every number the agent asserted, from the panel, and say which ones hold.

    A claim the text does not make is reported as absent, not as passing: an answer that
    quietly drops the window it was measured over has not been verified, it has been skirted.
    """
    truth = json.loads(yield_change(meeting_date))
    checks = []
    for field_name, pattern in CLAIM_PATTERNS.items():
        m = pattern.search(text)
        expected = truth.get(field_name)
        if m is None:
            checks.append({"claim": field_name, "stated": None, "expected": expected,
                           "verdict": "absent"})
            continue
        stated = float(m.group(1))
        ok = expected is not None and abs(stated - float(expected)) <= 0.05
        checks.append({"claim": field_name, "stated": stated, "expected": expected,
                       "verdict": "matches" if ok else "contradicted"})
    # The window is a claim too: an answer that names no dates cannot be checked against one.
    window_named = all(d in text for d in (truth["window_start"], truth["window_end"]))
    checks.append({"claim": "window_dates_stated", "stated": window_named, "expected": True,
                   "verdict": "matches" if window_named else "absent"})
    cited = bool(re.search(r"https?://\S*federalreserve\.gov\S*", text))
    checks.append({"claim": "statement_cited", "stated": cited, "expected": True,
                   "verdict": "matches" if cited else "absent"})
    verdicts = Counter(c["verdict"] for c in checks)
    return {"meeting_date": meeting_date, "truth": truth, "checks": checks,
            "n_checks": len(checks), "counts": dict(verdicts),
            "passes": verdicts["contradicted"] == 0 and verdicts["absent"] == 0}


# ======================================================================================
# Block A -- the scripted run, and the demonstrations (A2-A5, E1)
# ======================================================================================
SYSTEM = ("You answer one bounded question about a single FOMC meeting: what the Committee said "
          "about inflation, and what the ten-year Treasury yield did over the following month. "
          "Quote the statement and give its URL. State the window your yield numbers cover.")

TASK_MEETING = "2022-03-16"
PROMPT = (f"At the FOMC meeting of {TASK_MEETING}, what did the Committee say about inflation, and "
          "what did the ten-year Treasury yield do over the following month? Cite the statement.")

ALLOWED = ("fomc_search", "fomc_statement", "yield_change", "panel_stats", "Write")

FINAL_ANSWER = (
    "At its meeting of 2022-03-16 the Committee said: \"Inflation remains elevated, reflecting "
    "supply and demand imbalances related to the pandemic, higher energy prices, and broader "
    "price pressures.\" Source: "
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20220316a.htm . "
    "Over the following month the ten-year Treasury yield rose 57.0 basis points, from 2.32 percent "
    "to 2.89 percent. The panel is monthly, so that window runs 2022-03-31 to 2022-04-30 -- it "
    "opens 15 days after the Committee met, and does not cover the fortnight in between."
)


def scripted_run(budget: Budget | None = None, trace: bool = False) -> dict:
    """The replayed run: retrieve, read, measure, try a write the gate stops, write again, answer.

    Seven model turns. Turn 4 calls a tool that is not on the allow-list and turn 5 writes a CSV
    with no source column, so the refusal and the gate are both exercised by the loop itself
    rather than described.
    """
    model = scripted(
        ("I will find the meeting first.",
         [("fomc_search", {"query": "inflation", "start": "2022-03-01", "end": "2022-03-31", "k": 3})]),
        ("Now the statement's own words on inflation.",
         [("fomc_statement", {"date": TASK_MEETING, "topic": "inflation"})]),
        ("And the yield over the following month.",
         [("yield_change", {"meeting_date": TASK_MEETING})]),
        ("Let me check the file on disk directly.",
         [("Bash", {"command": "cat labs/session02_assets/yield_panel.csv"})]),
        ("I will save the answer as a table.",
         [("Write", {"file_path": "data/answer.csv",
                     "content": "meeting_date,change_bp\n2022-03-16,57.0\n"})]),
        ("Adding the source column the gate asked for.",
         [("Write", {"file_path": "data/answer.csv",
                     "content": "meeting_date,window_start,window_end,change_bp,source\n"
                                "2022-03-16,2022-03-31,2022-04-30,57.0,"
                                "labs/session02_assets/yield_panel.csv\n"})]),
        (FINAL_ANSWER, []),
    )
    registry = ToolRegistry(build_tools(), permissions=Permissions(ALLOWED), hooks=[working_hook()])
    WRITTEN.clear()
    run = run_agent(model, registry, SYSTEM, PROMPT, budget=budget or Budget(max_steps=8), trace=trace)
    run["verification"] = verify_claims(run["final_text"], TASK_MEETING)
    run["files_written"] = sorted(WRITTEN)
    return run


def budget_demo(max_steps: int = 3) -> dict:
    """The same run under a budget too small for it: the loop stops, mid-task, with no answer."""
    run = scripted_run(budget=Budget(max_steps=max_steps), trace=False)
    return {"max_steps": max_steps, "turns": run["turns"], "stop": run["stop"],
            "final_text": run["final_text"],
            "tokens_sent_total": run["tokens_sent_total"],
            "note": "a budget is a rule about money that is also a rule about completeness"}


def unverifiable_answer_demo() -> dict:
    """What the verifier does with an answer that is fluent, cited, and wrong about the number."""
    bad = FINAL_ANSWER.replace("57.0 basis points", "12.0 basis points")
    worse = re.sub(r"The panel is monthly.*$", "", bad).strip()
    return {"wrong_number": verify_claims(bad, TASK_MEETING),
            "window_dropped": verify_claims(worse, TASK_MEETING)}


# ======================================================================================
# Block E -- reliability over the recorded scenarios (E3)
# ======================================================================================
def scenario_summaries(root: Path | None = None) -> List[dict]:
    """Every summary.json of the Summer 2026 agent lab's recorded runs (shipped in lec09_summaries/)."""
    root = Path(root or LEC09_RESULTS)
    out = []
    for path in sorted(root.glob("*/*/summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_scenario_dir"] = path.parent.parent.name
        data["_run_dir"] = path.parent.name
        out.append(data)
    return out


def reliability(root: Path | None = None) -> dict:
    """How often repeated runs of the same agent on the same task reach the same answer.

    The recorded summaries hold per-defect counts out of k runs, not the finding set of each
    individual run, so the agreement rate cannot be read off exactly. Two bounds can be:

      lower  k - sum over defects of min(c, k - c)   every disagreement on a separate run
      upper  min over defects of max(c, k - c)       every disagreement on the same runs

    They coincide whenever every defect was found by all k runs or by none, which is what
    `unanimous` records. A run the harness could not complete is counted in `failed` and is a
    reliability fact in its own right.
    """
    cells = []
    for summary in scenario_summaries(root):
        k = summary.get("k", 1)
        for agent, row in summary.get("table", {}).items():
            runs = int(row.get("runs", 0))
            failed = int(row.get("failed", 0))
            defects = {d: int(c) for d, c in (row.get("defects") or {}).items()}
            if runs <= 0:
                cells.append({"scenario": summary.get("scenario"), "run_dir": summary["_run_dir"],
                              "agent": agent, "k": k, "runs": 0, "failed": failed,
                              "defects": defects, "hit_rate": {}, "unanimous": None,
                              "same_answer_lower": 0, "same_answer_upper": 0,
                              "same_answer_lower_rate": 0.0, "same_answer_upper_rate": 0.0})
                continue
            hit_rate = {d: round(c / runs, 4) for d, c in defects.items()}
            disagreements = {d: min(c, runs - c) for d, c in defects.items()}
            lower = max(1, runs - sum(disagreements.values())) if defects else runs
            upper = min((max(c, runs - c) for c in defects.values()), default=runs)
            cells.append({"scenario": summary.get("scenario"), "run_dir": summary["_run_dir"],
                          "agent": agent, "k": k, "runs": runs, "failed": failed,
                          "defects": defects, "hit_rate": hit_rate,
                          "unanimous": all(v == 0 for v in disagreements.values()),
                          "same_answer_lower": lower, "same_answer_upper": upper,
                          "same_answer_lower_rate": round(lower / runs, 4),
                          "same_answer_upper_rate": round(upper / runs, 4)})

    repeated = [c for c in cells if c["runs"] >= 2]
    unanimous = [c for c in repeated if c["unanimous"]]
    attempted = sum(c["runs"] + c["failed"] for c in cells)
    return {
        "source": repo_path(root or LEC09_RESULTS),
        "scenarios": sorted({c["scenario"] for c in cells if c["scenario"]}),
        "n_summaries": len({(c["scenario"], c["run_dir"]) for c in cells}),
        "n_cells": len(cells),
        "n_cells_with_repeats": len(repeated),
        "runs_attempted": attempted,
        "runs_completed": sum(c["runs"] for c in cells),
        "runs_failed": sum(c["failed"] for c in cells),
        "harness_failure_rate": round(sum(c["failed"] for c in cells) / attempted, 4) if attempted else 0.0,
        "unanimous_cells": len(unanimous),
        "unanimous_rate": round(len(unanimous) / len(repeated), 4) if repeated else None,
        "mean_same_answer_lower_rate": round(float(np.mean([c["same_answer_lower_rate"] for c in repeated])), 4) if repeated else None,
        "mean_same_answer_upper_rate": round(float(np.mean([c["same_answer_upper_rate"] for c in repeated])), 4) if repeated else None,
        "cells": cells,
    }


# ======================================================================================
# Block E -- the redaction, and its proof (E3)
# ======================================================================================
# The rules that produced labs/session06_assets/transcripts/. Ordered, most specific first, so
# a scratch path can never fall through to the generic rule. The raw session files stay in
# scratch: they carry the absolute path of the machine, the operator's account name, and the
# harness's session and request identifiers.
def account_name(explicit: str | None = None) -> str:
    """The operator's account name.

    Read from the environment rather than written down, so the rule that removes an account
    name does not itself have to carry one.
    """
    if explicit:
        return explicit
    import getpass
    return getpass.getuser()


def transcript_rules(account: str | None = None) -> List[Tuple[str, str, str]]:
    """(regular expression, replacement, why) -- the textual half of the redaction.

    Two of these were added after the first pass shipped: the harness writes a copy of every
    session under .claude/projects/<slug>, where the slug spells the whole absolute path with
    its slashes turned into dashes, and a shell PATH echoed by `which` names a home directory
    on a shared filesystem that is under neither /u nor /home. Neither is path-shaped, so
    neither is caught by a rule about paths.
    """
    account = account_name(account)
    return [
        (r"/tmp/claude-\d+/[^\s\"'`,;:)\]}]*?/scratchpad/[A-Za-z0-9_.\-]+/run\b", "<RUN_DIR>",
         "the folder the recorded run worked in"),
        (r"/tmp/claude-\d+/[^\s\"'`,;:)\]}]*?/scratchpad", "<SCRATCH>", "the session scratchpad root"),
        (r"/tmp/claude-\d+", "<SCRATCH>", "any remaining session scratch root"),
        (r"\.claude/projects/[A-Za-z0-9._\-]+", ".claude/projects/<PROJECT>",
         "the harness's project slug, which spells a whole absolute path with dashes"),
        (r"(?<![A-Za-z0-9/])/u/[A-Za-z0-9_.\-]+", "<HOME>", "the operator's home directory"),
        (r"(?<![A-Za-z0-9/])/(?:home|Users)/[A-Za-z0-9_.\-]+", "<HOME>",
         "a home directory on any other machine"),
        (r"(?<![A-Za-z0-9/])/projects/[A-Za-z0-9_.\-]+/" + re.escape(account), "<HOME>",
         "a home directory on a shared filesystem, which is under neither /u nor /home"),
        (r"\b" + re.escape(account) + r"\b", "<USER>",
         "the account name anywhere else: a bare word, or a segment of a path on a shared filesystem"),
        (r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", "<UUID>",
         "session, prompt and message identifiers"),
        (r"\b(?:msg|toolu|req)_[A-Za-z0-9]{8,}\b", "<ID>",
         "API message, tool-use and request identifiers"),
        (r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", "<EMAIL>", "a user identifier"),
    ]


# Structural half: fields that carry machine or account identity are never copied into the
# digest in the first place, so no substitution has to catch them.
DROPPED_FIELDS = ("uuid", "parentUuid", "sessionId", "session_id", "promptId", "requestId",
                  "request_id", "leafUuid", "sourceToolAssistantUUID", "id", "signature",
                  "cwd", "gitBranch", "userType", "entrypoint", "container",
                  "messaging_socket_path", "memory_paths", "apiKeySource", "tool_use_id",
                  "parent_tool_use_id")


def displayable_rules(account: str | None = None) -> List[dict]:
    """The rules as they may be written down: the one that spells the account name does not.

    A word-boundary pattern around a name is not itself matched by that same pattern, so the
    masking is done by substitution on the escaped literal rather than by re-running the rule.
    """
    masked = re.escape(account_name(account))
    return [{"pattern": pattern.replace(masked, "<USER>"), "replacement": replacement, "why": why}
            for pattern, replacement, why in transcript_rules(account)]


def redact_text(text: str, account: str | None = None) -> str:
    """Apply transcript_rules() to one string, in order."""
    for pattern, replacement, _ in transcript_rules(account):
        text = re.sub(pattern, replacement, text)
    return text


def redact(obj, account: str | None = None):
    """Apply both halves of the redaction to a parsed JSON value."""
    if isinstance(obj, str):
        return redact_text(obj, account)
    if isinstance(obj, list):
        return [redact(v, account) for v in obj]
    if isinstance(obj, dict):
        return {k: redact(v, account) for k, v in obj.items() if k not in DROPPED_FIELDS}
    return obj


_PRINTABLE = set(map(chr, range(32, 127))) | set("\n\r\t")


def looks_binary(text: str) -> bool:
    """A tool result that is bytes rather than prose: a downloaded file echoed to the terminal.

    Two signs, either sufficient: more than one character in twenty outside printable ASCII,
    or an unbroken run of more than two thousand characters with no whitespace in it. Such a
    result is elided rather than carried, because twenty thousand characters of base64 teach
    nothing and every coincidental substring inside them has to be explained away.
    """
    if not text:
        return False
    sample = text[:20_000]
    odd = sum(1 for ch in sample if ch not in _PRINTABLE) / len(sample)
    longest = max((len(run) for run in re.split(r"\s+", sample)), default=0)
    return odd > 0.05 or longest > 2_000


def forbidden_patterns(account: str | None = None) -> Dict[str, str]:
    """What must not survive anywhere in a shipped transcript."""
    return {
        "home directory": r"(?<![A-Za-z0-9/])/u/[A-Za-z0-9_.\-]+",
        "session scratch root": r"/tmp/claude-\d+",
        "harness project slug": r"-tmp-claude-\d+",
        "other home directory": r"(?<![A-Za-z0-9/])/(?:home|Users)/[A-Za-z0-9_.\-]+",
        "shared-filesystem home": r"(?<![A-Za-z0-9/])/projects/[A-Za-z0-9_.\-]+/(?:"
                                  + re.escape(account_name(account)) + r"|<USER>)",
        "account name": r"\b" + re.escape(account_name(account)) + r"\b",
        "e-mail address": r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
        "UUID": r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        "API identifier": r"\b(?:msg|toolu|req)_[A-Za-z0-9]{8,}\b",
        "session field": r"\"session_?[Ii]d\"",
    }


def check_redaction(folder: Path | None = None, account: str | None = None) -> dict:
    """Grep the shipped transcripts for anything private. Every count must be zero.

    The account-name check uses the account running it, so re-running this as the operator who
    produced the files is the test that matters; the structural patterns hold for anyone.
    """
    folder = Path(folder or TRANSCRIPTS)
    FORBIDDEN = forbidden_patterns(account)
    files = sorted(folder.glob("*.jsonl"))
    per_file = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        hits = {}
        for label, pattern in FORBIDDEN.items():
            found = re.findall(pattern, text)
            hits[label] = {"count": len(found), "examples": sorted(set(found))[:3]}
        per_file.append({"file": path.name, "bytes": len(text.encode()),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "hits": hits, "clean": all(h["count"] == 0 for h in hits.values())})
    return {"folder": repo_path(folder), "n_files": len(files),
            "totals": {label: sum(f["hits"][label]["count"] for f in per_file) for label in FORBIDDEN},
            "clean": all(f["clean"] for f in per_file),
            "rules": displayable_rules(account),
            "files": per_file}


# ======================================================================================
# Helpers, and the cost curve (A4)
# ======================================================================================
def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repo_path(path) -> str:
    """A path written from the repository root, so no record carries a machine's own layout."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return redact_text(str(path))


def cost_curve(n_turns: int = 12, new_tokens_per_turn: int = 900) -> dict:
    """The square-law ledger, alongside what the replayed run actually cost."""
    run = scripted_run(trace=False)
    return {"model": resend_cost(n_turns, new_tokens_per_turn),
            "new_tokens_per_turn": new_tokens_per_turn,
            "actual": [t["cumulative_sent"] for t in run["ledger"]]}


def plot_cost_curve(path: Path | None = None):
    """Cumulative tokens sent against turn, for the replayed run. matplotlib is imported here."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = scripted_run(trace=False)
    turns = [t["turn"] for t in run["ledger"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(turns, [t["cumulative_sent"] for t in run["ledger"]], marker="o", label="sent, cumulative")
    ax.plot(turns, [t["tokens_sent"] for t in run["ledger"]], marker="s", label="sent this turn")
    ax.set_xlabel("turn")
    ax.set_ylabel("tokens")
    ax.legend(frameon=False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig
