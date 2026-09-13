"""Helper module for the Lecture 9 agent lab.

Everything needed to watch an agentic loop run WITHOUT an API key, a network
connection, or a paid service.  The point of the lab is that the loop is small
and inspectable; the model is the only part we stub out.

Two tiers:

  Tier 1 (always available)  ScriptedModel replays a fixed list of responses,
                             so run_agent() exercises the real control flow
                             deterministically.  No credentials, no cost.

  Tier 2 (opt-in)            harness_*() shell out to a local terminal agent
                             -- Claude Code, Gemini CLI, or Codex CLI -- in
                             headless mode, using the student's own login.
                             No API key appears in this code.  The older
                             claude_code_*() names remain as thin wrappers.

Nothing here imports `anthropic`, `openai`, `crewai` or `langgraph`.  The lab
runs on the course's standard CPU environment.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable

# --------------------------------------------------------------------------- #
# Token accounting
# --------------------------------------------------------------------------- #

# A deliberately crude approximation.  Real tokenizers are model-specific and
# the Anthropic API exposes messages.count_tokens() for exact numbers; ~4
# characters per token is close enough to make the SHAPE of the cost curve
# visible, which is all this lab claims to show.
CHARS_PER_TOKEN = 4


def count_tokens(obj: Any) -> int:
    """Approximate token count of any JSON-serialisable message structure."""
    if isinstance(obj, str):
        text = obj
    else:
        text = json.dumps(obj, ensure_ascii=False, default=str)
    return max(1, len(text) // CHARS_PER_TOKEN)


def conversation_tokens(messages: list[dict]) -> int:
    """Tokens in the WHOLE list -- what you re-send on the next turn."""
    return count_tokens(messages)


# --------------------------------------------------------------------------- #
# Tools: real functions with real schemas
# --------------------------------------------------------------------------- #


@dataclass
class Tool:
    """A tool is code plus a schema. The model picks arguments; the code runs."""

    name: str
    description: str
    input_schema: dict
    fn: Callable[..., str]

    def to_api_dict(self) -> dict:
        """Exactly the shape the Messages API expects in `tools=[...]`."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {t.name: t for t in (tools or [])}

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return sorted(self._tools)

    def to_api_list(self) -> list[dict]:
        return [self._tools[n].to_api_dict() for n in sorted(self._tools)]

    def run(self, name: str, args: dict) -> tuple[str, bool]:
        """Execute a tool. Returns (result_text, is_error).

        A failing tool must still return a tool_result -- dropping it would
        leave a tool_use block unanswered and the API would reject the next
        request.
        """
        if name not in self._tools:
            return (f"Error: no tool named {name!r}.", True)
        try:
            return (str(self._tools[name].fn(**args)), False)
        except Exception as exc:                      # noqa: BLE001 - taught deliberately
            return (f"Error: {type(exc).__name__}: {exc}", True)


# --------------------------------------------------------------------------- #
# The stub model
# --------------------------------------------------------------------------- #


@dataclass
class Response:
    """Mirrors the fields of a real Messages API response that the loop reads."""

    content: list[dict]
    stop_reason: str


@dataclass
class ScriptedModel:
    """Replays a fixed script of responses, ignoring the prompt.

    This is NOT a language model.  It exists so the surrounding loop -- which
    IS the real thing -- can be run and inspected deterministically.
    """

    script: list[Response]
    name: str = "scripted"
    calls: int = 0
    seen_message_counts: list[int] = field(default_factory=list)

    def create(self, *, system: str, tools: list[dict], messages: list[dict]) -> Response:
        # Record how much the model was shown on this turn: the whole list,
        # every time.  This is the fact Step 1 of the notebook demonstrates.
        self.seen_message_counts.append(conversation_tokens(messages))
        if self.calls >= len(self.script):
            self.calls += 1
            return Response(content=[{"type": "text", "text": "(script exhausted)"}],
                            stop_reason="end_turn")
        resp = self.script[self.calls]
        self.calls += 1
        return resp


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def tool_use_block(block_id: str, name: str, args: dict) -> dict:
    return {"type": "tool_use", "id": block_id, "name": name, "input": args}


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #


def run_agent(model, registry: ToolRegistry, system: str, user_prompt: str,
              max_turns: int = 12, trace: bool = True) -> dict:
    """The agentic loop from T2a ("The Loop in Twenty Lines").

    Identical in structure to the Anthropic Messages API version shown on the
    slide; only `model.create` is stubbed.  Returns a trace dict so the
    notebook can plot what happened.
    """
    messages: list[dict] = [{"role": "user", "content": user_prompt}]
    history = []          # (turn, tokens_sent, n_tool_calls)
    final_text = ""

    for turn in range(1, max_turns + 1):
        tokens_sent = conversation_tokens(messages)
        resp = model.create(system=system, tools=registry.to_api_list(),
                            messages=messages)

        tool_uses = [b for b in resp.content if b.get("type") == "tool_use"]
        history.append({"turn": turn, "tokens_sent": tokens_sent,
                        "tool_calls": len(tool_uses)})

        if trace:
            said = " ".join(b["text"] for b in resp.content if b.get("type") == "text")
            print(f"  turn {turn}: sent {tokens_sent:>5} tok | "
                  f"stop={resp.stop_reason:<9} | {len(tool_uses)} tool call(s)"
                  + (f' | "{said[:48]}"' if said else ""))

        if resp.stop_reason != "tool_use":
            final_text = " ".join(b["text"] for b in resp.content
                                  if b.get("type") == "text")
            break

        # The assistant turn -- including its tool_use blocks -- goes into the list.
        messages.append({"role": "assistant", "content": resp.content})

        # Execute every tool call, and return ALL results in ONE user message.
        results = []
        for b in tool_uses:
            out, is_err = registry.run(b["name"], b["input"])
            if trace:
                flag = " [error]" if is_err else ""
                print(f"       -> {b['name']}({b['input']}) = {out[:44]}{flag}")
            results.append({"type": "tool_result", "tool_use_id": b["id"],
                            "content": out, "is_error": is_err})
        messages.append({"role": "user", "content": results})

    return {"messages": messages, "history": history, "final_text": final_text,
            "turns": len(history)}


# --------------------------------------------------------------------------- #
# Capacity: how many agents actually fit
# --------------------------------------------------------------------------- #


def agents_that_fit(tokens_per_minute_limit: int, avg_turn_tokens: int,
                    turns_per_agent: int, minutes: float = 1.0) -> int:
    """How many agents can run concurrently under a tokens/min rate limit.

    Deliberately simple: the point is that the quota is denominated in TOKENS,
    not in agents, so a verbose agent costs the same as several terse ones.
    """
    per_agent = avg_turn_tokens * turns_per_agent
    if per_agent <= 0:
        raise ValueError("per-agent token cost must be positive")
    return int((tokens_per_minute_limit * minutes) // per_agent)


def resend_cost(n_turns: int, new_tokens_per_turn: int) -> list[int]:
    """Cumulative tokens SENT when the whole list is re-sent on every turn.

    Turn k re-sends everything accumulated in turns 1..k, so the running total
    grows with the SQUARE of the turn count even though each turn adds only a
    fixed amount of new text.  That is why long sessions get expensive.
    """
    cumulative, total = [], 0
    for k in range(1, n_turns + 1):
        total += new_tokens_per_turn * k      # turn k sends a list of size k
        cumulative.append(total)
    return cumulative


# --------------------------------------------------------------------------- #
# Tier 2: local terminal agents in headless mode (no API key in this code)
# --------------------------------------------------------------------------- #
#
# One entry per harness.  The flags are dated facts: they were checked against
# the installed CLIs and are recorded, with how each was established, in
# source/Lec09_Agentic_AI/CLI_FACTS_2026-09.md.  Change the two together.


@dataclass(frozen=True)
class Harness:
    name: str
    binary: str
    login_hint: str
    status: str  # how the headless recipe below was established (CLI_FACTS)


HARNESSES: dict[str, Harness] = {
    "claude": Harness("claude", "claude", "run `claude` once in a terminal and complete /login",
                      "RUN on Claude Code 2.1.269"),
    "gemini": Harness("gemini", "gemini", "run `gemini` once and sign in (or set GEMINI_API_KEY)",
                      "HELP on Gemini CLI 0.47.0 -- headless call not yet run"),
    "codex": Harness("codex", "codex", "run `codex login`",
                     "HELP on Codex CLI 0.154.0 -- headless call not yet run"),
}


def harness_available(name: str) -> bool:
    """True if the harness's CLI is on PATH."""
    return shutil.which(HARNESSES[name].binary) is not None


def available_harnesses() -> list[str]:
    return [name for name in HARNESSES if harness_available(name)]


def _run_cli(cmd: list[str], cwd: str | None, timeout: int) -> subprocess.CompletedProcess:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              cwd=cwd, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"`{cmd[0]}` timed out after {timeout}s") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"`{cmd[0]}` exited {proc.returncode}: "
                           f"{(proc.stderr or proc.stdout)[-400:]}")
    return proc


def harness_ask(name: str, prompt: str, *, cwd: str | None = None,
                timeout: int = 300, read_only: bool = True, model: str | None = None) -> str:
    """One headless call -- rung 3 -- returning the agent's final text.

    read_only=True limits the agent to reading files: Read/Grep/Glob in Claude
    Code, plan approval mode in Gemini CLI, the read-only sandbox in Codex CLI.
    Every call is a fresh session, which is what makes it an honest test.
    `model` is passed only to Claude Code, whose agent files name a model
    alias (sonnet, opus, haiku); the other harnesses use their own default.
    """
    if name == "claude":
        cmd = ["claude", "-p", prompt, "--output-format", "json"]
        if model:
            cmd += ["--model", model]
        if read_only:
            cmd += ["--allowedTools", "Read", "Grep", "Glob", "--permission-mode", "dontAsk"]
        return json.loads(_run_cli(cmd, cwd, timeout).stdout).get("result", "")
    if name == "gemini":
        cmd = ["gemini", "-p", prompt, "-o", "json"]
        if read_only:
            cmd += ["--approval-mode", "plan"]
        return json.loads(_run_cli(cmd, cwd, timeout).stdout).get("response", "")
    if name == "codex":
        with tempfile.TemporaryDirectory() as tmp:
            last = os.path.join(tmp, "last_message.txt")
            cmd = ["codex", "exec", prompt, "--skip-git-repo-check", "-o", last,
                   "-s", "read-only" if read_only else "workspace-write"]
            _run_cli(cmd, cwd, timeout)
            with open(last, encoding="utf-8") as fh:
                return fh.read()
    raise ValueError(f"unknown harness {name!r}; choose from {sorted(HARNESSES)}")


def harness_auth_check(name: str, timeout: int = 60) -> tuple[bool, str]:
    """Probe a harness with a tiny headless prompt.  Returns (ok, message)."""
    if not harness_available(name):
        return False, f"`{HARNESSES[name].binary}` not found on PATH."
    try:
        reply = harness_ask(name, "Reply with the word OK.", timeout=timeout)
    except RuntimeError as exc:
        return False, f"{exc}\nFix: {HARNESSES[name].login_hint}."
    return True, reply.strip()[:200]


# Names used by earlier versions of the notebook.
def claude_code_available() -> bool:
    return harness_available("claude")


def claude_code_auth_check(timeout: int = 30) -> tuple[bool, str]:
    return harness_auth_check("claude", timeout=timeout)


def claude_code_ask(prompt: str, timeout: int = 120) -> str:
    return harness_ask("claude", prompt, timeout=timeout, read_only=False)
