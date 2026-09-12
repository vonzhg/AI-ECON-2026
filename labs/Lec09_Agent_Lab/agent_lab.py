"""Helper module for the Lecture 9 agent lab.

Everything needed to watch an agentic loop run WITHOUT an API key, a network
connection, or a paid service.  The point of the lab is that the loop is small
and inspectable; the model is the only part we stub out.

Two tiers:

  Tier 1 (always available)  ScriptedModel replays a fixed list of responses,
                             so run_agent() exercises the real control flow
                             deterministically.  No credentials, no cost.

  Tier 2 (opt-in)            claude_code_*() shell out to the local `claude`
                             CLI, mirroring the backend used in Lecture 8's
                             RAG demo.  No Anthropic API key is needed -- it
                             uses the student's own Claude Code login.

Nothing here imports `anthropic`, `openai`, `crewai` or `langgraph`.  The lab
runs on the course's standard CPU environment.
"""

from __future__ import annotations

import json
import shutil
import subprocess
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
    """The agentic loop from Topic 9.2b, section 1.

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
# Tier 2: the local Claude Code CLI (no API key required)
# --------------------------------------------------------------------------- #


def claude_code_available() -> bool:
    """True if the `claude` CLI is on PATH."""
    return shutil.which("claude") is not None


def claude_code_auth_check(timeout: int = 30) -> tuple[bool, str]:
    """Probe `claude -p` with a tiny prompt. Returns (ok, message)."""
    if not claude_code_available():
        return False, ("`claude` CLI not found on PATH. Install Claude Code, or "
                       "just skip this step -- every other cell runs offline.")
    try:
        proc = subprocess.run(
            ["claude", "-p", "--output-format", "text", "Reply with the word OK."],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"`claude -p` timed out after {timeout}s."
    if proc.returncode != 0:
        return False, (f"`claude -p` exited {proc.returncode}. Run `claude` in a "
                       f"terminal and complete /login first.\n{proc.stderr[:300]}")
    return True, proc.stdout.strip()[:200]


def claude_code_ask(prompt: str, timeout: int = 120) -> str:
    """One-shot `claude -p` call. Requires an interactive /login beforehand."""
    proc = subprocess.run(
        ["claude", "-p", "--output-format", "text", prompt],
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude -p failed ({proc.returncode}): {proc.stderr[:300]}")
    return proc.stdout.strip()
