"""LLM backends for the OLG RAG demo.

Adds a Claude Code subprocess backend so the notebook works against the user's
Claude Code subscription (no Anthropic API key required), plus a side-by-side
`compare()` helper that returns naive-vs-RAG answers for the notebook.

Mirrors the design used in demos/M4_T1_rag_ren_demo/rag_ren.py, adapted to this
package's existing types (`Chunk`, `SearchResult`, `build_prompt`).
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Optional

from .chunking import Chunk
from .generation import build_prompt
from .retrieval import SearchResult, TfidfIndex, search


# --------------------------------------------------------------------------- #
# System prompts (PhD economics framing)
# --------------------------------------------------------------------------- #


NAIVE_SYS_PROMPT = (
    "You are a PhD-level macroeconomics tutor. Answer the question concisely "
    "(4-7 sentences) using your training knowledge. Use precise terminology "
    "(Pareto optimality, dynamic efficiency, r vs n, golden rule, etc.). "
    "Do not cite specific papers — you do not have a corpus to cite from. "
    "Aim for the textbook intermediate-graduate-macro answer."
)

RAG_SYS_PROMPT = (
    "You are a PhD-level macroeconomics tutor answering a question grounded "
    "ONLY in retrieved excerpts from Spear and Young's OLG history paper. "
    "Cite every substantive claim with [S1], [S2], ... source labels. "
    "If the retrieved context is insufficient, say so explicitly — do not "
    "fall back to outside knowledge. 4-7 sentences. Prefer the paper's own "
    "phrasing where it is distinctive."
)


# --------------------------------------------------------------------------- #
# Claude Code subprocess backend
# --------------------------------------------------------------------------- #


def claude_code_available() -> bool:
    """Return True if the `claude` CLI is on PATH."""
    return shutil.which("claude") is not None


def claude_code_auth_check(timeout: int = 30) -> tuple[bool, str]:
    """Probe `claude -p` with a tiny prompt. Returns (ok, message)."""
    if not claude_code_available():
        return False, (
            "`claude` CLI not found on PATH. Install Claude Code from\n"
            "  https://docs.anthropic.com/claude-code\n"
            "then run `claude` in a terminal and execute /login to authenticate."
        )
    try:
        out = claude_code_answer(
            "Reply with only the word: ok",
            model="haiku",
            system_prompt=None,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "claude command timed out — your session may need refreshing. Run `claude` then /login."
    except Exception as exc:
        return False, (
            f"claude CLI is installed but the call failed: {exc}\n"
            "In a terminal run `claude`, execute /login, complete OAuth, then re-run this cell."
        )
    return True, f"Claude Code authenticated (probe response: {out[:60]!r})"


def claude_code_answer(
    prompt: str,
    *,
    model: str = "sonnet",
    system_prompt: Optional[str] = None,
    timeout: int = 180,
) -> str:
    """Call local `claude -p` headless. Uses Claude Code subscription OAuth.

    --tools ""              : disable every built-in tool (pure text Q&A)
    --no-session-persistence: don't save the session to disk
    --system-prompt <text>  : replaces the default system prompt so user's
                              CLAUDE.md / memory don't leak into the answer
    """
    cmd = [
        "claude", "-p",
        "--model", model,
        "--tools", "",
        "--no-session-persistence",
        "--output-format", "text",
    ]
    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])
    cmd.append(prompt)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip() or "(no output)"
        raise RuntimeError(f"claude CLI failed (rc={proc.returncode}): {msg}")
    return proc.stdout.strip()


# --------------------------------------------------------------------------- #
# Naive (no-retrieval) and RAG (retrieval-grounded) answer wrappers
# --------------------------------------------------------------------------- #


def naive_answer(question: str, *, model: Optional[str] = None) -> str:
    """Baseline: ask Claude with no retrieval context."""
    return claude_code_answer(
        question, model=model or "sonnet", system_prompt=NAIVE_SYS_PROMPT
    )


def rag_answer(
    question: str, retrieved: list[SearchResult], *, model: Optional[str] = None
) -> str:
    """Build the RAG prompt from retrieved chunks and call Claude."""
    prompt = build_prompt(question, retrieved)
    return claude_code_answer(
        prompt, model=model or "sonnet", system_prompt=RAG_SYS_PROMPT
    )


# --------------------------------------------------------------------------- #
# Side-by-side comparison helper for the notebook
# --------------------------------------------------------------------------- #


def compare(
    question: str,
    chunks: list[Chunk],
    index: TfidfIndex,
    *,
    top_k: int = 5,
    model: Optional[str] = None,
    skip_naive: bool = False,
) -> dict:
    """Run retrieval + naive + RAG and return everything for the notebook.

    Returns:
        {
          "question": str,
          "retrieved": list[SearchResult],
          "naive":     str,    # Claude with no retrieval
          "rag":       str,    # Claude grounded in retrieved chunks
          "prompt":    str,    # the RAG prompt actually sent
        }
    """
    retrieved = search(chunks, index, question, top_k=top_k)
    prompt = build_prompt(question, retrieved)
    naive = "(skipped)" if skip_naive else naive_answer(question, model=model)
    rag = rag_answer(question, retrieved, model=model)
    return {
        "question": question,
        "retrieved": retrieved,
        "naive": naive,
        "rag": rag,
        "prompt": prompt,
    }


def hits_anchor(retrieved: list[SearchResult], anchors: list[str], top_n: int = 5) -> list[str]:
    """Return list of anchor terms found inside any of the top-N retrieved chunks.

    Used for retrieval sanity-checks (does the retriever surface the section
    where the paper's distinctive argument actually lives?). Case-insensitive
    substring match against chunk text.
    """
    if not anchors:
        return []
    body = "\n".join(r.chunk.text for r in retrieved[:top_n]).lower()
    return [a for a in anchors if a.lower() in body]
