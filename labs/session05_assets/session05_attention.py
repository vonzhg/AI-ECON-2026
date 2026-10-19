"""Session 5 lab -- attention by hand, and Machine B opened up.

Part A  Scaled dot-product attention in numpy, one step at a time. Every intermediate comes back
        -- queries, keys, values, raw scores, scaled scores, masked scores, weights, the blend --
        so the notebook can print the middle of the formula, not only its result.
Part B  The book's nine-token example, reproduced as arithmetic. The chapter stipulates the nine
        scores for exposition; it does not compute them, and says so. What this module reproduces
        is therefore the arithmetic from those stated scores, to four decimals, beside the figures
        the chapter prints.
Part C  Causal masking, multi-head attention, and attention entropy -- the diagnostic that
        separates a head doing keyword search (low entropy) from a head averaging the context
        (high entropy near ln T).
Part D  Machine B. Session 1 trained a character-level MiniGPT on the FOMC corpus and kept the
        checkpoint; here the same five files are copied in under their recorded hashes, the model
        classes are reproduced verbatim so the state dictionary loads key for key, and a path is
        added that hands back the attention weights the original discarded. Then the model is
        made to do four things a slide can show: load, attend, generate at three temperatures,
        and train from scratch for a few dozen steps so the loss curve is watched rather than
        described.

Descended from Lab 7B of the 2025 Lecture 7 lab (`labs/Lec07_LLM_Lab/Lab7B_Attention_MiniGPT.ipynb`)
and from `labs/session01_assets/session01_demo.py`, rewritten as a library: no module-level
computation, hashes asserted before anything is loaded, and the architecture checked against the
checkpoint instead of assumed.
Imports: numpy, torch and the standard library. No network. CPU only.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import time

if __name__ == "__main__":      # instructor runs only: keep the linear-algebra library polite
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "4")
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
FOMC_DIR = HERE / "fomc"
MANIFEST = HERE / "fomc_manifest.json"
MACHINE_B = HERE / "minigpt"
MACHINE_B_MANIFEST = MACHINE_B / "MANIFEST.json"
RECORDED = HERE / "recorded_attention.json"

# Where the checkpoint came from, and the course-wide freeze manifest that fixes its hashes.
SESSION01 = HERE.parent / "session01_assets"
FREEZE_MANIFEST = HERE.parent.parent / "notes" / "Session05_06_Freeze_Manifest_2026-09-20.json"
FREEZE_GROUP = "session01_minigpt"
MACHINE_B_FILES = ("minigpt_fomc.pt", "minigpt_vocab.json", "minigpt_config.json",
                   "training_report.json", "session01_demo.py")

SEED = 42
EXPECTED_PARAMETERS = 631_424      # Session 1's MiniGPT at d=128, H=4, L=3, n=128, M=78


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------------------
# Part A -- scaled dot-product attention, every intermediate returned
# --------------------------------------------------------------------------------------
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Subtract the row maximum first: exp(700) overflows, exp(0) does not, and the answer is the same."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)


@dataclass
class AttentionTrace:
    """Every quantity in the attention formula, in the order it is computed."""
    queries: np.ndarray
    keys: np.ndarray
    values: np.ndarray
    scores_raw: np.ndarray          # Q K^T, before scaling
    head_dim: int
    scale: float                    # 1 / sqrt(head_dim)
    scores_scaled: np.ndarray       # Q K^T / sqrt(head_dim)
    scores_masked: np.ndarray       # after the causal mask, if one was applied
    weights: np.ndarray             # softmax of the masked scores; rows sum to one
    blend: np.ndarray               # weights @ values
    output: np.ndarray              # blend projected back to model width, if W_O was given
    masked: bool

    def row(self, i: int) -> dict:
        """One query position, laid out for printing."""
        return dict(scores_raw=self.scores_raw[i], scores_scaled=self.scores_scaled[i],
                    scores_masked=self.scores_masked[i], weights=self.weights[i],
                    entropy=attention_entropy(self.weights[i]))


def scaled_dot_product_attention(queries: np.ndarray, keys: np.ndarray, values: np.ndarray,
                                 causal: bool = False,
                                 projection: np.ndarray | None = None) -> AttentionTrace:
    """Attention from Q, K, V. The four steps, with the middle of the formula kept.

    1  scores       s_ij = <q_i, k_j>
    2  scaling      s_ij / sqrt(d_h); without it the scores grow with the head width and the
                    softmax saturates into a one-hot, which kills the gradient
    3  softmax      row-wise, so every query spends exactly one unit of attention
    4  weighted sum of the values

    `causal` blocks every position from reading its own future. `projection` is W_O; leave it out
    and `output` is the blend itself.
    """
    head_dim = queries.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)
    scores_raw = queries @ keys.T
    scores_scaled = scores_raw * scale
    scores_masked = apply_causal_mask(scores_scaled) if causal else scores_scaled
    weights = softmax(scores_masked, axis=-1)
    blend = weights @ values
    output = blend @ projection.T if projection is not None else blend
    return AttentionTrace(queries=queries, keys=keys, values=values, scores_raw=scores_raw,
                          head_dim=head_dim, scale=scale, scores_scaled=scores_scaled,
                          scores_masked=scores_masked, weights=weights, blend=blend,
                          output=output, masked=causal)


def attention_from_embeddings(X: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray,
                              W_O: np.ndarray | None = None, causal: bool = False) -> AttentionTrace:
    """The five steps of the lecture: project into Q, K, V; score; soften; blend; project back."""
    return scaled_dot_product_attention(X @ W_Q.T, X @ W_K.T, X @ W_V.T,
                                        causal=causal, projection=W_O)


def random_projections(d_model: int, head_dim: int, seed: int = SEED
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Untrained W_Q, W_K, W_V, W_O at the usual 1/sqrt(fan-in) scale.

    Attention patterns are learned, not built in: with these the weights come out near-uniform.
    That flat picture is the control the trained maps in Part D are read against.
    """
    rng = np.random.default_rng(seed)
    W_Q, W_K, W_V = (rng.normal(size=(head_dim, d_model)) / np.sqrt(d_model) for _ in range(3))
    W_O = rng.normal(size=(d_model, head_dim)) / np.sqrt(head_dim)
    return W_Q, W_K, W_V, W_O


# --------------------------------------------------------------------------------------
# Part B -- the book's nine-token example
# --------------------------------------------------------------------------------------
BOOK_SENTENCE = "The central bank raised interest rates amid rising inflation"
BOOK_TOKENS: tuple[str, ...] = ("The", "central", "bank", "raised", "interest",
                                "rates", "amid", "rising", "inflation")
BOOK_QUERY_POSITION = 5           # one-based, as the chapter numbers them: "interest"

# The chapter stipulates these nine scores to illustrate the formula. They are not the output of
# any model -- the text says so in the example box -- so what is reproduced here is the arithmetic.
BOOK_SCORES: dict[str, float] = {
    "The": -0.2, "central": 0.0, "bank": 0.5, "raised": 1.1, "interest": 0.0,
    "rates": 2.3, "amid": 0.0, "rising": 0.0, "inflation": 1.9,
}
BOOK_PRINTED: dict[str, str] = {
    "sum_of_exponentials": "26.1", "rates": "0.38", "inflation": "0.26", "raised": "0.11",
    "bank": "0.06", "The": "0.03", "each_zero_score_position": "0.04",
    "remaining_five_total": "0.18",
}
BOOK_STIPULATION = (
    "The chapter's example box states that these values are illustrative and set for exposition, "
    "not the output of any model. The lab therefore reproduces the arithmetic from the stated "
    "scores; it does not claim to recover the scores themselves from a trained network."
)


def book_attention(scores: dict[str, float] = BOOK_SCORES) -> dict:
    """Softmax over the chapter's nine stipulated scores, to four decimals, beside its printed figures.

    Two numbers a reader can check by hand: the exponentials sum to about 26.1, and the four
    positions scored zero each carry exp(0)/26.1 -- roughly 0.04 of the attention, which is where
    the last fifth of the mass goes.
    """
    tokens = list(scores)
    s = np.array([scores[t] for t in tokens], dtype=float)
    exps = np.exp(s)                       # no max-shift: the chapter's arithmetic, literally
    total = float(exps.sum())
    weights = exps / total
    zero_positions = [t for t in tokens if scores[t] == 0.0]
    zero_weight = float(weights[tokens.index(zero_positions[0])])
    remaining = [t for t in tokens if t not in ("rates", "inflation", "raised", "bank")]
    computed = {
        "sum_of_exponentials": total,
        "rates": float(weights[tokens.index("rates")]),
        "inflation": float(weights[tokens.index("inflation")]),
        "raised": float(weights[tokens.index("raised")]),
        "bank": float(weights[tokens.index("bank")]),
        "The": float(weights[tokens.index("The")]),
        "each_zero_score_position": zero_weight,
        "remaining_five_total": float(sum(weights[tokens.index(t)] for t in remaining)),
    }
    comparison = []
    for key, printed in BOOK_PRINTED.items():
        value = computed[key]
        decimals = len(printed.split(".")[1]) if "." in printed else 0
        comparison.append(dict(quantity=key, computed=round(value, 4), printed=printed,
                               computed_to_printed_precision=f"{value:.{decimals}f}",
                               agrees=bool(f"{value:.{decimals}f}" == printed)))
    return dict(
        sentence=BOOK_SENTENCE, tokens=tokens, query_position=BOOK_QUERY_POSITION,
        query_token=BOOK_TOKENS[BOOK_QUERY_POSITION - 1],
        scores={t: scores[t] for t in tokens},
        exponentials={t: round(float(e), 4) for t, e in zip(tokens, exps)},
        sum_of_exponentials=round(total, 4),
        weights={t: round(float(w), 4) for t, w in zip(tokens, weights)},
        weights_sum=round(float(weights.sum()), 10),
        entropy=round(attention_entropy(weights), 4),
        zero_score_positions=zero_positions,
        comparison=comparison,
        all_agree=bool(all(row["agrees"] for row in comparison)),
        stipulation=BOOK_STIPULATION,
        rounding_note=(
            "Adding the chapter's already-rounded display figures gives 0.03 + 4 x 0.04 = 0.19, "
            "while the printed remainder is 0.18. Both are right: the unrounded weights are "
            "0.0313 and four times 0.0383, which sum to 0.1844 and round to 0.18. The gap is an "
            "artefact of adding rounded numbers, not an arithmetic error."),
    )


def lab7b_five_score_variant() -> dict:
    """The 2025 lab softened only the five named scores, and printed 0.45 / 0.30 / 0.14 / 0.07 / 0.04.

    Same formula, a different set: dropping the four zero-scored positions removes 4 / 26.1 of the
    denominator and lifts every surviving weight. Recorded because a slide that shows both has to
    say which is which.
    """
    named = {"rates": 2.3, "inflation": 1.9, "raised": 1.1, "bank": 0.5, "The": -0.2}
    w = softmax(np.array(list(named.values())))
    return dict(note="softmax over the five named scores only, as Lab 7B cell 3 did",
                tokens=list(named), scores=named,
                weights={t: round(float(v), 4) for t, v in zip(named, w)},
                printed={"rates": "0.45", "inflation": "0.30", "raised": "0.14", "bank": "0.07",
                         "The": "0.04"},
                difference_from_the_nine_token_softmax=(
                    "the nine-token denominator is 26.13, the five-token one is 22.13; every "
                    "weight is larger by the ratio 26.13 / 22.13 = 1.18"))


# --------------------------------------------------------------------------------------
# Part C -- masking, heads, entropy
# --------------------------------------------------------------------------------------
def causal_mask(n: int) -> np.ndarray:
    """True where a query may look: the lower triangle, diagonal included."""
    return np.tril(np.ones((n, n), dtype=bool))


def apply_causal_mask(scores: np.ndarray) -> np.ndarray:
    """Minus infinity above the diagonal, so the softmax puts exactly zero there."""
    n = scores.shape[-1]
    return np.where(causal_mask(n), scores, -np.inf)


def attention_entropy(row: np.ndarray) -> float:
    """Shannon entropy of one attention row, in nats.

    Zero means the query reads one position: keyword search. ln T means it reads every position
    equally: a global average. Everything a head can do sits between the two, and the number is
    comparable across heads only when T is the same.
    """
    p = np.asarray(row, dtype=float)
    p = p[p > 0.0]
    return float(-(p * np.log(p)).sum())


def entropy_profile(weights: np.ndarray, from_position: int = 1) -> dict:
    """Mean and spread of the row entropies of one attention map, with ln T for scale.

    Row 0 of a causal map is always zero -- the first token can only read itself -- so it is left
    out by default rather than allowed to drag the mean down.
    """
    rows = [attention_entropy(weights[i]) for i in range(from_position, weights.shape[0])]
    n = weights.shape[-1]
    return dict(mean=float(np.mean(rows)) if rows else float("nan"),
                minimum=float(np.min(rows)) if rows else float("nan"),
                maximum=float(np.max(rows)) if rows else float("nan"),
                max_possible=float(np.log(n)), n_rows=len(rows),
                per_row=[round(r, 4) for r in rows])


def multi_head_attention(X: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray,
                         W_O: np.ndarray, n_heads: int, causal: bool = True) -> dict:
    """H heads of width d/H, run independently and concatenated, then projected once.

    One head can express one relation per position. The reason to split the width rather than add
    a second full-width head is that H heads of width d/H cost exactly what one head of width d
    costs -- the division of labour is free.
    """
    d_model = X.shape[1]
    if d_model % n_heads:
        raise ValueError(f"d_model {d_model} is not divisible by n_heads {n_heads}")
    head_dim = d_model // n_heads
    traces, blends = [], []
    for h in range(n_heads):
        sl = slice(h * head_dim, (h + 1) * head_dim)
        trace = scaled_dot_product_attention(X @ W_Q[sl].T, X @ W_K[sl].T, X @ W_V[sl].T,
                                             causal=causal)
        traces.append(trace)
        blends.append(trace.blend)
    concatenated = np.concatenate(blends, axis=1)
    return dict(heads=traces, head_dim=head_dim, n_heads=n_heads,
                weights=np.stack([t.weights for t in traces]),
                concatenated=concatenated, output=concatenated @ W_O.T)


# --------------------------------------------------------------------------------------
# Part D -- Machine B: the checkpoint, copied in under its hashes
# --------------------------------------------------------------------------------------
def freeze_hashes(group: str = FREEZE_GROUP) -> dict[str, str]:
    """The recorded sha256 of each Session 1 checkpoint file, from the course freeze manifest."""
    manifest = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
    return {Path(row["path"]).name: row["sha256"] for row in manifest["groups"][group]["files"]}


def copy_machine_b(source: Path = SESSION01, destination: Path = MACHINE_B) -> dict:
    """Copy the five Session 1 files here, check every hash, and write `minigpt/MANIFEST.json`.

    Session 1's own lab hash-checks its assets where they live, so nothing is moved and nothing in
    `session01_assets` is touched. Idempotent: a file already present with the right hash is left
    alone.
    """
    recorded = freeze_hashes()
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in MACHINE_B_FILES:
        want = recorded.get(name)
        if want is None:
            raise RuntimeError(f"{name} is not in the freeze manifest group '{FREEZE_GROUP}'")
        origin = source / name
        if file_sha256(origin) != want:
            raise RuntimeError(f"{origin} does not match the freeze manifest hash")
        target = destination / name
        if not target.is_file() or file_sha256(target) != want:
            shutil.copy2(origin, target)
        got = file_sha256(target)
        if got != want:
            raise RuntimeError(f"{target} does not match the freeze manifest hash after copying")
        rows.append(dict(file=name, bytes=target.stat().st_size, sha256=got, matches_freeze=True))
    manifest = dict(
        copied="2026-09-20",
        source="labs/session01_assets",
        authority=f"notes/{FREEZE_MANIFEST.name}, group '{FREEZE_GROUP}'",
        note="Byte-identical copies. Session 1's lab hash-checks these files where they live, so "
             "the originals are never moved or modified; this folder is a read-only mirror for "
             "Session 5, and session05_attention.py asserts every hash before loading anything.",
        n=len(rows), files=rows,
    )
    MACHINE_B_MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def check_machine_b(directory: Path = MACHINE_B) -> dict[str, str]:
    """Assert the mirrored files still match their recorded hashes. Returns {name: sha256}.

    `minigpt/MANIFEST.json` ships with the lab and is the authority a clone carries; the course
    freeze manifest is an instructor file, checked as well only where it exists (2026-10-19: the
    release audit ran the notebook from a clone, which has no notes/, and this call failed).
    """
    if not MACHINE_B_MANIFEST.is_file():
        raise RuntimeError(f"{MACHINE_B_MANIFEST} is missing; run copy_machine_b() first")
    recorded = {row["file"]: row["sha256"] for row in
                json.loads(MACHINE_B_MANIFEST.read_text(encoding="utf-8"))["files"]}
    frozen = freeze_hashes() if FREEZE_MANIFEST.is_file() else None
    for name in MACHINE_B_FILES:
        got = file_sha256(directory / name)
        if got != recorded[name]:
            raise RuntimeError(f"{name} does not match minigpt/MANIFEST.json")
        if frozen is not None and got != frozen[name]:
            raise RuntimeError(f"{name} does not match the course freeze manifest")
    return recorded


# --------------------------------------------------------------------------------------
# Session 1's model classes, reproduced verbatim apart from one added return path
# --------------------------------------------------------------------------------------
# These are `session01_demo.py` lines 198-266, copied rather than imported: a lab folder that
# imports another lab folder breaks the moment either is moved, and the checkpoint's state
# dictionary is keyed by these exact attribute names. The only change is `return_attn`, which is
# off unless asked for and never touches the arithmetic: `weights` is already computed on every
# forward pass, and Session 1 simply dropped it on the floor. Dropout still applies to the
# projected output, not to the weights, so training behaves exactly as it did.
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, context: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(context, context)).bool())

    def forward(self, values: torch.Tensor, return_attn: bool = False):
        batch, length, width = values.shape
        query, key, value = self.qkv(values).split(width, dim=2)
        reshaped = []
        for tensor in (query, key, value):
            reshaped.append(
                tensor.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
            )
        query, key, value = reshaped
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.mask[:length, :length], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        output = weights @ value
        output = output.transpose(1, 2).contiguous().view(batch, length, width)
        projected = self.dropout(self.projection(output))
        return (projected, weights) if return_attn else projected


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, context: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, n_heads, context, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor, return_attn: bool = False):
        # The two residual connections: each sublayer adds to its input rather than replacing it.
        if return_attn:
            attended, weights = self.attention(self.norm1(values), return_attn=True)
        else:
            attended, weights = self.attention(self.norm1(values)), None
        values = values + attended
        values = values + self.feed_forward(self.norm2(values))
        return (values, weights) if return_attn else values


class MiniGPT(nn.Module):
    def __init__(self, vocab_size: int, config: dict[str, object]) -> None:
        super().__init__()
        d_model = int(config["d_model"])
        context = int(config["context"])
        n_heads = int(config["n_heads"])
        n_layers = int(config["n_layers"])
        dropout = float(config["dropout"])
        self.context = context
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context, d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(d_model, n_heads, context, dropout) for _ in range(n_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor, return_attn: bool = False):
        _, length = token_ids.shape
        positions = torch.arange(length, device=token_ids.device)
        values = self.token_embedding(token_ids) + self.position_embedding(positions)
        maps = []
        for block in self.blocks:
            if return_attn:
                values, weights = block(values, return_attn=True)
                maps.append(weights)
            else:
                values = block(values)
        logits = self.head(self.final_norm(values))
        return (logits, maps) if return_attn else logits


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def parameter_table(model: nn.Module) -> list[dict]:
    """Where the parameters are, by part. The two embeddings and the head are most of a small model."""
    groups: dict[str, int] = {}
    for name, p in model.named_parameters():
        parts = name.split(".")
        key = ("blocks.*." + ".".join(parts[2:-1])) if parts[0] == "blocks" else ".".join(parts[:-1])
        groups[key] = groups.get(key, 0) + p.numel()
    total = sum(groups.values())
    return [dict(part=k, parameters=v, share=round(v / total, 4))
            for k, v in sorted(groups.items(), key=lambda kv: -kv[1])]


# --------------------------------------------------------------------------------------
# Machine B: load, attend, generate, train
# --------------------------------------------------------------------------------------
@dataclass
class MachineB:
    """The trained model and everything needed to talk to it in characters."""
    model: MiniGPT
    vocabulary: list[str]
    stoi: dict[str, int]
    config: dict
    report: dict
    parameters: int

    def encode(self, text: str) -> torch.Tensor:
        unknown = sorted(set(text) - set(self.stoi))
        if unknown:
            raise ValueError(f"characters outside the 78-character vocabulary: {unknown}")
        return torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.vocabulary[int(i)] for i in ids)


def load_machine_b(directory: Path = MACHINE_B, check: bool = True) -> MachineB:
    """Load Session 1's trained model, after proving the file is the one that was trained.

    Three checks before any weight is used: the five files match their recorded hashes; the class
    defined above produces exactly the state-dictionary keys and shapes the checkpoint carries; and
    the parameter count is the 631,424 Session 1 reported. A checkpoint that loads with missing or
    unexpected keys is a different model wearing the same filename.
    """
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if check:
        check_machine_b(directory)
    config = json.loads((directory / "minigpt_config.json").read_text(encoding="utf-8"))
    vocabulary = json.loads((directory / "minigpt_vocab.json").read_text(encoding="utf-8"))
    report = json.loads((directory / "training_report.json").read_text(encoding="utf-8"))
    model = MiniGPT(len(vocabulary), config)
    state = torch.load(directory / "minigpt_fomc.pt", map_location="cpu", weights_only=True)
    mine, theirs = model.state_dict(), state
    if set(mine) != set(theirs):
        missing, unexpected = sorted(set(mine) - set(theirs)), sorted(set(theirs) - set(mine))
        raise RuntimeError(f"state-dict keys differ; missing {missing}, unexpected {unexpected}")
    bad = [k for k in mine if tuple(mine[k].shape) != tuple(theirs[k].shape)]
    if bad:
        raise RuntimeError(f"state-dict shapes differ at {bad}")
    model.load_state_dict(state)
    model.eval()
    n = parameter_count(model)
    if n != EXPECTED_PARAMETERS:
        raise RuntimeError(f"parameter count is {n:,}, Session 1 recorded {EXPECTED_PARAMETERS:,}")
    if file_sha256(directory / "minigpt_fomc.pt") != report["checkpoint_sha256"]:
        raise RuntimeError("checkpoint hash does not match its own training report")
    stoi = {c: i for i, c in enumerate(vocabulary)}
    return MachineB(model=model, vocabulary=vocabulary, stoi=stoi, config=config, report=report,
                    parameters=n)


@torch.no_grad()
def attention_map(machine: MachineB, text: str) -> dict:
    """Every head's attention over one sentence, plus the entropy of each.

    The rows are query positions and the columns are the positions they read; the upper triangle is
    exactly zero because the model is causal. Low mean entropy is a head that has found something
    to look at, high mean entropy is a head still averaging.
    """
    machine.model.eval()
    ids = machine.encode(text)[None, :]
    if ids.shape[1] > machine.model.context:
        raise ValueError(f"text is {ids.shape[1]} characters, the context is {machine.model.context}")
    _, maps = machine.model(ids, return_attn=True)
    arrays = [m[0].detach().numpy() for m in maps]       # (n_heads, T, T) per layer
    length = ids.shape[1]
    entropy = [[entropy_profile(arrays[l][h])["mean"] for h in range(arrays[l].shape[0])]
               for l in range(len(arrays))]
    upper = max(float(np.abs(np.triu(a[h], 1)).max()) for a in arrays for h in range(a.shape[0]))
    return dict(text=text, characters=length, maps=arrays, entropy=entropy,
                max_possible_entropy=float(np.log(length)),
                largest_weight_above_the_diagonal=upper,
                n_layers=len(arrays), n_heads=int(arrays[0].shape[0]))


@torch.no_grad()
def generate(machine: MachineB, prompt: str = "The Committee decided", n_new: int = 200,
             temperature: float = 0.8, seed: int = SEED) -> str:
    """Sample one character at a time: predict, divide the logits by the temperature, draw, append.

    Temperature is the one dial. Below one the distribution sharpens and the model repeats the
    house phrasing; at one it is the distribution the model actually learned; above one the tail
    gets its share and the spelling goes.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    machine.model.eval()
    torch.manual_seed(seed)
    ids = machine.encode(prompt)[None, :]
    for _ in range(n_new):
        logits = machine.model(ids[:, -machine.model.context:])[0, -1] / temperature
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        ids = torch.cat((ids, nxt.reshape(1, 1)), dim=1)
    return machine.decode(ids[0].tolist())


# --------------------------------------------------------------------------------------
# Training a fresh model, so the notebook can watch the loss fall
# --------------------------------------------------------------------------------------
def corpus_statements(directory: Path = FOMC_DIR, check_hashes: bool = True) -> list[str]:
    """The 245 statement bodies in date order, headers removed, hashes checked."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    recorded = {row["file"]: row["sha256"] for row in manifest["files"]}
    bodies = []
    for path in sorted(Path(directory).glob("fomc_*.txt")):
        if check_hashes and file_sha256(path) != recorded.get(path.name):
            raise RuntimeError(f"{path.name} does not match the corpus manifest")
        body = "\n".join(ln for ln in path.read_text(encoding="utf-8").splitlines()
                         if not ln.startswith("#")).strip()
        if body:
            bodies.append(body)
    return bodies


def corpus_text(directory: Path = FOMC_DIR, check_hashes: bool = True) -> str:
    """The statements joined the way Session 1 joined them: a blank line between each pair.

    Session 1 recorded the sha256 of exactly this string in `training_report.json`, so
    `corpus_agrees_with_training_report` can prove that the fresh model below starts from the same
    text the checkpoint was trained on.
    """
    return "\n\n".join(corpus_statements(directory, check_hashes))


def corpus_agrees_with_training_report(directory: Path = MACHINE_B) -> dict:
    """Is the corpus here byte-for-byte the text Session 1 trained Machine B on?"""
    report = json.loads((directory / "training_report.json").read_text(encoding="utf-8"))
    text = corpus_text()
    got = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return dict(characters=len(text), expected_characters=report["corpus_characters"],
                sha256=got, expected_sha256=report["corpus_sha256"],
                agrees=bool(got == report["corpus_sha256"]
                            and len(text) == report["corpus_characters"]))


def character_split(statements: Sequence[str], vocabulary: Sequence[str], every: int = 20) -> dict:
    """Session 1's split: every 20th statement held out, spread across three decades.

    The unit is a whole statement, never a random window, so a validation loss cannot be read off a
    sentence the model already met in training. `statements` is the list of bodies -- splitting the
    joined text on blank lines would cut it into paragraphs instead, which is a different and much
    leakier split.
    """
    text = "\n\n".join(statements)
    chars = sorted(set(text))
    if list(vocabulary) != chars:
        raise RuntimeError(f"corpus has {len(chars)} distinct characters, the recorded vocabulary "
                           f"has {len(vocabulary)}")
    stoi = {c: i for i, c in enumerate(vocabulary)}
    hold = "\n\n".join(s for i, s in enumerate(statements) if i % every == 0)
    train = "\n\n".join(s for i, s in enumerate(statements) if i % every != 0)
    enc = lambda s: torch.tensor([stoi[c] for c in s], dtype=torch.long)
    return dict(train=enc(train), validation=enc(hold), n_statements=len(statements),
                n_held_out=sum(1 for i in range(len(statements)) if i % every == 0),
                characters=len(text), vocabulary_size=len(vocabulary),
                train_characters=len(train), validation_characters=len(hold), every=every)


def train_a_few_steps(machine: MachineB, n_steps: int = 60, batch: int = 32,
                      learning_rate: float = 3e-4, warmup: int = 100, clip: float = 1.0,
                      seed: int = SEED, evaluate_every: int = 10,
                      sample_prompt: str = "The Committee decided") -> dict:
    """Train a freshly initialised MiniGPT for `n_steps`, from noise, on the same corpus.

    Session 1 ran 3,000 steps in about eighteen minutes and reached a validation loss of 0.761.
    Sixty steps is not a small version of that -- it is the first inch of it, and it is here so the
    loss can be watched falling from ln 78 = 4.36 (a model that knows nothing picks uniformly among
    78 characters) rather than asserted. The recipe is Session 1's: AdamW, lr 3e-4, linear warm-up
    over 100 steps, gradient clipping at 1.0, batches of 32 windows of 128 characters.
    """
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = character_split(corpus_statements(), machine.vocabulary)
    fresh = MiniGPT(len(machine.vocabulary), machine.config)
    if parameter_count(fresh) != EXPECTED_PARAMETERS:
        raise RuntimeError("the fresh model is not the architecture Session 1 trained")
    context = fresh.context
    optimiser = torch.optim.AdamW(fresh.parameters(), lr=learning_rate)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda s: min(1.0, (s + 1) / warmup))
    size = len(machine.vocabulary)

    def batch_from(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(0, len(ids) - context - 1, (batch,))
        x = torch.stack([ids[s: s + context] for s in starts])
        y = torch.stack([ids[s + 1: s + context + 1] for s in starts])
        return x, y

    @torch.no_grad()
    def validation_loss(n_batches: int = 5) -> float:
        fresh.eval()
        losses = [F.cross_entropy(fresh(x).view(-1, size), y.view(-1)).item()
                  for x, y in (batch_from(data["validation"]) for _ in range(n_batches))]
        fresh.train()
        return float(np.mean(losses))

    fresh.train()
    train_curve, validation_curve = [], []
    for step in range(1, n_steps + 1):
        x, y = batch_from(data["train"])
        loss = F.cross_entropy(fresh(x).view(-1, size), y.view(-1))
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(fresh.parameters(), clip)
        optimiser.step()
        schedule.step()
        train_curve.append((step, float(loss.item())))
        if step == 1 or step % evaluate_every == 0 or step == n_steps:
            validation_curve.append((step, validation_loss()))
    machine_fresh = MachineB(model=fresh, vocabulary=machine.vocabulary, stoi=machine.stoi,
                             config=machine.config, report={}, parameters=parameter_count(fresh))
    return dict(
        n_steps=n_steps, batch=batch, learning_rate=learning_rate, warmup=warmup, clip=clip,
        seed=seed, context=context, parameters=parameter_count(fresh),
        uniform_loss=float(np.log(size)),
        first_loss=train_curve[0][1], last_loss=train_curve[-1][1],
        first_validation=validation_curve[0][1], last_validation=validation_curve[-1][1],
        train_curve=[(s, round(v, 4)) for s, v in train_curve],
        validation_curve=[(s, round(v, 4)) for s, v in validation_curve],
        sample_after=generate(machine_fresh, sample_prompt, n_new=120, temperature=0.8),
        session1_final_validation=machine.report.get("final_validation_loss"),
        session1_steps=machine.report.get("steps"),
        data={k: v for k, v in data.items() if k not in ("train", "validation")},
        seconds=round(time.perf_counter() - t0, 2),
    )


# --------------------------------------------------------------------------------------
# The recorded run
# --------------------------------------------------------------------------------------
DEMO_SENTENCE = "The Committee decided to raise the target range for the federal funds rate."
TEMPERATURES = (0.5, 0.8, 1.2)


def _round(x, k: int = 6):
    return np.round(np.asarray(x, dtype=float), k).tolist()


def build_records(path: Path = RECORDED) -> dict:
    """Recompute every number the notebook or a slide might print, and write it to JSON."""
    t0 = time.perf_counter()
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    mirror = copy_machine_b()
    machine = load_machine_b()

    # Part A -- the toy trace, on untrained projections
    tokens = list(BOOK_TOKENS)
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(len(tokens), 16)) / 4.0
    W_Q, W_K, W_V, W_O = random_projections(16, 8, seed=SEED)
    free = attention_from_embeddings(X, W_Q, W_K, W_V, W_O, causal=False)
    causal = attention_from_embeddings(X, W_Q, W_K, W_V, W_O, causal=True)
    query = BOOK_QUERY_POSITION - 1

    # Part B
    book = book_attention()
    lab7b = lab7b_five_score_variant()

    # Part C -- a two-token check anyone can do on paper, and the entropy of the toy maps
    identity = np.eye(2)
    two = scaled_dot_product_attention(identity, identity, identity)
    sigma = math.exp(1 / math.sqrt(2)) / (math.exp(1 / math.sqrt(2)) + 1)
    wide = random_projections(16, 16, seed=SEED + 1)
    heads = multi_head_attention(rng.normal(size=(len(tokens), 16)) / 4.0, *wide, n_heads=4)

    # Part D
    maps = attention_map(machine, DEMO_SENTENCE)
    samples = {f"{tau}": generate(machine, "The Committee decided", n_new=200, temperature=tau)
               for tau in TEMPERATURES}
    fresh = train_a_few_steps(machine, n_steps=60)

    record = dict(
        _meta=dict(
            module=Path(__file__).name, run_date=time.strftime("%Y-%m-%d"), seed=SEED,
            python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
            device="cpu", torch_threads=int(torch.get_num_threads()),
            corpus_manifest_sha256=file_sha256(MANIFEST),
            machine_b_manifest_sha256=file_sha256(MACHINE_B_MANIFEST),
            omp_num_threads=os.environ.get("OMP_NUM_THREADS", "unset"),
            elapsed_seconds=None),
        attention_by_hand=dict(
            tokens=tokens, d_model=int(X.shape[1]), head_dim=int(free.head_dim),
            scale=round(free.scale, 6),
            scale_note="scores are divided by sqrt(d_h); at d_h = 8 that is 1 / 2.8284 = 0.3536",
            query_position=BOOK_QUERY_POSITION, query_token=tokens[query],
            bidirectional=dict(
                scores_raw=_round(free.scores_raw[query], 4),
                scores_scaled=_round(free.scores_scaled[query], 4),
                weights=_round(free.weights[query], 4),
                weights_sum=round(float(free.weights[query].sum()), 10),
                entropy=round(attention_entropy(free.weights[query]), 4)),
            causal=dict(
                scores_masked=[None if not np.isfinite(v) else round(float(v), 4)
                               for v in causal.scores_masked[query]],
                weights=_round(causal.weights[query], 4),
                upper_triangle_is_zero=bool(np.allclose(np.triu(causal.weights, 1), 0.0)),
                row_entropies=[round(attention_entropy(causal.weights[i]), 4)
                               for i in range(len(tokens))],
                entropy_of_last_row=round(attention_entropy(causal.weights[-1]), 4),
                max_possible_entropy=round(float(np.log(len(tokens))), 4)),
            untrained_is_flat=dict(
                note="with untrained projections every row is near-uniform: attention patterns are "
                     "learned, not built into the mechanism",
                mean_row_entropy=round(float(np.mean([attention_entropy(free.weights[i])
                                                      for i in range(len(tokens))])), 4),
                max_possible=round(float(np.log(len(tokens))), 4)),
            two_token_check=dict(
                note="Q = K = V = I2, so the scores are 1/sqrt(2) on the diagonal and 0 off it",
                sigma_by_hand=round(sigma, 6),
                weights=_round(two.weights, 6),
                agrees=bool(np.allclose(two.weights, [[sigma, 1 - sigma], [1 - sigma, sigma]]))),
            multi_head=dict(
                n_heads=heads["n_heads"], head_dim=heads["head_dim"],
                note="four heads of width 4 cost exactly what one head of width 16 costs",
                per_head_mean_entropy=[round(entropy_profile(heads["weights"][h])["mean"], 4)
                                       for h in range(heads["n_heads"])],
                output_shape=list(heads["output"].shape))),
        book_example=dict(**book, lab7b_five_score_variant=lab7b),
        machine_b=dict(
            mirror=mirror,
            architecture=dict(**machine.config, vocabulary_size=len(machine.vocabulary)),
            parameters=machine.parameters,
            parameters_expected=EXPECTED_PARAMETERS,
            parameters_match=bool(machine.parameters == EXPECTED_PARAMETERS),
            parameter_table=parameter_table(machine.model),
            state_dict_keys=sorted(machine.model.state_dict()),
            n_state_dict_keys=len(machine.model.state_dict()),
            training=dict(
                steps=machine.report["steps"], batch_size=machine.report["batch_size"],
                optimizer=machine.report["optimizer"],
                learning_rate=machine.report["learning_rate"],
                seed=machine.report["seed"],
                elapsed_seconds=machine.report["elapsed_seconds"],
                corpus_characters=machine.report["corpus_characters"],
                final_validation_loss=machine.report["final_validation_loss"],
                hold_out="every 20th statement",
                train_curve=machine.report["train_curve"],
                validation_curve=machine.report["validation_curve"]),
            return_attn=dict(
                change="MultiHeadSelfAttention.forward, TransformerBlock.forward and "
                       "MiniGPT.forward take return_attn=False and hand back the softmax weights "
                       "when it is True",
                why="Session 1 computed `weights` on every forward pass and returned only the "
                    "projected output, so the attention map could not be seen",
                numerics="unchanged: the weights are the same tensor the original computed, and "
                         "dropout still applies to the projection, never to the weights",
                state_dict_keys_match=True)),
        attention_maps=dict(
            sentence=maps["text"], characters=maps["characters"],
            n_layers=maps["n_layers"], n_heads=maps["n_heads"],
            max_possible_entropy=round(maps["max_possible_entropy"], 4),
            mean_entropy_by_layer_and_head=[[round(e, 4) for e in layer] for layer in maps["entropy"]],
            lowest_entropy_head=_lowest(maps["entropy"]),
            highest_entropy_head=_lowest(maps["entropy"], highest=True),
            causal_check=dict(largest_weight_above_the_diagonal=maps["largest_weight_above_the_diagonal"],
                              is_zero=bool(maps["largest_weight_above_the_diagonal"] == 0.0)),
            layer1_head1_first_rows=[_round(maps["maps"][0][0][i][: i + 1], 4) for i in range(6)],
            reading="layer 1 stays diffuse across all four heads; by layer 2 at least one head has "
                    "collapsed onto a narrow band. Nobody asked for the division of labour",
            comparison_note=(
                "Lab 7B printed 3.22 / 3.30 / 3.18 / 3.29 for layer 1 and 1.54 / 1.84 / 0.45 / 1.77 "
                "for layer 2 on the same 75-character sentence. Those came from a model trained "
                "inside that notebook; Machine B is Session 1's separate run at seed 2026. Same "
                "architecture, same corpus, different weights -- so the pattern repeats (layer 1 "
                "diffuse near ln T, one layer-2 head far below it) while the digits do not, and "
                "should not be expected to.")),
        generation=dict(
            prompt="The Committee decided", n_new=200, seed=SEED,
            temperatures=list(TEMPERATURES), samples=samples,
            note="one character at a time from a 0.6M-parameter model; the spelling, the house "
                 "phrasing and the sentence ends are all learned from the next-character objective"),
        training_from_scratch=dict(
            **fresh, corpus=corpus_agrees_with_training_report(),
            split_check=dict(
                lab7b_printed=dict(train_characters=537303, validation_characters=31661,
                                   vocabulary_size=78, corpus_characters=568966),
                agrees=bool(fresh["data"]["train_characters"] == 537303
                            and fresh["data"]["validation_characters"] == 31661
                            and fresh["data"]["vocabulary_size"] == 78),
                note="the hold-out unit is a whole statement, 13 of 245")),
    )
    record["_meta"]["elapsed_seconds"] = round(time.perf_counter() - t0, 2)
    Path(path).write_text(json.dumps(record, indent=1, default=_json_default), encoding="utf-8")
    return record


def _lowest(entropy: Sequence[Sequence[float]], highest: bool = False) -> dict:
    flat = [(l + 1, h + 1, e) for l, layer in enumerate(entropy) for h, e in enumerate(layer)]
    layer, head, value = (max if highest else min)(flat, key=lambda row: row[2])
    return dict(layer=layer, head=head, mean_entropy=round(value, 4))


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"{type(obj)} is not JSON serialisable")


def self_test() -> None:
    """Run the whole module and print the numbers the lab turns on."""
    t0 = time.perf_counter()
    copy_machine_b()
    book = book_attention()
    print(f"book example: sum of exponentials {book['sum_of_exponentials']} "
          f"(printed {BOOK_PRINTED['sum_of_exponentials']})")
    for row in book["comparison"]:
        print(f"   {row['quantity']:>26s}  computed {row['computed']:<9.4f} printed {row['printed']:>5s}"
              f"   {'agrees' if row['agrees'] else 'DIFFERS'}")

    machine = load_machine_b()
    print(f"\nMachine B: {machine.parameters:,} parameters, {len(machine.vocabulary)} characters, "
          f"validation loss {machine.report['final_validation_loss']:.4f} after "
          f"{machine.report['steps']:,} steps")

    maps = attention_map(machine, DEMO_SENTENCE)
    print(f"\nattention over {maps['characters']} characters (ln T = {maps['max_possible_entropy']:.2f}):")
    for l, layer in enumerate(maps["entropy"]):
        print(f"   layer {l + 1}: " + "  ".join(f"head {h + 1}: {e:.2f}" for h, e in enumerate(layer)))

    for tau in TEMPERATURES:
        text = generate(machine, "The Committee decided", n_new=120, temperature=tau)
        print(f"\ntemperature {tau}: {text!r}")

    fresh = train_a_few_steps(machine, n_steps=20)
    print(f"\nfresh model, {fresh['n_steps']} steps in {fresh['seconds']} s: "
          f"loss {fresh['first_loss']:.3f} -> {fresh['last_loss']:.3f} "
          f"(uniform would be {fresh['uniform_loss']:.3f})")
    print(f"\n{time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["test"]:
        self_test()
    else:
        rec = build_records()
        print(f"wrote {RECORDED.name} in {rec['_meta']['elapsed_seconds']} s")
        print(f"parameters {rec['machine_b']['parameters']:,} "
              f"(expected {rec['machine_b']['parameters_expected']:,}): "
              f"{rec['machine_b']['parameters_match']}")
        print("book arithmetic agrees with every printed figure:", rec["book_example"]["all_agree"])
        for row in rec["book_example"]["comparison"]:
            print(f"   {row['quantity']:>26s}  {row['computed']:<9.4f} vs {row['printed']}")
