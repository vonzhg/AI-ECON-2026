"""Fast, offline demos for the Session 1 course checkout.

The growth example is a deliberately small Euler-residual solver.  The language
example uses the same decoder-only architecture as Lab 7B, but loads a checkpoint
so the opening-session activity stays inside fifteen minutes.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_report() -> dict[str, object]:
    """Return the small environment contract shared by every later lab."""
    required_assets = (
        "growth_reference.pt",
        "growth_report.json",
        "minigpt_fomc.pt",
        "minigpt_config.json",
        "minigpt_vocab.json",
        "training_report.json",
    )
    assets = {name: (HERE / name).is_file() for name in required_assets}
    return {
        "ready": all(assets.values()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_threads": torch.get_num_threads(),
        "working_directory": str(Path.cwd()),
        "asset_directory": str(HERE),
        "assets": assets,
    }


class GrowthPolicy(nn.Module):
    """Feasible policy k' = f(k) * sigmoid(g_theta(log k))."""

    def __init__(
        self,
        alpha: float = 0.33,
        beta: float = 0.96,
        productivity: float = 1.0,
        hidden: int = 32,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.productivity = productivity
        self.k_ss = (alpha * beta * productivity) ** (1.0 / (1.0 - alpha))
        self.k_min = 0.2 * self.k_ss
        self.k_max = 2.0 * self.k_ss
        self.log_center = (math.log(self.k_min) + math.log(self.k_max)) / 2.0
        self.log_scale = (math.log(self.k_max) - math.log(self.k_min)) / 2.0
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def resources(self, capital: torch.Tensor) -> torch.Tensor:
        return self.productivity * capital.pow(self.alpha)

    def forward(self, capital: torch.Tensor) -> torch.Tensor:
        normalized = (torch.log(capital) - self.log_center) / self.log_scale
        saving_share = torch.sigmoid(self.net(normalized))
        return self.resources(capital) * saving_share

    def analytical_policy(self, capital: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.beta * self.resources(capital)


def _growth_residual(model: GrowthPolicy, capital: torch.Tensor) -> torch.Tensor:
    next_capital = model(capital)
    consumption = model.resources(capital) - next_capital
    next_next_capital = model(next_capital)
    next_consumption = model.resources(next_capital) - next_next_capital
    log_lhs = -torch.log(consumption)
    log_rhs = (
        math.log(model.beta * model.alpha * model.productivity)
        + (model.alpha - 1.0) * torch.log(next_capital)
        - torch.log(next_consumption)
    )
    return log_lhs - log_rhs


def growth_diagnostics(model: GrowthPolicy, n_grid: int = 200) -> dict[str, object]:
    model.eval()
    with torch.no_grad():
        grid = torch.logspace(
            math.log10(model.k_min), math.log10(model.k_max), n_grid
        ).reshape(-1, 1)
        learned = model(grid)
        exact = model.analytical_policy(grid)
        relative_error = (learned - exact).abs() / exact
        residual = _growth_residual(model, grid).abs()
    return {
        "capital": grid.squeeze().numpy(),
        "learned": learned.squeeze().numpy(),
        "exact": exact.squeeze().numpy(),
        "median_relative_error": float(relative_error.median()),
        "max_relative_error": float(relative_error.max()),
        "residual_p95": float(torch.quantile(residual, 0.95)),
    }


def train_growth_policy(
    steps: int = 500,
    batch_size: int = 256,
    seed: int = 2026,
    learning_rate: float = 3e-3,
) -> tuple[GrowthPolicy, list[tuple[int, float]], float]:
    """Train from primitives only; no target policy enters the loss."""
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = GrowthPolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[tuple[int, float]] = []
    started = time.perf_counter()
    for step in range(steps + 1):
        uniform = torch.rand(batch_size, 1)
        capital = torch.exp(
            math.log(model.k_min)
            + uniform * (math.log(model.k_max) - math.log(model.k_min))
        )
        loss = _growth_residual(model, capital).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or step % 50 == 0 or step == steps:
            history.append((step, float(loss.detach())))
    return model, history, time.perf_counter() - started


def load_growth_reference() -> tuple[GrowthPolicy, dict[str, object]]:
    """Load the validated fallback without changing the from-scratch default."""
    checkpoint = HERE / "growth_reference.pt"
    report_path = HERE / "growth_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if file_sha256(checkpoint) != report["checkpoint_sha256"]:
        raise RuntimeError("growth checkpoint hash does not match its report")
    model = GrowthPolicy()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, report


def plot_growth_before_after(
    before: dict[str, object],
    after: dict[str, object],
    history: list[tuple[int, float]],
):
    """Build the opening lab's main visual."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for diagnostics, title, axis in (
        (before, "Before: random policy", axes[0]),
        (after, "After: Euler residual trained", axes[1]),
    ):
        axis.plot(diagnostics["capital"], diagnostics["exact"], "--", lw=2, label="analytical")
        axis.plot(diagnostics["capital"], diagnostics["learned"], lw=2, label="network")
        axis.set_title(title)
        axis.set_xlabel("capital today")
        axis.set_ylabel("capital tomorrow")
        axis.legend(frameon=False)
    steps, losses = zip(*history)
    axes[2].semilogy(steps, losses, color="#B4493D", lw=2)
    axes[2].set_title("The equation becomes the loss")
    axes[2].set_xlabel("training step")
    axes[2].set_ylabel("mean squared log Euler residual")
    fig.tight_layout()
    return fig


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

    def forward(self, values: torch.Tensor) -> torch.Tensor:
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
        return self.dropout(self.projection(output))


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

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values + self.attention(self.norm1(values))
        return values + self.feed_forward(self.norm2(values))


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

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, length = token_ids.shape
        positions = torch.arange(length, device=token_ids.device)
        values = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            values = block(values)
        return self.head(self.final_norm(values))


def load_minigpt() -> tuple[MiniGPT, dict[str, int], list[str], dict[str, object]]:
    # Tiny matrix operations are faster and more reproducible without thread oversubscription.
    torch.set_num_threads(1)
    config = json.loads((HERE / "minigpt_config.json").read_text(encoding="utf-8"))
    vocabulary = json.loads((HERE / "minigpt_vocab.json").read_text(encoding="utf-8"))
    stoi = {character: index for index, character in enumerate(vocabulary)}
    model = MiniGPT(len(vocabulary), config)
    checkpoint = HERE / "minigpt_fomc.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    report = json.loads((HERE / "training_report.json").read_text(encoding="utf-8"))
    if file_sha256(checkpoint) != report["checkpoint_sha256"]:
        raise RuntimeError("MiniGPT checkpoint hash does not match its training report")
    return model, stoi, vocabulary, report


@torch.no_grad()
def generate_text(
    model: MiniGPT,
    stoi: dict[str, int],
    vocabulary: list[str],
    prompt: str = "The Committee decided",
    n_new: int = 200,
    temperature: float = 0.8,
    seed: int = 2026,
) -> str:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    unknown = sorted(set(prompt) - set(stoi))
    if unknown:
        raise ValueError(f"prompt contains characters outside the vocabulary: {unknown}")
    torch.manual_seed(seed)
    token_ids = torch.tensor([[stoi[character] for character in prompt]], dtype=torch.long)
    for _ in range(n_new):
        logits = model(token_ids[:, -model.context :])[0, -1] / temperature
        next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
        token_ids = torch.cat((token_ids, next_id.reshape(1, 1)), dim=1)
    return "".join(vocabulary[index] for index in token_ids[0].tolist())
