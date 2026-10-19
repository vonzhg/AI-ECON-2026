"""FinBERT 情绪打分：FOMC 风格语句的三分类演示

对应章节：第14章《文本表示与语言模型基础》"BERT家族：双向掩码语言模型"
小节（subsec:bert_family）的 FinBERT 迷你教程；亦为章末练习5的起点。

设定
----
- 模型：ProsusAI/finbert（Hugging Face 模型库）。构建路径见正文：
  取预训练 BERT-Base -> 在路透社 TRC2 金融新闻子集上继续运行 MLM
  （领域适配预训练 DAPT）-> 在 Financial PhraseBank（约 4800 句
  人工标注金融新闻语句）上微调 positive/neutral/negative 三分类
  情感头（Araci, 2019, arXiv:1908.10063）。
- 输入：6 句 FOMC 声明风格的英文句子（直接嵌入代码，无外部数据）。
- 输出：每句的三类概率与预测标签，打印为对齐表格。

说明
----
- 首次联网运行会从 Hugging Face 下载约 440MB 模型权重，之后使用本地缓存。
- 若未安装 transformers，或无法联网下载模型，脚本自动退回到预存的
  示意输出（FALLBACK_ROWS）。这些数值用于展示输出格式与解读方式，
  量级与真实输出一致，但精确数值以实际联网运行为准。无论哪条路径，
  脚本均以退出码 0 结束，便于离线教学环境使用。
- 模型推理不含随机采样，同一版本权重下输出可复现，无需设随机种子。
"""

from __future__ import annotations

import sys

# 6 句 FOMC 声明风格语句（前 5 句与章末练习5的句子列表重合，便于对照）
SENTENCES = [
    "The Committee decided to raise the target range for the federal funds rate.",
    "Job gains have been robust in recent months.",
    "Inflation remains elevated, reflecting supply and demand imbalances.",
    "Recent indicators suggest that spending and production have softened.",
    "The labor market remains strong.",
    "The Committee anticipates that ongoing increases in the target range will be appropriate.",
]

# 预存示意输出：(句子, P(positive), P(negative), P(neutral))
# 数值为示意性的典型输出（每行概率和为 1.00），用于离线环境展示预期结果。
FALLBACK_ROWS = [
    (SENTENCES[0], 0.04, 0.36, 0.60),
    (SENTENCES[1], 0.91, 0.02, 0.07),
    (SENTENCES[2], 0.02, 0.86, 0.12),
    (SENTENCES[3], 0.03, 0.79, 0.18),
    (SENTENCES[4], 0.93, 0.02, 0.05),
    (SENTENCES[5], 0.10, 0.32, 0.58),
]

MODEL_ID = "ProsusAI/finbert"


def score_with_finbert(sentences: list[str]) -> list[tuple[str, float, float, float]]:
    """加载 FinBERT 并逐句打分。失败时抛出异常，由调用方处理回退。"""
    from transformers import pipeline  # 未安装时抛 ImportError

    # 正文中提到的"三行用法"即下面三行：构造 pipeline、传入句子、读取概率。
    clf = pipeline("text-classification", model=MODEL_ID, top_k=None)
    raw = clf(sentences)

    rows = []
    for sentence, scores in zip(sentences, raw):
        probs = {item["label"]: float(item["score"]) for item in scores}
        rows.append(
            (sentence, probs["positive"], probs["negative"], probs["neutral"])
        )
    return rows


def predicted_label(p_pos: float, p_neg: float, p_neu: float) -> str:
    """取概率最大的类别作为预测标签。"""
    pairs = [("positive", p_pos), ("negative", p_neg), ("neutral", p_neu)]
    return max(pairs, key=lambda kv: kv[1])[0]


def print_table(rows: list[tuple[str, float, float, float]], source: str) -> None:
    width = 50  # 句子列宽（超长截断）
    print(f"\n结果来源：{source}")
    print("-" * (width + 38))
    header = f"{'句子':<{width}} {'pos':>6} {'neg':>6} {'neu':>6}  标签"
    print(header)
    print("-" * (width + 38))
    for sentence, p_pos, p_neg, p_neu in rows:
        shown = sentence if len(sentence) <= width else sentence[: width - 3] + "..."
        label = predicted_label(p_pos, p_neg, p_neu)
        print(f"{shown:<{width}} {p_pos:>6.2f} {p_neg:>6.2f} {p_neu:>6.2f}  {label}")
    print("-" * (width + 38))


def main() -> int:
    try:
        rows = score_with_finbert(SENTENCES)
        source = f"实时推理（{MODEL_ID}）"
    except ImportError:
        print("[提示] 未检测到 transformers 库。")
        print("       安装方式：pip install transformers torch")
        print("       以下展示预存的示意输出（格式与真实运行一致）。")
        rows, source = FALLBACK_ROWS, "预存示意输出（未安装 transformers）"
    except Exception as exc:  # 无网络、下载中断、缓存损坏等
        print(f"[提示] 模型加载或推理失败：{type(exc).__name__}: {exc}")
        print("       常见原因：无网络连接，无法从 Hugging Face 下载模型权重。")
        print("       以下展示预存的示意输出（格式与真实运行一致）。")
        rows, source = FALLBACK_ROWS, "预存示意输出（模型不可用）"

    print_table(rows, source)

    print(
        "\n解读：FinBERT 对逐词计数必然误判的上下文依赖措辞（如第 6 句的"
        "\n前瞻性表述）给出中性而非负向判断；与词典法的系统对比见章末练习5。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())