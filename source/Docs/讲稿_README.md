# 中文讲稿 索引（Chinese Teaching Scripts — Index）

面向中国研究生、用中文授课的**逐字讲稿**。每一讲一个 Markdown 文件，与英文
Beamer 幻灯片**逐张对应**：老师讲每一张幻灯片时，可直接照读对应的中文段落，
也可据此自由发挥。Markdown 与 LaTeX 构建完全解耦——可在第二屏幕、平板或打印稿
上阅读，无需编译。

## 怎么用

1. 上课时一边放英文幻灯片，一边对照本讲的 `LecXX_讲稿.md`。
2. 每张幻灯片对应一个 `### 幻灯片 N：英文标题（中文意译）` 段落；编号在每个
   topic deck 内从 1 开始，与该 deck 的放映顺序一致。
3. 公式已被"读成话"并解释；术语首次出现给中英对照；图给出"指哪里"的提示；
   `For economists` 旁注已展开为经济学直觉。

## 写作规范与术语

- **`讲稿_style_guide.md`** —— 统一写作规范、~110 词中英对照 master glossary、
  以及 5 个黄金范例。新增/修订讲稿时请先读它，保证 11 个文件风格一致。

## 合并 PDF（讲稿_合集.pdf）

- **`讲稿_合集.pdf`** —— 全部十一篇讲稿合并成一份 PDF（约 267 页，含封面、目录、
  每讲一章、每张幻灯片一个带编号的小节），便于整本翻阅或打印。
- 重新生成（编辑任意 `LecXX_讲稿.md` 之后）：运行 `bash 讲稿_pdf_build/build.sh`，
  会用 lualatex（TeX Live 的 `markdown` + `ctex` + Fandol 字体）重新编译并覆盖
  `讲稿_合集.pdf`。构建脚本与样式见 `讲稿_pdf_build/`。

## 各讲讲稿

> 第一讲有两条轨道：`Lec01_Introduction`（AI 与经济学导论，较新）与
> `Lec01_Quant_Macro`（定量宏观 / 动态规划）。按你实际开课的版本选用，另一条可忽略。

| 讲 | 讲稿文件 | 张数 | 主题 |
|----|---------|------|------|
| 1a | [Lec01_Introduction/Lec01_Introduction_讲稿.md](../Lec01_Introduction/Lec01_Introduction_讲稿.md) | 47 | AI 与经济学、AI 作为研究合作者、课程工具与项目 |
| 1b | [Lec01_Quant_Macro/Lec01_Quant_Macro_讲稿.md](../Lec01_Quant_Macro/Lec01_Quant_Macro_讲稿.md) | 54 | 定量宏观、动态规划、均衡、AI/RL 替代方案 |
| 2 | [Lec02_What_is_AI/Lec02_讲稿.md](../Lec02_What_is_AI/Lec02_讲稿.md) | 34 | 机器学习三支柱、预测栈、生成式与智能体式 AI |
| 3 | [Lec03_ML_for_Macro/Lec03_讲稿.md](../Lec03_ML_for_Macro/Lec03_讲稿.md) | 74 | 模式识别、神经网络、训练与反向传播、专用网络 |
| 4 | [Lec04_DL_RL_Macro/Lec04_讲稿.md](../Lec04_DL_RL_Macro/Lec04_讲稿.md) | 75 | 神经网络 VFI、actor-critic、欧拉方程、结构化网络 |
| 5 | [Lec05_RL_Nutshell/Lec05_讲稿.md](../Lec05_RL_Nutshell/Lec05_讲稿.md) | 80 | RL 框架、蒙特卡洛 / TD / Q-learning、actor-critic |
| 6 | [Lec06_HA_Models/Lec06_讲稿.md](../Lec06_HA_Models/Lec06_讲稿.md) | 123 | Aiyagari、KS/DeepHAM、DEQN、OLG、平均场博弈 |
| 7 | [Lec07_LLM/Lec07_讲稿.md](../Lec07_LLM/Lec07_讲稿.md) | 125 | 分词、词嵌入、注意力、Transformer、LLM、文本分析 |
| 8 | [Lec08_RAG/Lec08_讲稿.md](../Lec08_RAG/Lec08_讲稿.md) | 99 | 提示词、分块与嵌入、向量 / 图检索、增强生成 |
| 9 | [Lec09_Agentic_AI/Lec09_讲稿.md](../Lec09_Agentic_AI/Lec09_讲稿.md) | 135 | 终端智能体、工作流、项目组织、排错与上下文 |
| 10 | [Lec10_Case_Studies/Lec10_讲稿.md](../Lec10_Case_Studies/Lec10_讲稿.md) | 166 | 六个 AI 辅助研究案例 + 综合（另含 18 张分节页） |

合计约 **1012 张幻灯片**的逐字讲稿（Lec10 另含 18 张分节页过渡）。

## 范围与维护说明

- 覆盖 `FullCourse_10Lecs` 的全部 10 讲（含两条 Lec01 轨道）。**不含**
  `MiniCourse_8hr/` 与 `Labs/`、`Notebooks/` 实验册（可后续补充）。
- **以磁盘上的 `.tex` 为准**生成；个别 deck（如 `Lec10_T8_Case6_DeepRamsey.tex`）
  内联了 preamble，故其真实内容帧数以讲稿为准（该 deck 为 32 张，而非 grep 的 34）。
- 校验各讲覆盖完整性：
  `grep -c '^### 幻灯片' LecXX_*/LecXX_讲稿.md` 应等于该文件夹
  `grep -cF '\begin{frame}' LecXX_*/Lec*_T*.tex` 的内容帧数。
- 幻灯片或 `.tex` 改动后，按 `讲稿_style_guide.md` 同步更新对应段落即可。
