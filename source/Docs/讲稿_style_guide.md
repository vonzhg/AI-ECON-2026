# 中文讲稿 写作规范与黄金范例（Style Guide）

> 本文件是全套《AI for Economic Research》中文讲稿的**统一写作规范**。
> 每一讲的讲稿（`LecXX_讲稿.md`）都必须遵循此规范，使 11 个文件风格、术语、
> 体例完全一致。写每一讲之前，请先通读本文件，尤其是末尾的「黄金范例」。

---

## 1. 这套讲稿是什么 / 怎么用

- **用途**：授课老师（冯志钢）面向**中国研究生**用中文授课，幻灯片是英文的。
  讲稿提供**可直接照读的中文逐字稿**，让老师讲每一张英文幻灯片时，不必临场翻译
  公式与术语，也能讲得流畅、连贯。
- **形态**：每一讲一个 Markdown 文件，放在该讲文件夹内（如
  `Lec03_ML_for_Macro/Lec03_讲稿.md`）。与 LaTeX 构建完全解耦——Markdown，
  无需编译，可在第二屏幕、平板或打印稿上阅读。
- **逐字稿（verbatim）**：每一张幻灯片对应一段**完整的、可照着念的中文口语**，
  不是要点提纲。老师可以一字不差地读，也可以据此自由发挥。

---

## 2. 文件结构模板

每个 `LecXX_讲稿.md` 严格按以下结构：

```markdown
# LecXX 讲稿：<中文讲题>（<English lecture title>）

> 配套幻灯片：<本讲所有 topic deck 的文件名>
> 建议时长：X 小时 ｜ 先修：<前置知识>

## 本讲概览
<2–4 句：本讲要解决什么问题、为什么重要、与上一讲/下一讲怎么衔接>

## 学习目标
- <3–6 条，动词开头，如「能解释……」「能推导……」「能用 PyTorch 实现……」>

## 关键术语表（中英对照）
| 英文 | 中文 | 一句话说明 |
|------|------|-----------|
| ... | ... | ... |
（继承第 6 节 master glossary 中本讲用到的词，并补充本讲特有术语）

---

## Topic 1：<deck 文件名>（共 N 张）

### 幻灯片 1：<英文 frame 标题>（<中文意译>）
<逐字讲稿>

### 幻灯片 2：<英文 frame 标题>（<中文意译>）
<逐字讲稿>

...

## Topic 2：<deck 文件名>（共 N 张）
...
```

**要点**：
- 每个 topic deck 用一个 `## Topic k：<文件名>（共 N 张）` 分节。
- 每一张幻灯片用一个 `### 幻灯片 N：英文标题（中文意译）` 标题，**N 按该 deck 内
  从 1 开始连续编号**（标题页/title 页可记为「幻灯片 0」或并入「幻灯片 1」前的
  开场白，见下）。
- **覆盖完整**：deck 里每一个 `\begin{frame}` 都要有对应条目（分节页/纯图页见
  第 4 节的特殊处理）。这样 `### 幻灯片` 的条数应与 `\begin{frame}` 数量基本一致。

---

## 3. 逐字稿的质量标准（核心）

1. **口语化、可照读**：用第二人称面向学生（"我们来看…"、"大家注意…"、
   "这里我提醒一句…"）。按 **问题 → 方法 → 为什么重要** 的节奏展开。
   英文 bullet 是**改写、串讲**，不是逐句直译。每张 3–8 句；信息密集的张数可更长。
2. **公式必须"读出来 + 讲含义"**：先用中文把公式念成一句话，再点出关键项和直觉，
   **不要只照抄 LaTeX**。例：`v(k)=\max_{k'}\{u(\cdot)+\beta v(k')\}` →
   "价值函数 v(k) 等于：在所有可行的下期资本 k′ 里，挑一个让'当期效用加上 β 乘
   下期价值'最大的。这里 β 是贴现因子，体现我们对未来的耐心。"
3. **术语中英对照**：技术术语**首次出现**给中英对照，如"神经元 (neuron)"、
   "Bellman 方程"、"actor-critic（演员-评论家）"，之后用中文即可；译名须与
   第 6 节 master glossary 一致。专有名词/缩写（ReLU、PyTorch、Transformer、
   PPO、Word2Vec、RAGAS 等）保留英文。
4. **代码块 (lstlisting)**：用中文讲清这段代码**做什么、为什么这么写、关键行在哪**，
   不要逐行念语法。可点名 1–2 个关键函数/张量操作。
5. **图 / TikZ 示意图**：描述图在展示什么、**讲课时鼠标/激光笔该指向哪里**、
   要点是什么（例："我会从左往右点一遍：输入 → 加权求和 → 激活 → 输出"）。
6. **配色框（box）**对应口吻：
   - `conceptbox` → "核心概念是……"
   - `cautionbox` → "这里要特别小心 / 一个常见误区是……"
   - `examplebox` → "举个例子……"
   - `takeawaybox` → "这一段一句话总结：……"
7. **"For economists:" 旁注**是天然的讲解桥梁——务必展开成连接经济学直觉的话
   （"给经济学背景的同学点一句……"）。
8. **衔接语**：每个 topic 开头给一句承上启下；deck 之间、讲与讲之间用
   "上一节我们…，这一节…" 自然过渡。幻灯片里出现的 "Next (T3)…" /
   "Looking forward (Lec.7)…" 之类前瞻，要在讲稿里说出来。

**语域 / 语气**：研究生水平，默认听众熟悉 Bellman 方程与基本线性代数；精确但温暖、
善于打比方（尤其用经济学直觉解释 ML 概念）。避免翻译腔；像一位有耐心的老师在
讲台上说话。

---

## 4. 特殊幻灯片的处理

- **标题页 (`\makebeamertitle`)**：不单列一条，写成一句开场白并入本讲第一段
  （或作为「## Topic 1」前的引入）。可在「本讲概览」里体现。
- **Agenda / 目录页**：一两句话交代"这一节我们讲哪几件事"，不必逐条展开。
- **分节页 (`\sectiondivider{...}{...}`) 或 `[plain]` 纯过渡页**：**不写整段逐字
  稿**，给一句过渡语即可，例："下面进入第二部分——……"。仍保留一个 `###` 条目，
  标题后注明（分节页）。
- **纯图页（一张大图、无文字）**：按第 3 节第 5 点，讲"图在说什么、指哪里"。
- **表格页**：用中文把表格"读"成对比性的话（"左列是经典方法，右列是 RL 方法，
  关键差别在……"），不要逐格念。
- **附录页 (Appendix)**：可写简短讲稿，并注明"（附录/选讲，时间紧可跳过）"。

---

## 5. 输出与一致性约定

- **输出文件名**（务必按此命名，放在对应讲文件夹内）：
  `Lec02_讲稿.md`、`Lec03_讲稿.md` …；两个 Lec01 轨道分别为
  `Lec01_Introduction_讲稿.md`、`Lec01_Quant_Macro_讲稿.md`。
- **以磁盘上的 `.tex` 为准**：先列出该文件夹下所有 `LecXX_T*.tex` 教学 deck
  并**逐个读全**，按文件名中的 T1、T2…顺序排列。跳过非 deck 的辅助文件
  （如 `infra_stack.tex`、`interface.tex`、`retrieval_flow.tex`、
  `MPE_Hyper_v44.tex`、`*_paper.tex`）。
- **中文标点**：正文用中文全角标点（，。、：；""）；公式、变量名、英文术语周围
  可用半角，自然即可。
- **不改动任何 `.tex` 或 PDF**：只新建/写入 `*_讲稿.md`。

---

## 6. Master Glossary（统一译名，按需扩展）

> 写讲稿时，下列术语**必须用此译名**，确保 11 个文件一致。每讲在自己的
> 「关键术语表」中收录本讲用到的词，并可补充本讲特有术语。

**动态规划 / 宏观（DP & Macro）**
| EN | 中文 | EN | 中文 |
|----|------|----|------|
| dynamic programming | 动态规划 | value function iteration (VFI) | 价值函数迭代 |
| Bellman equation | Bellman 方程 | Euler equation | 欧拉方程 |
| value function | 价值函数 | policy function | 策略函数 |
| state (variable) | 状态(变量) | control / action | 控制 / 动作 |
| contraction mapping | 压缩映射 | fixed point | 不动点 |
| principle of optimality | 最优性原理 | discount factor β | 贴现因子 β |
| optimal growth model | 最优增长模型 | competitive equilibrium | 竞争性均衡 |
| recursive equilibrium | 递归均衡 | calibration | 校准 |
| stationary distribution | 平稳分布 | heterogeneous-agent (HA) | 异质性主体 |
| OLG | 世代交叠(模型) | mean-field game (MFG) | 平均场博弈 |
| HJB equation | HJB 方程 | Kolmogorov forward eq. (KFE) | Kolmogorov 前向方程 |

**机器学习 / 深度学习（ML & DL）**
| EN | 中文 | EN | 中文 |
|----|------|----|------|
| machine learning | 机器学习 | deep learning | 深度学习 |
| supervised learning | 监督学习 | unsupervised learning | 无监督学习 |
| neural network | 神经网络 | neuron | 神经元 |
| (hidden) layer | (隐藏)层 | depth | 深度 |
| weight | 权重 | bias | 偏置 |
| activation function | 激活函数 | loss function | 损失函数 |
| gradient descent | 梯度下降 | SGD | 随机梯度下降 |
| forward pass | 前向传播 | backpropagation | 反向传播 |
| overfitting | 过拟合 | regularization | 正则化 |
| training/validation/test set | 训练/验证/测试集 | function approximation | 函数逼近 |
| universal approximation thm | 通用逼近定理 | curse of dimensionality | 维度灾难 |
| feature | 特征 | representation | 表示 |
| tensor | 张量 | autograd | 自动微分 |

**强化学习（RL）**
| EN | 中文 | EN | 中文 |
|----|------|----|------|
| reinforcement learning (RL) | 强化学习 | agent | 智能体 |
| environment | 环境 | reward | 奖励 |
| return | 回报 | Markov decision process (MDP) | 马尔可夫决策过程 |
| policy | 策略 | Monte Carlo (MC) | 蒙特卡洛 |
| temporal difference (TD) | 时序差分 | Q-learning | Q 学习 |
| exploration/exploitation | 探索/利用 | actor-critic | 演员-评论家 |
| on-policy / off-policy | 同策略 / 异策略 | policy gradient | 策略梯度 |
| bootstrapping | 自举 | advantage | 优势(函数) |

**大语言模型 / 文本（LLM & Text）**
| EN | 中文 | EN | 中文 |
|----|------|----|------|
| large language model (LLM) | 大语言模型 | tokenization | 分词 |
| token | 词元 | embedding | 嵌入(向量) |
| word embedding | 词嵌入 | self-attention | 自注意力 |
| multi-head attention | 多头注意力 | query/key/value | 查询/键/值 (Q/K/V) |
| positional encoding | 位置编码 | pre-training | 预训练 |
| fine-tuning | 微调 | SFT | 监督微调 |
| RLHF | 人类反馈强化学习 | DPO | 直接偏好优化 |
| scaling law | 缩放定律 | hallucination | 幻觉 |
| prompt | 提示词 | | |

**RAG / 智能体（RAG & Agentic）**
| EN | 中文 | EN | 中文 |
|----|------|----|------|
| retrieval-augmented generation (RAG) | 检索增强生成 | chunking | 分块/切块 |
| vector database / store | 向量数据库/向量库 | retrieval | 检索 |
| dense retrieval | 稠密检索 | hybrid retrieval | 混合检索 |
| GraphRAG | 图检索增强 | knowledge graph | 知识图谱 |
| agentic AI | 智能体式 AI | terminal agent | 终端智能体 |
| tool use | 工具调用 | sub-agent | 子智能体 |
| context window | 上下文窗口 | homotopy | 同伦(延拓) |
| provenance | 数据溯源 | weighted statistics | 加权统计 |

（缩写如 ReLU、Adam、PyTorch、Transformer、Word2Vec、TF-IDF、A2C/PPO/DDPG/SAC、
RAGAS、CLAUDE.md 等保留英文；首次出现可加一句中文解释。）

---

## 7. 黄金范例（GOLD STANDARD —— 务必照此水准与口吻）

> 下列范例取自真实幻灯片 `Lec03_ML_for_Macro/Lec03_T2_NN_Concepts.tex`。
> **这就是合格逐字稿的样子**：公式读成话、术语中英对照、有经济学桥梁、有衔接。

### 范例 A —— 概念/嵌套要点页

**对应幻灯片：`Neural Networks: The Basic Building Block`**

> ### 幻灯片 2：Neural Networks: The Basic Building Block（神经网络的基本构件）
> 好，我们正式进入神经网络 (neural network)。先建立一个最朴素的图景：人工神经网络
> 的灵感来自生物神经系统，它由大量相互连接的"神经元 (neuron)"组成，这些神经元被
> 组织成一层一层的结构；而所谓"训练"，本质上就是根据数据不断调整连线上的权重
> (weights)。大家这一张只要抓住三个核心部件就够了。第一，**神经元**——它接收若干
> 输入，套上一个激活函数，然后吐出一个输出。第二，**激活函数 (activation
> function)**——这是每个神经元施加的非线性映射，常见的有 sigmoid、tanh、ReLU，
> 它是整个网络"非线性能力"的来源。第三，**损失函数 (loss function)**——它衡量我们
> 的预测和真实值差多远，是后面训练时要最小化的目标。神经元、激活函数、损失函数，
> 这三个词会贯穿后面两讲，现在先记住它们各自的分工。

### 范例 B —— 含公式页（公式读成话）

**对应幻灯片：`Single Hidden Layer Network`**

> ### 幻灯片 4：Single Hidden Layer Network（单隐藏层网络）
> 现在我们把单个神经元拼成一个最简单的网络——单隐藏层网络。先约定记号：输入是一个
> M 维向量 x，输出是 N 维向量 y，中间夹着一层隐藏层。先看隐藏层里第 j 个神经元
> 怎么算，也就是屏幕上第一个公式：它先把所有输入做一次**加权求和**——权重 w 乘以
> 输入 x、对 m 从 1 到 M 求和，再加上一个偏置 b——然后整体套上激活函数 σ。一句话，
> "**先线性、后非线性**"，这是神经网络最核心的一个动作，后面每一层都是在重复它。
> 再看第二个公式，输出层：输出 y_i 就是把隐藏层那些 h 再做一次线性组合、加上偏置，
> 注意输出这一步通常**不再**加激活函数。最后强调一句：σ 必须是非线性的，常用的就是
> tanh、ReLU（也就是 max(0, x)）和 sigmoid。下一张我们就把隐藏层叠成两层，看看
> "深度"到底带来什么。

### 范例 C —— 定理框 + 经济学旁注 + 前瞻页

**对应幻灯片：`Why Neural Networks Work: Theoretical Guarantees`**

> ### 幻灯片 12：Why Neural Networks Work: Theoretical Guarantees（神经网络为何
> 有效：两条理论保证）
> 我们已经会搭网络了，但有个理论问题必须回答：凭什么相信神经网络"什么函数都能
> 近似"？这一张给出两块基石。先看蓝框，**通用逼近定理 (Universal Approximation
> Theorem，Cybenko 1989、Hornik 1991)**：只要激活函数是 sigmoid 型的，定义在
> 单位立方体 [0,1]^d 上的任意连续函数，都能用一个两层神经网络一致逼近；后来 Hornik
> 把条件放宽到任意有界、非常数的激活函数，Leshno 等人 1993 年进一步锐化为"只要不是
> 多项式"就行。一句话——**表达能力不是问题**。再看绿框，Barron 1993 年的结果，讲的
> 是"维度灾难 (curse of dimensionality)"：单层网络的积分平方误差是 O(1/M) 这个
> 量级，M 是节点数；而多项式、样条这类级数逼近是 O(1/M^{2/N})，N 是输入维度——维度
> 一高，就指数级地变差。这里我专门给经济学背景的同学点一句 (For economists)：
> Barron 这个结果，正好解释了为什么在**高维状态空间**里神经网络还能用，而 Chebyshev
> 多项式那一套会崩掉——这正是我们后面用神经网络去解高维宏观模型的理论依据。最后预告
> 一下：到第七讲讲 Transformer 的注意力机制时，背后还是同一个通用逼近定理——架构在
> 变，逼近理论没变。

### 范例 D —— 纯图页（讲"指哪里"）

**对应幻灯片：`Single Neuron: Visual`**

> ### 幻灯片 3：Single Neuron: Visual（单个神经元示意图）
> 这一张是一张示意图，把刚才说的单个神经元画了出来。大家看：左边是若干输入 x_1 到
> x_M，每一条连线对应一个权重；它们汇聚到中间这个节点做加权求和、再加偏置；然后经过
> 右边的激活函数，输出一个数。我讲的时候会用激光笔从左往右点一遍——输入 → 加权求和
> → 激活 → 输出。这张图要让大家看清一件事：**信息是单向地从左流到右的**，这就是
> 后面"前向传播"的雏形。

### 范例 E —— Agenda 页（一句带过）

**对应幻灯片：`Topic 3.2 Agenda`**

> ### 幻灯片 1：Topic 3.2 Agenda（本节路线图）
> 这一节我们就讲两件事：先把神经网络的**结构**搭起来——神经元、层、深度；再讲
> **激活函数**，并由此引出"通用逼近定理"。讲完这两块，下一节 T3 我们就能讲怎么
> "训练"它了。

---

*规范结束。开写前请确认：① 已读完本讲所有 `.tex`；② 译名对齐 master glossary；*
*③ 每张幻灯片都有逐字稿；④ 公式都读成了话。*
