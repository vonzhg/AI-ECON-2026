"""Ten business-school discussion questions for the RAG-on-Ren demo.

Pedagogical principle:
- A generic LLM (no retrieval) gives a textbook MBA-style answer.
- The RAG system, retrieving from Ren Zhengfei's actual speeches/interviews,
  surfaces a *distinctive, often counter-intuitive* position.
- The divergence between the two is the teaching moment.

Each entry includes:
- q                  : the question (Chinese, business-school framing).
                       KEPT IN CHINESE ON PURPOSE: this string is the retrieval
                       query against a Chinese corpus — translating it would
                       cripple TF-IDF matching.
- q_en               : English gloss of the question (display only, never
                       used for retrieval)
- theme / theme_en   : topic label, Chinese + English
- conventional_view / conventional_view_en : what a naive LLM is most likely
                       to say (1 line, zh + en)
- ren_view / ren_view_en : the position Ren actually argues in the corpus
                       (1 line, zh + en)
- divergence_note    : 1-2 sentences for the instructor — what to highlight
                       (Chinese only; instructor-facing)
- anchors            : substrings of source filenames the retriever should hit
                       (sanity check, not the central artifact)
"""

QUESTIONS = [
    {
        "id": 1,
        "theme": "中美科技战 / 自主可控",
        "theme_en": "US-China tech war / self-reliance",
        "q": "面对美国的技术封锁和实体清单，中国科技公司应该全面'去美化'实现完全自主可控，还是继续深度融入全球供应链？",
        "q_en": "Facing US technology blockades and the Entity List, should Chinese tech companies fully 'de-Americanize' toward complete self-reliance, or stay deeply integrated in global supply chains?",
        "conventional_view": "应当加快自主可控、降低对美国的依赖。",
        "conventional_view_en": "Accelerate self-reliance and reduce dependence on the US.",
        "ren_view": "自主只是'备胎'不是目标；华为仍要每年从高通买5000万套芯片，甚至愿意把5G授权给西方公司换取共生。",
        "ren_view_en": "Self-reliance is only a 'spare tire', not the goal; Huawei still buys 50M Qualcomm chipsets a year and would even license 5G to Western firms for symbiosis.",
        "divergence_note": "Ren的反直觉立场——把自主可控作为底线而非目标，把开放协作作为优先选择——正是商学院学生最容易错过的判断。",
        "anchors": ["20190521", "20190624", "20190910", "20120712"],
    },
    {
        "id": 2,
        "theme": "5G 战略",
        "theme_en": "5G strategy",
        "q": "5G 真的是基础设施级别的革命吗？运营商和企业应该激进部署还是观望等待杀手应用？",
        "q_en": "Is 5G really an infrastructure-level revolution? Should carriers and enterprises deploy aggressively or wait for a killer app?",
        "conventional_view": "5G是数字经济的基础设施，应当抓紧部署。",
        "conventional_view_en": "5G is the infrastructure of the digital economy — deploy fast.",
        "ren_view": "5G 被严重夸大了；真正价值在B2B/工业互联网，消费者用4G已经足够；不要让消费端炒作绑架部署节奏。",
        "ren_view_en": "5G is wildly over-hyped; the real value is B2B / industrial internet, 4G is plenty for consumers — don't let consumer hype dictate the rollout pace.",
        "divergence_note": "作为5G最大供应商之一，任正非反而泼冷水——这种'卖锄头的不吹牛'的姿态对学生很有冲击力。",
        "anchors": ["20181017", "20190521", "20190624", "20190910", "20160505", "20131228"],
    },
    {
        "id": 3,
        "theme": "周期管理 / 研发投入",
        "theme_en": "Downturn management / R&D spending",
        "q": "经济下行或制裁压力下，科技公司应该裁员降本以保住现金流，还是继续加大研发投入？",
        "q_en": "Under a downturn or sanctions pressure, should a tech company cut staff and costs to protect cash flow, or increase R&D investment?",
        "conventional_view": "现金流优先，先裁员、削减非核心开支、保住生存。",
        "conventional_view_en": "Cash flow first: lay off, cut non-core spending, survive.",
        "ren_view": "越是冬天越要投研发；用战略预备队循环转岗而不是简单裁员；研发预算不能砍。",
        "ren_view_en": "The colder the winter, the more you invest in R&D; recycle people through the strategic reserve instead of layoffs; the R&D budget is untouchable.",
        "divergence_note": "Ren把研发投入当作不可压缩成本——这是任何标准CFO逻辑都会反对的。讨论：他是怎么算账的？",
        "anchors": ["20180515", "20180929", "20181107", "20190521"],
    },
    {
        "id": 4,
        "theme": "决策哲学 / 灰度",
        "theme_en": "Decision philosophy / 'grayscale'",
        "q": "管理者面对复杂战略决策时，应该追求数据驱动的最优解，还是接受'方向大致正确即可'的灰度判断？",
        "q_en": "Facing complex strategic decisions, should managers pursue the data-driven optimum, or accept 'grayscale' judgment — roughly the right direction is enough?",
        "conventional_view": "数据驱动、寻找局部最优、迭代优化。",
        "conventional_view_en": "Data-driven, find local optima, iterate.",
        "ren_view": "灰度哲学——方向大致正确就行，组织要充满活力；妥协是智慧不是软弱；过度精确反而僵化。",
        "ren_view_en": "The grayscale philosophy: roughly-right direction plus a vital organization; compromise is wisdom, not weakness; over-precision breeds rigidity.",
        "divergence_note": "灰度论几乎与现代决策理论（精益、OKR、A/B测试）相反。它强调'人的活力'高于'决策准确度'。",
        "anchors": ["20090115", "20171006"],
    },
    {
        "id": 5,
        "theme": "基础研究投资",
        "theme_en": "Investing in basic research",
        "q": "公司应不应该花重金请基础研究科学家（数学家、物理学家），即使他们短期内不产出商业成果？",
        "q_en": "Should a company spend heavily on basic-research scientists (mathematicians, physicists) even if they produce no commercial results in the short run?",
        "conventional_view": "基础研究是高校和国家实验室的事，公司应聚焦应用研究和工程化。",
        "conventional_view_en": "Basic research belongs to universities and national labs; companies should focus on applied research and engineering.",
        "ren_view": "华为雇了 700+ 数学家、800+ 物理学家、120+ 化学家；允许失败、允许试错；甚至允许他们在大学半时工作。",
        "ren_view_en": "Huawei employs 700+ mathematicians, 800+ physicists, 120+ chemists; failure and trial-and-error are allowed — some even work half-time at universities.",
        "divergence_note": "学生通常知道华为投入大，但低估了'允许失败'和'科学家自治'的具体程度。这与中国大多数民企做法相反。",
        "anchors": ["20181119", "20180515", "20171120"],
    },
    {
        "id": 6,
        "theme": "人才管理 / 老员工",
        "theme_en": "Talent management / veteran employees",
        "q": "如何处理表现不再适应岗位、跟不上技术演进的老员工？是优化淘汰还是再赋能？",
        "q_en": "What to do with veteran employees who no longer fit their roles or keep up with the technology — manage them out, or re-skill and redeploy?",
        "conventional_view": "PIP（绩效改进）→ 调岗 → 离职补偿。这是硅谷公司的标准做法。",
        "conventional_view_en": "PIP → reassignment → severance; the standard Silicon Valley playbook.",
        "ren_view": "战略预备队再赋能；'烂土豆理论'——老员工不是垃圾，是没擦干净的土豆，让他们重新上前线。",
        "ren_view_en": "Re-empower them through the strategic reserve; the 'muddy potato theory' — veterans aren't garbage, they're potatoes that need washing; send them back to the front line.",
        "divergence_note": "Ren拒绝把员工当作流动资源，把'再循环'制度化。讨论：在20%员工年流失的科技行业这套机制可行吗？",
        "anchors": ["20170711", "20180426", "20150730", "20170427"],
    },
    {
        "id": 7,
        "theme": "公司治理 / 上市",
        "theme_en": "Corporate governance / going public",
        "q": "创始人应不应该让公司上市、引入外部资本，并最终交班给职业经理人？",
        "q_en": "Should a founder take the company public, bring in outside capital, and eventually hand over to professional managers?",
        "conventional_view": "IPO + 独立董事会 + 职业经理人是现代公司治理的最佳实践。",
        "conventional_view_en": "IPO + independent board + professional managers is modern governance best practice.",
        "ren_view": "华为坚决不上市——上市后短期主义会侵蚀长期投入；用全员持股+轮值CEO制；老板靠'思想权'而非股权领导。",
        "ren_view_en": "Huawei firmly refuses to list — public-market short-termism erodes long-term investment; instead: employee shareholding + rotating CEOs; the founder leads by 'the power of ideas', not equity.",
        "divergence_note": "华为是规模最大、不上市的私营公司之一，治理结构高度独特。讨论：思想权 vs 股权——谁更稳定？",
        "anchors": ["20120423", "20171120", "20130330"],
    },
    {
        "id": 8,
        "theme": "组织设计 / 大企业病",
        "theme_en": "Organization design / big-company disease",
        "q": "公司从几千人增长到几万、几十万人后，如何避免'大企业病'——决策迟缓、内耗、创新停滞？",
        "q_en": "As a company grows from thousands to hundreds of thousands of people, how do you avoid 'big-company disease' — slow decisions, infighting, stalled innovation?",
        "conventional_view": "扁平化组织、敏捷小团队、OKR、文化建设。",
        "conventional_view_en": "Flat org, agile small teams, OKRs, culture building.",
        "ren_view": "熵减——把热力学第二定律映射到组织活力；蓝军红军内部对抗机制；干部之字形循环；战略预备队人才流动。",
        "ren_view_en": "'Entropy reduction' — mapping the second law of thermodynamics onto organizational vitality; internal red-team/blue-team adversaries; zig-zag cadre rotation; talent flow through the strategic reserve.",
        "divergence_note": "Ren用物理学概念（熵）建立管理哲学，并落实到具体机制（蓝军、之字形）。这种把抽象哲学制度化的能力很罕见。",
        "anchors": ["20171219", "20161130", "20130905", "20180515", "20010424", "20010201"],
    },
    {
        "id": 9,
        "theme": "竞争战略",
        "theme_en": "Competitive strategy",
        "q": "面对成熟市场的强大对手，企业应该差异化（蓝海、避开正面竞争）还是正面对抗（资源压强、攻进核心阵地）？",
        "q_en": "Against strong incumbents in a mature market, should a firm differentiate (blue ocean, avoid head-on fights) or attack frontally (concentrate resources, storm the core position)?",
        "conventional_view": "波特差异化战略——找到蓝海、避开红海。",
        "conventional_view_en": "Porter-style differentiation — find the blue ocean, avoid the red.",
        "ren_view": "压强原则——把炮弹集中到一个城墙口，攻进无人区上甘岭；同时承认在非主战场可以购买他人技术（不是所有事都自己做）。",
        "ren_view_en": "The 'pressure principle' — concentrate every shell on one breach in the wall, push into the uncharted zone; while on non-core battlefields, buy others' technology rather than build everything.",
        "divergence_note": "Ren的策略既不是单纯差异化也不是全面进攻，而是'主战场极度集中、非主战场极度开放'。讨论：这种二元策略如何执行？",
        "anchors": ["20160227", "20180515", "20180622", "20000408", "20060510"],
    },
    {
        "id": 10,
        "theme": "顶级人才激励",
        "theme_en": "Motivating top talent",
        "q": "顶级技术人才靠什么留住？是高薪、股权、使命感，还是别的什么？",
        "q_en": "What actually retains top technical talent — pay, equity, mission, or something else?",
        "conventional_view": "综合包：竞争力薪酬+股权+文化+成长机会。这是大型科技公司的标准做法。",
        "conventional_view_en": "The standard big-tech package: competitive pay + equity + culture + growth.",
        "ren_view": "'天才少年'计划——博士起薪三五倍、不设KPI、自由选课题；同时强调'分钱机制'让奋斗者富起来，不让雷锋吃亏。",
        "ren_view_en": "The 'Genius Youth' program — 3-5x starting pay for PhDs, no KPIs, free choice of research topic; plus a wealth-sharing mechanism so strivers get rich and self-sacrifice is never punished.",
        "divergence_note": "学生通常以为'高薪+股权'是华为人才战略的核心。Ren的真实强调是'让奋斗者富'+'让他们做想做的事'，钱只是结果不是主因。",
        "anchors": ["20190910", "20190624", "20171113"],
    },
]
