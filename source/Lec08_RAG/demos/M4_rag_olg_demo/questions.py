"""Curated classroom questions for the OLG RAG notebook.

Each question is designed to show the difference between a generic
macroeconomics answer and a grounded answer from Spear and Young's OLG paper.
The anchors are simple sanity checks: at least one should appear in the top
retrieved chunks for the question.
"""

QUESTIONS = [
    {
        "id": 1,
        "theme": "OLG vs ILA",
        "q": "How do Spear and Young distinguish overlapping-generations models from infinite-lived-agent models?",
        "conventional_view": "Textbook macro often treats OLG and ILA as alternative workhorse models for intertemporal equilibrium.",
        "paper_view": "The paper emphasizes the double infinity of agents and periods, finite lives, and the historical morphology separating OLG from ILA.",
        "discussion_note": "Use this to show that RAG retrieves the paper's historical framing, not only a generic definition.",
        "anchors": ["overlapping generations", "Infinite Lived Agents", "neoclassical growth"],
    },
    {
        "id": 2,
        "theme": "Samuelson 1958",
        "q": "Why was Samuelson 1958 important for monetary economics and the OLG tradition?",
        "conventional_view": "Samuelson introduced a canonical consumption-loans model that later became central to OLG monetary theory.",
        "paper_view": "The paper presents Samuelson as the formalized canonical foundation that clarified money, interest, and intergenerational exchange.",
        "discussion_note": "Ask students why a monetary model became a life-cycle macro workhorse.",
        "anchors": ["Samuelson", "consumption loans", "money"],
    },
    {
        "id": 3,
        "theme": "Diamond 1965",
        "q": "What did Diamond 1965 add to the OLG approach?",
        "conventional_view": "Diamond added production and capital accumulation to the OLG framework.",
        "paper_view": "The paper treats Diamond as the contribution that made Samuelson's canonical model macroeconomic by introducing production.",
        "discussion_note": "This is a clean case where the retrieved answer should include the specific historical contribution.",
        "anchors": ["Diamond", "production", "capital"],
    },
    {
        "id": 4,
        "theme": "Lucas 1972",
        "q": "What role did Lucas 1972 play in the paper's history of OLG models?",
        "conventional_view": "Lucas used an OLG-like environment in the development of monetary equilibrium and expectations-based macro.",
        "paper_view": "The paper places Lucas in the dissemination and transformation of OLG methods, especially around money and equilibrium.",
        "discussion_note": "Use this to compare a name-recognition answer with retrieved historical context.",
        "anchors": ["Lucas", "1972", "money"],
    },
    {
        "id": 5,
        "theme": "ILA Ascendance",
        "q": "Why do the authors argue ILA became dominant over OLG?",
        "conventional_view": "ILA models became dominant because they were tractable, recursive, and fit representative-agent growth and DSGE methods.",
        "paper_view": "The paper links ILA's rise to the parallel development of RBC/DSGE methods, computation, and the profession's model-selection path.",
        "discussion_note": "This is the best prompt for discussing why good models are also organizational technologies.",
        "anchors": ["Why ILA", "DSGE", "RBC", "comput"],
    },
    {
        "id": 6,
        "theme": "Stochastic OLG",
        "q": "What does the paper say about stochastic OLG models and recursive equilibrium?",
        "conventional_view": "Stochastic OLG models add uncertainty but can be difficult because the state space includes distributions across cohorts.",
        "paper_view": "The paper stresses that general multi-period stochastic OLG models raise hard equilibrium and computation problems.",
        "discussion_note": "Use this question to motivate why modern computational methods matter for OLG.",
        "anchors": ["stochastic OLG", "recursive equilibrium", "SOLG"],
    },
    {
        "id": 7,
        "theme": "Textbook Coverage",
        "q": "What does the paper say about graduate textbook coverage of OLG models?",
        "conventional_view": "Graduate textbooks usually present simple deterministic OLG models but emphasize ILA and DSGE frameworks elsewhere.",
        "paper_view": "The appendix surveys textbook coverage and shows uneven treatment of deterministic versus stochastic and multi-period OLG models.",
        "discussion_note": "This is a good retrieval audit question because the answer should come from the appendix.",
        "anchors": ["Graduate Textbook Coverage", "textbook", "deterministic OLG"],
    },
    {
        "id": 8,
        "theme": "Unsupported Query",
        "q": "What does this paper say about the December 2025 FOMC meeting?",
        "conventional_view": "A generic LLM may answer from outside knowledge or speculate.",
        "paper_view": "The retrieved paper should not support this question, so the correct RAG behavior is refusal.",
        "discussion_note": "Use this to show that grounded systems should know when not to answer.",
        "anchors": [],
        "expect_refusal": True,
    },
]
