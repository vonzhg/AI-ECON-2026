# Cross-References: How Concepts Connect Across Lectures

> This article traces three threads that run through Lectures 7-9, showing how
> ideas introduced in one lecture become building blocks in the next.

## Thread 1: Embeddings

| Lecture | Role of Embeddings |
|---------|-------------------|
| **Lec 7** | **Token embeddings** are the first layer of a transformer — they map discrete token IDs to continuous vectors. Internal to the model; the user never touches them. |
| **Lec 8** | **Document/chunk embeddings** are the foundation of RAG retrieval. Same idea (text -> vector), but applied at larger granularity (chunks, not tokens) and used for similarity search rather than as model input. |
| **Lec 9** | Agentic systems use embeddings implicitly — the agent decides *when* and *what* to embed/retrieve as part of a multi-step plan. |

See: [Embeddings](../concepts/embeddings.md)

## Thread 2: The Scaling Problem

| Lecture | What Scales, What Breaks |
|---------|------------------------|
| **Lec 7** | Context windows limit how much text an LLM can process at once (~200K tokens). Knowledge is frozen at training time. |
| **Lec 8** | RAG is the solution — instead of cramming everything into the context window, index externally and retrieve only what's relevant. The personal wiki is an intermediate step: it works at ~400K words but breaks at corpus scale. |
| **Lec 9** | Agentic RAG manages the retrieval process itself — deciding what to search for, evaluating whether results are sufficient, and iterating. |

## Thread 3: Knowledge Representation

| Type | Where | Properties |
|------|-------|-----------|
| **Parametric** (Lec 7) | Model weights | Fixed after training. Fast at inference. No attribution. |
| **Non-parametric** (Lec 8, RAG) | External documents indexed in vector DB | Updatable. Provides attribution. Scales to millions of docs. |
| **Curated** (Lec 8, Wiki) | Structured markdown maintained by LLM | Human-readable. Cross-linked. Works at small scale. |
| **Agentic** (Lec 9) | Built on-the-fly by the agent | Dynamic. Task-specific. Combines retrieval with tool use. |

## Thread 4: From Static to Dynamic

The arc of Lectures 7-9 is a progression from static to dynamic systems:

1. **Lec 7 (LLM)**: A frozen model that processes one prompt at a time.
2. **Lec 8 (RAG)**: A static pipeline — one query in, one retrieve-then-generate
   cycle, one answer out. But the knowledge store is updatable.
3. **Lec 9 (Agentic AI)**: A dynamic system — the agent plans, retrieves,
   reasons, and loops. It decides *when* to use RAG, *what* to retrieve, and
   *whether* to try again with a refined query.

## Economic Research Implications

For a macro researcher building an empirical pipeline:

- **Start with a personal wiki** (Lec 8): Collect your FOMC statements, key
  papers, and data descriptions. Let an LLM compile them into a searchable wiki.
  Good enough for a single research project.

- **Graduate to RAG** (Lec 8): When your corpus outgrows the context window —
  say, 30 years of FOMC + speeches + Beige Books — build a proper vector index.

- **Add agency** (Lec 9): When your research question requires multi-step
  investigation — "compare the evolution of forward guidance language across
  the Fed, ECB, and BoJ from 2010 to 2025" — an agentic system can plan the
  retrieval strategy, execute it, and synthesize the results.
