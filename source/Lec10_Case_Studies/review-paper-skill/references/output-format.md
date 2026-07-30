# Output Format — refine.ink Style

The final review MUST match this exact format. Study the structure carefully.

## Template

```markdown
# [Paper Title]

**Date**: [MM/DD/YYYY]
**Domain**: [domain/subdomain]
**Taxonomy**: [academic/type]
**Filter**: Active comments

---

## Overall Feedback

Here are some overall reactions to the document.

**[Issue Title 1]**

[Issue body paragraph — 4-8 sentences explaining the concern, its implications, and remediation. Reference specific sections, equations, theorems. End with a concrete suggestion.]

**[Issue Title 2]**

[Issue body paragraph...]

[... repeat for all issues ...]

**Recommendation**: [accept/minor revision/major revision/reject — with 2-3 sentence justification]

**Key revision targets**:
1. [Specific revision target]
2. [Specific revision target]
...

**Status**: [Pending]

---

## Detailed Comments ([N])

### 1. [Comment title]

**Status**: [Pending]

**Quote**:
> [Exact verbatim text from the paper — copy character-for-character]

**Feedback**:
[Explanation of issue + constructive remediation guidance. 3-8 sentences. End with a concrete fix.]

---

### 2. [Comment title]

**Status**: [Pending]

**Quote**:
> [Verbatim quote]

**Feedback**:
[Feedback text]

---

[... repeat for all comments ...]
```

## Format Rules

1. The header block uses `**Date**:`, `**Domain**:`, `**Taxonomy**:`, `**Filter**: Active comments`
2. Overall Feedback section starts with "Here are some overall reactions to the document."
3. Issue titles are bold `**Title**` — NOT numbered
4. Each issue is a substantive body paragraph (4-8 sentences)
5. After all issues: recommendation + revision targets + `**Status**: [Pending]`
6. Detailed Comments section header includes the count: `## Detailed Comments (N)`
7. Each comment: `### N. Title`, `**Status**: [Pending]`, `**Quote**:`, `> quoted text`, `**Feedback**:`
8. Comments separated by `---`
9. Quotes use `>` blockquote prefix — every line of multi-line quotes gets `> `
10. Date format: MM/DD/YYYY (e.g., 04/10/2026)

## Reference Example

See `data/refine_examples/r3d/feedback-regression-discontinuity-design-with-distribution--2026-03-03.md` for the gold standard output format.
