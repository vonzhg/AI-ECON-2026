# Agent Design Canvas

Companion to **T3a, "Build an Agent From Scratch,"** and the T4 Design Lab. One page,
eight cells. Fill it in *before* you open an editor — if a cell is empty, the
agent is not designed yet, it is a wish.

Print this page, or copy it into your project as
`docs/agent_canvas_<name>.md` and commit it next to the agent file it produced.

Agent name: `________________________`   Date: `__________`

---

### 1. Produces / fails when

One sentence, in this shape: *it produces ⟨one artifact⟩, and I will know it
failed when ⟨something I can observe⟩.*

>
>

### 2. Job description

Who it is, what it checks, and the one thing it must refuse to do. This becomes
the prose under the front matter — the `system` argument of every request the
agent makes.

>
>
>

### 3. Tools

The allowlist, and one line on why nothing more. Remember that this is read two
ways: what it can *do* to your project, and what it can *learn*.

>

### 4. Brief in

The files and facts that enter `messages`. Then, separately, what you are
deliberately keeping out — your hypothesis, your hunch about the bug, the dead
ends you already tried.

> In:
>
> Out:

### 5. Returns

The shape of the report. One line per finding, with what in it? And how must it
say "I found nothing in this category"?

>
>

### 6. Budget / stop rule

Model, turn ceiling, and the conditions under which it must stop and ask rather
than continue.

>

### 7. Planted bug it must catch

A defect you will insert on purpose, whose answer you already know. Also name
one *adjacent* defect it must stay silent about, because that one belongs to a
different agent.

> Must catch:
>
> Must stay silent about:

### 8. Path in the repo

Where the file lives, and who else on the team gets it when they pull.

>

---

## Then debug the English

Two prompts, each in a fresh session:

1. One the agent **should** handle. Does it get dispatched?
2. One adjacent it should **not** handle. Does it get dispatched anyway?

Too broad is the more common failure — the agent gets pulled into work it is
bad at. Narrow the `description` until both cases behave. You are debugging
English, which is the whole skill.

## Acceptance test: the detection matrix

Plant one defect per background and dispatch all three specialists. The
diagonal must fire; the off-diagonal must stay quiet; and on a line that is
**correct**, all three must say nothing.

| Planted defect (or control) | `tooling-agent` | `math-agent` | `econ-agent` |
|---|---|---|---|
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
| *control:* a line that is correct | silent | silent | silent |

An empty column means the `description` is too narrow. A full column means it
is too broad, and you have three copies of one generalist. An agent that finds
something in every category is not being careful — it is being agreeable.
