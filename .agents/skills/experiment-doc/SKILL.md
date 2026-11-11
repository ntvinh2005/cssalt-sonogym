---
name: experiment-doc
description: Required for writing or editing an experiment log, research doc, or results write-up in any repo. Defines a mandatory section template, a reproducible-setup standard, and notation rules. Load before the first edit to such a file, not after.
---

# Experiment docs

**Audience: the author, now and in six months.** Assume full domain context — do not re-explain the
field's standard method. Assume **zero memory of the run** — every result must be reproducible from
the doc alone, without opening the code. That tension decides every call below: terse on *why*,
exhaustive on *what was run*.

## Before the first edit in a repo you have not written docs for

Establish these four things, from the repo or by asking, and do not guess:

1. **Where experiment docs live** and whether an existing one sets a house style to match.
2. **How a step is run** — the actual entry point (`make`, `sbatch`, `python -m`, a notebook, a CI
   job). The Setup section must quote the real command.
3. **What identifies a run** — a run id, a checkpoint, a commit hash, a config file, a sweep id.
   Whatever it is, it goes in every Setup table.
4. **Whether the project has a pass/fail criterion convention.** If none exists, ask what threshold
   the work is being held to before writing a Results section without one.

## The template is mandatory

Every experiment step is a `## N. Step name — the question this step answers`, and contains these
subsections, **in this order, all of them, even when one is a single line**:

```markdown
## N. Step name — the question this step answers

`path/to/module`, run by `<entry point>`. One sentence on what it computes.

**Criterion.** The threshold this must clear and where the threshold came from. If there is no
threshold, write "No absolute threshold" and name what the result is compared against instead.
Never omit this subsection.

### N.0 Quantities        <- only in the first section that introduces a symbol; omit elsewhere

| symbol | definition | source |
|---|---|---|

### N.1 Method

The rule, the maths, and only those design decisions that change a number.

$$ formula $$

- decision -> consequence
- decision -> consequence

### N.2 Setup

| what | value | spec |
|---|---|---|
| run / checkpoint | `<run id>`, `<checkpoint>` | §N |
| inputs | `<path to the artefact>` | §N |
| filters | `<threshold>`, `<flag value>` | §N |

```
<the exact invocation>
# <the resolved command, every default spelled out>
```

Then the sample-accounting table whenever anything was dropped:

| | value |
|---|---|
| attempted | n |
| dropped, <reason> | n |
| **final** | **n** |

### N.3 Results

| ... | ... |

**Verdict: pass/fail**, with the comparison in the same sentence.

### N.4 Reading it

- bullets only, one fact each

### N.5 Files

| file | contents |
|---|---|
```

If a subsection would be empty, write one line saying so (`Nothing was dropped.`). Do not delete the
heading — a missing heading reads as an oversight, an explicit line reads as a fact.

Whole-document top matter:

```markdown
# Title

Two sentences: what this is, what is in scope.

## Result

**One-sentence verdict for the whole project, in bold.**

| step | what was measured | criterion | measured | verdict |

One or two sentences on what the pattern of verdicts means.
```

## Extending the template — ask, do not invent

**When a step needs a section the template does not define, ask the user what to add before adding
it.** Propose the section by name, say what it would contain, and say where it goes in the order. Do
not silently invent structure — inconsistent structure across steps is exactly what makes a log
unreadable, and the fix is a decision the author makes once and then applies everywhere.

The same applies in reverse: if a step genuinely cannot have a Criterion (exploratory, no hypothesis
yet), ask whether to mark it `**Exploratory — no criterion**` rather than dropping the heading.

## Hard rules

- **Every result section states a verdict in bold.** A number with no criterion is not a result.
- **No debugging narrative in the body.** How a defect was found goes in an appendix as
  `| defect | effect |`, one row. In the body: at most one clause plus `(§N)`.
- **Every quantity defined once**, in a Quantities table, never re-derived.
- **Every table names the setting that produced it** — which arm, which config, which checkpoint.
- **Bullets over prose.** A paragraph is for an argument; a bullet is for a fact.
- **Cut every non-load-bearing word.** No "it is worth noting", "as mentioned above", or restating
  the previous sentence.
- **Cross-reference by section number**, never "see above".

## The reproducibility standard

The Setup section fails if a reader must open the code to re-run the step. It must carry:

- run and checkpoint identity — never just the model or dataset name
- the input artefact's **path**, not a description of it
- every filter and threshold, with a `§` pointer to where it is justified
- the function or metric by name, whenever more than one exists in the codebase
- the resolved command, with defaults written out — not just the wrapper script
- sample sizes: attempted, what was dropped and why, final
- for anything with arms: explicit definitions of each arm, plus a table of what the control is
  matched on **and what it is not**

## Notation

- Define symbols before use; never introduce one in a results table.
- `$$ ... $$` for anything a reader would otherwise reconstruct; backticks for one-symbol references.
- State units, and state explicitly when two columns are **not** comparable.
- Counts as `k / n (p%)`, never a bare percentage.
- Statistics with the test named: `Fisher exact two-sided p = 0.015`, `paired t, t(19) = 2.3`.
- Numbers keep the precision they were measured at; do not round a p-value into significance.

## Updating an existing doc

- Superseded numbers: delete them, or keep them only if a comparison needs them — and then say in
  the same sentence that they are superseded and why.
- `grep` for `§N` before renumbering sections.
- When a result changes, grep the whole doc for the old number: the top-matter verdict table, risk
  tables and summary lists all quote results and all go stale silently.
- Preserve an existing doc's house style where it conflicts with this template only in surface
  matters (heading case, table style). Structure and the mandatory subsections do not bend.
