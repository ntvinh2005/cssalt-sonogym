---
name: communication
description: Required for any text written for the user — chat replies, docs, code comments, commit messages, PR descriptions. Enforces one definition file (notation.md) for every symbol, method and metric, requires the user's approval before any new term is introduced, and fixes the message shape (claim first, explanation only where non-obvious). Load before explaining a method, reporting a result, writing a doc/commit/PR, or naming anything.
---

# Communication

Audience: an expert in the field. Assume domain knowledge. Never assume a name you made up is known.

## 1. `notation.md` is the only place terms are defined

Every symbol, method, metric and named object used in chat, docs, comments, commits or PRs has exactly
one entry in `notation.md`. Standard mathematics (sets, norms, $\mathbb{E}$, $\operatorname{tr}$, ...) needs no
entry. Methods and metrics from the literature do need one, with the citation as source.

**Location.** Look for an existing file (`notation.md`, `NOTATION.md`, `docs/notation.md`). If there is
none, ask the user where to create it. Do not guess.

**Entry format:**

```markdown
### <symbol or name>
- **Kind:** symbol | method | metric | object
- **Definition:** formal, using only standard mathematics and earlier entries. $$ ... $$
- **Intuition:** what it measures or does, and why it exists. At most two sentences.
- **Source:** citation, `file:function`, or "approved by user, <date>".
```

**One symbol, one meaning.** Before adding or reusing a symbol, `grep` `notation.md` and the repo's docs
for it. On a collision, ask the user. Do not overload or silently rename.

**Changing a definition** needs approval (§2). After approval, `grep` every use and update it.

**Experiment docs.** A Quantities table lists the symbols a step uses and links to their `notation.md`
entries. It never restates a definition.

## 2. Ask before introducing any new term

"New term" means anything not in `notation.md` and not standard mathematics. That includes abbreviations,
a new name for an existing thing, a name for a combination of existing things (e.g. calling a filtered
set "clean features"), and a variant of an existing metric (changing the normalization makes a new metric).
Local variable names in code are exempt. Any name that appears in prose, comments, docstrings, commits
or PRs is not.

Before using the term anywhere, stop and ask:

```
Proposed term: `<name>`
Definition: <formal, in notation.md terms>
Intuition: <one line>
Needed because: <what existing entries cannot express>
Alternative: <the plain description you would use instead>
```

Add the entry only after an explicit yes. Until then, use the plain description.

## 3. Message shape

1. **Claim, first sentence.** State what this is or does and why it matters for the user's goal. If the
   claim is unverified, the same sentence says so.
2. **Explanation, only for steps an expert would not see immediately.** Give why it holds, then what it
   implies. State the formal version first (definition or derivation in `notation.md` terms), then the
   intuition.
3. **Nothing else.** Banned: preamble, restating the question, recapping earlier messages, filler ("it is
   worth noting", "interestingly"), closing summaries that repeat the body, unrequested options.

| output | shape |
|---|---|
| chat reply | claim → non-obvious why/implication |
| doc | per the doc skill; terms per §1–2 |
| commit | subject = claim (imperative); body = why, only if non-obvious |
| PR | first line = claim; then why/implication; then a list of `notation.md` entries added or changed |
| code comment | why, not what; `notation.md` names only |