---
name: book-write
description: Add knowledge to the strix-halo-maxx book (common/book, tapnext/book, moondream/book) without letting it rot. Load at the end of a round or experiment, when a result changes what a chapter says, when an idea fails, or before editing any book chapter, the negative-results index or the root README.
---

# Writing the book

The book is a manual that is kept current, not a log. Chronology lives in the commits and the rNN/ folders. A chapter says
what is true now, why, and what didn't work, with numbers and evidence.

## Choose the layer
- **common/book/**: true on this machine for any model (hardware facts, toolchain rules, measurement method, a technique's
  general mechanism). Write it so a reader who doesn't know TAPNext still gets it.
- **tapnext/book/ or moondream/book/**: shapes, layouts, op names, the specific attempts and their numbers.
- Most results have both parts. Put the mechanism and the rule in common, and the instance and its numbers in the model chapter.
  Link both ways (common → `../../tapnext/book/x.md#what-didnt-work`, model → `../../common/book/y.md#section`).
- New chapter only when no existing one fits. Add it to that book's README table with one line, and keep file names and
  section headings stable (other files link to the anchors).

## Update in place
Chapter structure: "In brief" bullets (the point first); topic sections, each giving the hypothesis, what was observed, what worked,
what didn't and the lesson; `## What didn't work`; `## Lessons`; `## Going deeper`. Model chapters also have `## What worked`.
- Edit the section the result belongs to. Change the numbers, the lead and the lessons if they moved.
- Every claim gets a number with its status (MEASURED / MODEL / EXTRAPOLATED) and an evidence link (path, round folder, commit,
  MACHINE R-id). Keep a chapter to a 5-10 minute read; push depth into links.

## Presentation: write a book for people
The reader knows ML and some systems programming, not this chip and not our shorthand. They read on the site
(MkDocs Material) or on GitHub, so use only syntax both render.
- **Open with "In brief"**: 3-5 bullets after the title, result first (what this chapter shows, the key number, the lesson).
- **Show, then tell.** Every chapter has at least one figure:
  - a ```` ```mermaid ```` diagram for dataflow (`flowchart`), ordering and handshakes (`sequenceDiagram`) or timelines (`gantt`);
  - a table or a code-block sketch for memory layouts and tile maps (exact bytes and shapes);
  - LaTeX for cost models and formulas, inline `$...$` and display `$$...$$` (e.g. $t = \max(t_\text{compute}, t_\text{DDR})$);
  - code blocks with a language tag (`asm`, `cpp`, `python`, `bash`) for real snippets, kept short and annotated.
- **Structure over prose.** One idea per paragraph, at most ~4 sentences. Lists for enumerations, tables for comparisons and
  numbers. Headings state a question or a claim ("Why one dispatch?", "The k-step is bank-bound").
- **Define before use.** Spell out an abbreviation at first use (BD = buffer descriptor, CP = command processor, AJ = average
  Jaccard). Internal ids (proposal ids like P23, switch names like ONE_SEG, build names like full12r9a) appear only as evidence,
  never as the subject of a sentence the reader must parse.
- **Evidence without clutter.** Put measurement conditions once per chapter (a "Conditions" line) or in a table column, and
  push long source lists into footnotes (`[^1]`) or a `<details><summary>Evidence</summary>` block. The claim stays readable;
  the citation stays one click away.
- **Callouts** as blockquotes: `> **Pitfall.** ...`, `> **Note.** ...`, `> **Open.** ...` (not MkDocs-only `!!!` admonitions).
- **Figures that fit the page** (the content column is ~900 px, ~88 characters of code):
    - flowcharts run `LR` and group steps into `subgraph`s; a one-column chart taller than the screen is a list, not a figure;
    - sequence diagrams: at most 5 participants with 1-2 word aliases, messages of at most ~6 words, details in a table after;
    - quote node labels; no `-beta` diagram types except `xychart-beta`, which renders on the site and GitHub;
    - code blocks at most 86 columns: put explanations on their own lines or in the prose;
    - check the render before committing: build the site, serve it on localhost and take a headless screenshot
      (`firefox --headless --no-remote --profile <scratch dir> --window-size=1300,9000 --screenshot out.png <url>`), then look at it.
- **Terms line**: a chapter that uses more than three abbreviations gets one `**Terms.**` line after "In brief".
- **"What didn't work" as a table**: idea | outcome | why | index link, with the `<!-- nr: ID -->` marker in the row.
- **Raw ids in evidence**: evidence tables, footnotes and `<details>` blocks may keep build and switch names (`full12r24z`,
  `ONE_SEG`); prose may not. A heading that other files link to keeps its wording (its anchor); new headings may be claims.
- **Footnote markers** go in prose, never inside a code block or a mermaid block (they don't render there).
- **Markdown hygiene**: a blank line before every list and table; nested lists indented by 4 spaces; relative links to `.md`
  files; link targets that exist.
- **Reference pages** (`common/MACHINE.md`) are organised by topic, never by revision: a fact lives in its topic section with a
  stable id, a tag (DOC / MEAS / INF) and its evidence. When a fact changes, edit it in place; the history is in git.

## Record every failure
- Add the entry to `common/book/negative-results.md`: id, name, project (tapnext / moondream / general), class (fundamental,
  measured-here, unfinished, revived, method), a one-line reason that says WHY it failed, and the link to the chapter section.
- Mention it in that chapter's `## What didn't work` with its one-line lesson and a link `negative-results.md#<id>`, plus
  `<!-- nr: ID -->` beside the mention.
- When a dead idea is made to work, reclassify it as revived and update the chapter text that called it dead.

## Delete or rewrite superseded text
- If a measurement overturns a claim, rewrite the claim. Don't append "update:" paragraphs. The old version is in git.
- Remove numbers that are no longer current (keep a historical number only when the lesson needs it, labelled with its round).
- Don't copy round-by-round narrative from NOTES. Summarise the mechanism and link the round.

## Keep the indexes compact
- Root `README.md`: projects, headline numbers, where things live. A few lines per project, no explanations.
- Book READMEs: one line per chapter.
- `CLAUDE.md`: entry points, safety and verification only.

## Pre-commit checklist
- [ ] Right layer; links both ways between common and the model chapter.
- [ ] Every new claim has a number, its status and an evidence link; the conditions are given for absolute timings.
- [ ] Every failure is in the negative-results index with class and reason, and in the chapter's What didn't work.
- [ ] Superseded text is rewritten or deleted, not appended to.
- [ ] Anchors and file names you link to exist; no renamed headings without fixing their links.
- [ ] READMEs still one line per item. Source docs (NOTES, RESULTS, HANDOFF, ANALYSIS, MACHINE, PROPOSALS) unchanged unless fixing an error.
- [ ] Public-repo hygiene: no personal names; no absolute home paths (repo-relative paths or `$SHM_SCRATCH`); no descriptions of private data (frames, recordings, labels); no product or other-tenant/service references; no clock times (dates are fine); neutral lab-notebook voice (no messages between agents, no "the user said").
- [ ] `python3 common/tools/bookcheck.py` from the repo root passes (links, anchors, `nr` markers, index classes and counts).
- [ ] One commit per chapter: `docs: <chapter> - <what changed>`.
