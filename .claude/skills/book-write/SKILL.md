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
Chapter structure: a lead paragraph (the point first); topic sections, each giving the hypothesis, what was observed, what worked,
what didn't and the lesson; `## What didn't work`; `## Lessons`; `## Going deeper`. Model chapters also have `## What worked`.
- Edit the section the result belongs to. Change the numbers, the lead and the lessons if they moved.
- Every claim gets a number with its status (MEASURED / MODEL / EXTRAPOLATED) and an evidence link (path, round folder, commit,
  MACHINE R-id). Keep a chapter to a 5-10 minute read; push depth into links.

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
