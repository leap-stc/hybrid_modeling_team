# CLAUDE.md

Guidance for Claude Code (or any assistant) working in this repository.

## What this repo is

`hybrid_modeling_team` is a **hub repo**, not a single application. It aggregates
independent sub-projects — each developing an ML-based parameterization (or
supporting analysis) for CESM — into one place for team visibility and shared
documentation. It is not a monorepo in the usual sense: sub-projects do not
share a build system, dependency file, or Python environment.

Each sub-project directory (e.g. `Convection_trigger_function/`) is pulled in
via `git subtree`, not a submodule. That means:

- The code is physically present in this repo (no `.gitmodules`, no separate
  clone step needed to see the files).
- Each sub-project still has its own upstream GitHub repo
  (`github.com/<owner>/<sub-project>.git`) that it was squashed in from.
- History for a sub-project's own repo is *not* fully preserved here — only a
  single squashed commit per subtree merge.

## Working across sub-projects

- **Don't assume a shared environment.** Each sub-project may use a different
  conda env, Python version, or set of dependencies (XGBoost, PyTorch,
  ESMF/CESM tooling, etc.). Look for an env file or README inside the
  sub-project folder before running anything.
- **Don't hoist code to the repo root.** Shared utilities that genuinely apply
  across sub-projects belong in a clearly named `shared/` or `common/`
  directory (create one if/when it's actually needed) — not scattered at the
  top level.
- **Follow the existing README template** when adding or editing a
  sub-project README: a `MOTIVATION` section (what physical process is
  misrepresented, what dataset/tool enables improvement) and an `APPROACH`
  section (method, evaluation strategy). See
  [`Convection_trigger_function/README.md`](./Convection_trigger_function/README.md)
  for the pattern.
- For the shared conceptual workflow every sub-project follows (data → model →
  offline eval → CESM coupling → online eval), see [`Start Here.md`](./Start%20Here.md).

## Updating a sub-project (git subtree)

Pull in upstream changes to an existing sub-project:

```bash
git subtree pull --prefix=<Sub_Project_Dir> <upstream-url> main --squash
```

Push local changes made under a sub-project's folder back to its own repo:

```bash
git subtree push --prefix=<Sub_Project_Dir> <upstream-url> main
```

Add a brand-new sub-project the same way it's been done before:

```bash
git subtree add --prefix=<New_Sub_Project_Dir> <upstream-url> main --squash
```

`<upstream-url>` should point at the sub-project's own GitHub repo (typically
under a team member's account or the `leap-stc` org), not this hub repo.

## Environment notes (NCAR Derecho/Casper)

This repo is generally worked on from `/glade/work/<user>/...` on Derecho or
Casper. Tools like `gh` and `git-lfs` are not on `PATH` by default and need to
be loaded:

```bash
module load gh
```

Large data (TOOCAN tracks, reanalysis, model output) should live on
`/glade/campaign` or `/glade/derecho/scratch`, **not** be committed to git —
sub-project READMEs should document where their input data lives instead.

## When adding documentation

- Keep the root `README.md` sub-project table up to date when a new
  sub-project is added.
- Prefer updating this file and `Start Here.md` over creating new top-level
  docs — keep the "front door" small.
