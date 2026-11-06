# Git & Pull Request Standards

> [!IMPORTANT]
> **Trigger Paths**: Any workspace Git source control action (branching, committing, pushing, PR/issue creation).
> **When to Read**: MUST be read before staging changes, formatting Conventional Commits, running pre-push CI verification, or opening GitHub PRs.

This document governs all version control, branching, committing, and pull request operations across `inbed-pose-mlops`.

---

## 1. Branching Strategy & Pre-Implementation Invariant

### Strict Pre-Implementation Invariant
Never write, modify, or stage code directly on the `main` branch. Before making ANY changes to the codebase, check `git branch --show-current`. If on `main`, pull latest changes and create a dedicated branch first:

```bash
git checkout main
git pull origin main
git checkout -b <branch-name>
```

### Branch Naming Conventions
- **New Features**: `feat/<short-description>`
- **Bug Fixes**: `fix/<short-description>`
- **Refactoring**: `refactor/<short-description>`
- **Documentation**: `docs/<short-description>`
- **Chores / Maintenance / CI**: `chore/<short-description>`

*Examples:* `feat/slp-data-pipeline`, `feat/optuna-hpo-pipeline`, `chore/repo-scaffolding-and-ci`

---

## 2. Conventional Commits & Atomic Commit Protocol

All commit messages must strictly adhere to the Conventional Commits specification:

```
<type>[optional scope]: <description>
```

### Allowed Types
- `feat`: A new feature or capability is introduced.
- `fix`: A bug is resolved.
- `refactor`: Code rewriting without changing external behavior or fixing a bug.
- `style`: Formatting changes that do not alter code logic.
- `test`: Adding or updating tests.
- `docs`: Documentation-only changes.
- `chore`: Updating build tasks, configurations, or CI workflows.

### Scope Guidelines
- Scopes must be single, lowercase nouns: `data`, `models`, `training`, `metrics`, `hpo`, `ci`, `agents`, `docs`.
- Avoid file paths as scopes.

### Granularity Protocol (Atomic Commits Checklist)
Before staging and committing changes, evaluate this internal 3-point checklist:
1. **Single Purpose**: Does this commit solve exactly one logical problem or implement one discrete feature?
2. **State Stability**: If this individual commit is checked out independently, does the codebase build cleanly and do all tests pass?
3. **Diff Isolation**: Are logic changes kept separated from formatting, styling, or unrelated refactoring changes?

---

## 3. Mandatory Pre-Push Verification

Before executing `git push`, you MUST run the local equivalents of the GitHub Actions CI pipeline:

```bash
# 1. Lint with Ruff
python -m ruff check .

# 2. Check formatting with Ruff
python -m ruff format --check .

# 3. Type check with Mypy
python -m mypy src scripts --config-file mypy.ini

# 4. Run tests with pytest
python -m pytest tests/ -v
```

*Ensure all checks pass with 0 errors before pushing.*

---

## 4. Pull Request Protocol via Temporary Body File

To avoid PowerShell/terminal quotation and escaping errors with multiline markdown strings, **never pass inline multi-line markdown strings** to `gh`. Always use a temporary file:

1. **Write PR body to `.tmp_pr_body.md`**:
```markdown
## Summary
- Bullet point overview of high-level changes.
- Key outcomes and user-facing/system impacts.

## Motivation
- Why this change is necessary.

## Key Changes
- Specific technical modifications made.

## Verification
- [x] `ruff check .` passed with 0 errors
- [x] `ruff format --check .` passed cleanly
- [x] `mypy src scripts` passed with 0 errors
- [x] `pytest tests/` passed
```

2. **Execute `gh pr create`**:
```bash
gh pr create --title "<type>(<scope>): <short description>" --body-file .tmp_pr_body.md
```

3. **Delete `.tmp_pr_body.md`**:
```bash
Remove-Item .tmp_pr_body.md
```

4. **Monitor CI**:
```bash
gh run list --limit 1
gh run watch <run-id> --exit-status
```
