---
name: git-pr-workflow
description: End-to-end Git development workflow for inbed-pose-mlops. Guides branch creation with standardized names, atomic Conventional Commits, local CI verification, context updates, and Pull Request creation via gh CLI using temporary markdown body files.
---

# Inbed-Pose-MLOps Git & Pull Request Workflow

Use this skill when developing any feature, bugfix, refactor, or maintenance task on **inbed-pose-mlops**.

---

## Process

### Step 1: Branch Creation
**Strict Invariant**: NEVER modify, create, or stage files on `main`. Before writing any code, check your active branch (`git branch --show-current`). If on `main`, pull latest and branch out immediately:

```bash
git checkout main
git pull origin main
git checkout -b <branch-name>
```

#### Branch Naming Guidelines
- Features: `feat/<short-description>`
- Fixes: `fix/<short-description>`
- Refactors: `refactor/<short-description>`
- Chores / Docs: `chore/<short-description>` or `docs/<short-description>`

---

### Step 2: Implement & Commit with Conventional Commits
Write atomic commits as you complete discrete, logical units of work:

```bash
git add <files>
git commit -m "<type>[optional scope]: <description>"
```

- **Allowed types**: `feat`, `fix`, `refactor`, `style`, `test`, `docs`, `chore`
- **Scope**: Noun describing affected area (`data`, `models`, `training`, `metrics`, `hpo`, `ci`, `agents`, `docs`).
- **Linguistic rule**: Imperative, present tense (e.g., `feat(models): add resnet pose architecture`)
- **Atomic Checklist**:
  1. *Single Purpose*: Diff solves exactly one problem or feature.
  2. *State Stability*: Codebase compiles and tests pass at this commit checkpoint.
  3. *Diff Isolation*: Logic changes are isolated from formatting/style changes.

---

### Step 3: Mandatory Local CI Verification
Run the exact equivalents of the repository's GitHub Actions CI workflows:

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

*Ensure all checks pass with 0 errors before proceeding.*

---

### Step 4: Check & Synchronize `.agents/` Context
Evaluate if your changes triggered any self-maintenance updates:
- Added new model or data loader? -> update [`.agents/context/architecture_overview.md`](../../context/architecture_overview.md)
- Completed roadmap milestones? -> check off in [`.agents/context/project_context.md`](../../context/project_context.md)

---

### Step 5: Push Branch & Create PR via Temporary Body File
Push the branch to GitHub:

```bash
git push -u origin <branch-name>
```

Create PR using `.tmp_pr_body.md`:
1. Write PR body to `.tmp_pr_body.md`.
2. Execute `gh pr create --title "<type>(<scope>): <short description>" --body-file .tmp_pr_body.md`.
3. Delete `.tmp_pr_body.md`.
4. Monitor GitHub Actions CI:
   ```bash
   gh run list --limit 1
   gh run watch <run-id> --exit-status
   ```
