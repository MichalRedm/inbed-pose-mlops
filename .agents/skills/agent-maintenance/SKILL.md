---
name: agent-maintenance
description: Keep .agents/ context and rules synchronized with project evolution and completed PRs.
---

# Agent Context Self-Maintenance Skill

Use this skill at the conclusion of a feature, refactor, or PR to ensure documentation does not drift from implementation.

---

## Maintenance Checklist

1. **Context Update**:
   - Check if new modules or scripts were added to `src/` or `scripts/`.
   - Update `.agents/context/architecture_overview.md` to document class signatures and data flows.
2. **Roadmap Tracking**:
   - Update milestones in `.agents/context/project_context.md` from `[ ]` to `[x]`.
3. **Rule Calibration**:
   - If new linting rules, typing policies, or testing patterns were introduced, update `.agents/rules/code_quality_and_ci_standards.md` or `.agents/rules/ml_training_standards.md`.
