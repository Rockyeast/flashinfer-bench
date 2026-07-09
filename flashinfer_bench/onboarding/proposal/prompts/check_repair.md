# Check Repair Prompt

Use the review-onboarding-proposal skill in repair-pass mode to repair this review-only proposal.
This is a static proposal repair pass before any runtime run.

## Scope

- proposal_dir: $proposal_dir
- proposal_check: $proposal_check
- review_checklist: $review_checklist
- hf_config: $hf_config
- flashinfer_root: $flashinfer_root

## Hard Rules

- Edit only proposal artifacts: proposal/candidate_targets.json, proposal/architecture.md, proposal/review_checklist.md, proposal/definitions, and proposal/definition_hints.
- Do not edit config/approved_targets.json, config/run_config.json, output/, reports/, or committed source code.
- Do not approve candidates automatically.
- Do not run Modal or any GPU job.
- Fix the proposal so the deterministic checker passes, then stop for human review.
- If merge_report.json or review_checklist.md reports merge conflicts, read the listed source proposal dirs before editing the merged proposal.
- When resolving merge conflicts, add or update a `## Manual Conflict Resolution` section in review_checklist.md explaining which source draft was kept and why.

## Definition Schema / Naming Rules

$non_fi_definition_rules

## Current Proposal Check Summary

- ready_for_human_review: $ready_for_human_review
- errors: $errors
- warnings: $warnings

$findings_block
## Required Check

After editing the proposal, run:

```bash
$check_cmd
```

Repeat proposal edits only until `ready for human review: True`, then stop.
