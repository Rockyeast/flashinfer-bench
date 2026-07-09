# Repair Prompt

Use the review-onboarding-proposal skill in repair-pass mode to repair this review-only proposal.
Do not restart initial proposal generation for this run.

## Scope

- run_dir: $run_dir
- proposal_dir: $proposal_dir
- review_checklist: $review_checklist
- hf_config: $hf_config
- flashinfer_root: $flashinfer_root

## Hard Rules

- This is repair-pass, not initial proposal generation.
- Edit only proposal artifacts: proposal/candidate_targets.json, proposal/architecture.md, proposal/review_checklist.md, proposal/definitions, and proposal/definition_hints.
- Do not edit config/approved_targets.json, config/run_config.json, output/, reports/, or committed source code.
- Do not approve candidates automatically.
- Do not run Modal or any GPU job.
- Fix the proposal so the deterministic checker passes, then stop for human review.

## Definition Schema / Naming Rules

$non_fi_definition_rules

## Current Diagnostics Summary

- status: $diagnostics_status
- errors: $diagnostics_errors
- warnings: $diagnostics_warnings
- action_required: $diagnostics_action_required

$diagnostic_findings_block
$proposal_check_block
## Required Check

After editing the proposal, run:

```bash
$check_cmd
```

Repeat proposal edits only until `ready for human review: True`, then stop.
