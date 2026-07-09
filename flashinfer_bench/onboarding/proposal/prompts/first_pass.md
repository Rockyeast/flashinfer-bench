# $model_name Initial Proposal Draft

Use the `review-onboarding-proposal` skill to generate a review-only proposal for:

```text
$model_name
```

Working directory:

```text
.
```

Inputs:

```text
HF config: $hf_config_path
SGLang source root: $sglang_root
SGLang model implementation hints:
$sglang_model_hints
FlashInfer source root: $flashinfer_root
sgl-cookbook root: $cookbook_root
diagnostics: omit; initial draft
run dir: $run_dir
```

Outputs:

```text
$architecture_path
$candidate_targets_path
$review_checklist_path
$definitions_path only for non-FI definition_source=agent drafts
$definition_hints_path only for non-FI definition_source=agent drafts
$run_config_path
```

$runtime_guidance
Strictly follow `.claude/skills/review-onboarding-proposal/SKILL.md`.

Do not apply or approve anything. Do not write `config/approved_targets.json`.
Do not write runtime `output/definitions`, `output/workloads`, or `output/blob`.
Do not run Modal, collect, validate, or commit.

Definition schema/naming rules for non-FI `definition_source=agent` drafts:

$non_fi_definition_rules

After writing the proposal, run the deterministic check once and leave the result in the proposal status files:

```bash
$check_cmd
```

Do not run a self-repair loop here. If this first draft is still `FIX_REQUIRED`, leave the
check result in `proposal_check.json` / `review_checklist.md`; the merged proposal will be repaired
later through `check-repair-loop`.
