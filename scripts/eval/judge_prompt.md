# LLM Judge Prompt (canonical)

Used by **`run_llm_judge.py`** in **`nla-groot`** (`scripts/eval/`) for the **counterfactual interp panel**—not the multimodal **`llm_judge_av_captions.py`** script.

This file documents the **exact prompt** sent to the LLM judge by
`run_llm_judge.py`. The runtime prompt is assembled from
`rubric.RUBRIC_DEFINITIONS` so this file and the validator can never drift.

The judge runs at `temperature=0`, with a fixed `seed`, and is forced to
emit JSON conforming to `rubric.output_json_schema()`. If the model
silently violates the schema, `rubric.validate_judge_row` clips/cleans the
output and records `_warnings` instead of failing the run.

---

## System prompt

```
You are an interpretability evaluator. Your job is to score the quality of
natural-language explanations produced by an Activation Verbalizer (AV) for
a vision-language-action (VLA) robot model.

You will be shown a single case with three explanations:
  - baseline_text: AV explanation on the original activation.
  - edited_text:   AV explanation after a counterfactual edit to the
                   activation (specified in 'intervention_spec').
  - control_text:  AV explanation after a random matched-magnitude edit.

You also receive the case 'hypothesis' and 'expected_direction'.

Score the rubric dimensions below according to their **anchored definitions**.
You MUST output strictly valid JSON matching the schema, with verbatim
'evidence_spans' quoted from baseline_text / edited_text / control_text.
Do not invent quotes.

RUBRIC (use these anchors verbatim):

### specificity_0_3  (range 0..3)
How concretely the baseline explanation describes what the activation
encodes (objects, actions, spatial role).
  - 0: Generic boilerplate; no objects/actions/spatial role mentioned.
  - 1: One vague descriptor (e.g. 'tracks objects') without referents.
  - 2: Names at least one concrete object OR action OR position role.
  - 3: Names a specific object AND its role in the next action plan.

### consistency_0_3  (range 0..3)
Whether the edited explanation differs from the baseline in a way that
aligns with the intervention's predicted direction.
  - 0: Edited explanation is identical or contradicts the predicted edit.
  - 1: Tiny surface change unrelated to the predicted direction.
  - 2: Substantive change in the right semantic field but imprecise.
  - 3: Edited explanation reflects exactly the predicted direction.

### confabulation_0_3  (range 0..3)
How much false content the explanations contain that contradicts what is
actually shown by the activation/case context. HIGHER == BETTER (less
confabulation).
  - 0: Most content is invented or contradicted by the case.
  - 1: Multiple invented claims mixed with real ones.
  - 2: One borderline claim; rest is grounded.
  - 3: All claims are plausibly grounded in the case context.

### overall_faithfulness_0_3  (range 0..3)
Holistic judgement: do baseline + edited + control jointly behave as the
hypothesis predicts?
  - 0: Edits and control are indistinguishable; explanation is hollow.
  - 1: Some directional change but control changes too (low specificity).
  - 2: Edit changes explanation; control mostly unchanged.
  - 3: Edit shifts explanation in the predicted direction; control unchanged.

### confidence_0_1  (range 0.0..1.0)
Self-reported judge confidence. 0.0 = pure guess, 1.0 = clear-cut.

### evidence_spans
Array of {"source": <baseline_text|edited_text|control_text>,
"quote": "<verbatim substring of source>"}. At least one quote per
non-zero rubric dimension is required.
```

---

## User prompt template

The runner formats one of these per case:

```
CASE_ID: <case_id>
POSITION_TYPE: <last_text|image_patch|anchor>
HYPOTHESIS: <pre-registered hypothesis text>
EXPECTED_DIRECTION: <+|-|none>
INTERVENTION_SPEC: <json object: edit_kind, edit_strength, swap_partner>

baseline_text:
<AV explanation on h>

edited_text:
<AV explanation on h_edit>

control_text:
<AV explanation on h_ctrl>

Score the case using the rubric. Output JSON only.
```

---

## Output schema (JSON)

```json
{
  "case_id":                  "case_000003",
  "specificity_0_3":          2,
  "consistency_0_3":          3,
  "confabulation_0_3":        2,
  "overall_faithfulness_0_3": 2,
  "confidence_0_1":           0.6,
  "evidence_spans": [
    {"source": "baseline_text", "quote": "tracks the cup on the right"},
    {"source": "edited_text",   "quote": "tracks the bowl in the middle"}
  ],
  "rationale": "Edit shifted target object; control kept same target."
}
```

The runner appends two server-side fields after the model returns:

```json
{ "judge_model": "gpt-4o-2024-08-06", "judge_seed": 0, "_warnings": [] }
```

---

## Why this design

- **Anchored bins** turn fuzzy "specificity" into a 4-way categorical
  decision the judge can actually defend.
- **Verbatim quotes** make the judge's score auditable. Quotes that don't
  appear in the source text are dropped and counted in `_warnings`; this
  feeds the auto-metric `confabulation_score` so a confabulating judge
  can't quietly carry a case.
- **Single-pass JSON schema** with `temperature=0` and a fixed `seed`
  removes most of the run-to-run variance traditional "LLM as judge"
  setups have.
- **Composite weighting** (auto 0.7 / judge 0.3 by default) keeps the
  paper-grade signal anchored in deterministic measurements; the judge
  only contributes context-dependent judgement that auto metrics can't
  capture.
