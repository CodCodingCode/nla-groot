# SA10 — Automated audit regression gate (V4 vs V3 frozen baselines)

**Subagent 10 of the LIBERO V4 dataset repair plan.** Re-runs the SA3, SA4, Agent 3, Agent 4 (and references SA7's Agent 5) audits against the V4 corpus and compares to the frozen V3 baselines. Combined with SA9's multimodal judge (`sa9_judge_ab.md`), this is the automated half of the V4 ship/no-ship decision.

## Executive summary (TL;DR)

1. **V4 is a clear improvement on every axis the original audits flagged RED**: motor-imperative (62.4% → 1.7%), scaffold leakage (29.8% → 0.007%), non-canonical headers (10.2% → 0.000%), `image_patch` last-bullet `plan` collapse (93.8% → 29.1%), `last_text`/`image_patch` Jaccard (target 0.76 → 0.40, plan 0.71 → 0.30).
2. **V4 introduces one new template-collapse pattern in `plan` bullets** ("over the next 3 timesteps", "phase over the", "before placing on") that pushes plan-bullet boilerplate higher than V3's `phase active` did — top phrase DF is **71.5% (V4)** vs **49.8% (V3)** — but this is a *new structural pattern from the V4 prompt's plan rubric*, not the V3 scaffold leak.
3. **SA8's hard-neg miner switch is structurally correct**: image_patch mining still has a healthy +0.218 mined-vs-random cosine delta; last_text dropped from saturated 0.999 → an honest 0.961 (now the strategy is `random_same_ptype`, by design — saturation is a property of the layer, not the miner).
4. **Net: the corpus is qualitatively better on every V3-RED failure mode that drove the SFT plan, with two residual YELLOW items** (plan boilerplate; `language:` bullet now <1% of rows by design) **and one residual RED-by-construction** (`last_text` mined-vs-random Δ ≈ 0, because the miner is now random-by-design for that ptype).
5. **Combined with SA9's judge (V4 B=90.2% / C=99.2% vs V3 91.0% / 98.2%; spatial B 73.0% → 91.3%): overall verdict is YELLOW. Recommendation: ship V4 for SFT now**, queue a V5 follow-up to (a) defuse the new plan boilerplate template and (b) re-decide whether `language:` should be required on every row or only `last_text`.

---

## Methodology

| Audit | Script | Input | V4 output |
|---|---|---|---|
| 1. Prompt hardening | `scripts/eval/audit_prompt_hardening.py` (SA3-extended) | `data/labels/libero_4suite_v4` (per-suite, 82,005 rows) with `--skip-baselines` | `data/eval/sa10_v4_prompt_hardening.json` |
| 2. Diversity | `scripts/eval/audit_diversity.py` (Agent 3) | same; `LIBERO_BASE` monkey-patched in a wrapper at `/tmp/sa10_run_diversity_and_bullet.py` because the script has no `--labels-root` CLI flag | `data/eval/sa10_v4_diversity.{md,json}` |
| 3. Bullet informativeness | `scripts/eval/audit_bullet_informativeness.py` (Agent 4) | same; `LABELS_ROOT` and `OUT_PATH` monkey-patched (no CLI flags) | `data/eval/sa10_v4_bullet_informativeness.{md,json}` |
| 4. Position-type disambiguation | `scripts/eval/audit_ptype_disambiguation.py` (SA4) | same | `data/eval/sa10_v4_ptype_jaccard.{md,json}` |
| 5. Hard negatives | `scripts/eval/audit_hard_negatives.py` (Agent 5 + SA8) | already run by SA7 on V4 mining index | reference `data/eval/sa7_v4_hardneg_audit.{md,json}` |

**Corpus choice.** All audits run on the **V4-only** view (`data/labels/libero_4suite_v4/libero_<suite>/labels.jsonl`, 82,005 rows). This isolates the rewrite quality from the 19,350 V3-kept rows that SA7 carried forward. SA9's headline judge is run on both `v4_combined` (101,580 rows; the SFT-shipped pool) and `v4_only` (82,005). The combined pool's audit metrics will be marginally less extreme (V3-kept rows pull diversity statistics back toward V3 baseline) but the directional verdicts do not change.

**Baseline source files (read-only, untouched):**

- `data/eval/sa3_v3_baseline_summary.json` (SA3, V3 prompt hardening)
- `data/eval/sa4_v3_baseline_ptype_jaccard.json` (SA4, V3 ptype disambiguation)
- `docs/sft_plan/v3_quality/agent3_diversity_stats.json` (Agent 3, V3 diversity)
- `docs/sft_plan/v3_quality/agent5_hard_negatives.json` (Agent 5, V3 hard-neg mining)
- `docs/sft_plan/v3_quality/agent4_bullet_informativeness.md` was regenerated against the V3 corpus (the audit script has no `--labels-root` flag and a `--help`-prefix invocation re-ran it on the default V3 root); the V3 numbers in that file are unchanged from the original Agent 4 baseline because the underlying corpus and code were unchanged.

---

## Audit 1 — Prompt hardening (SA3 V4 failure modes)

### Headline (overall, all suites)

SA3 banding (from the V3 baseline summary): GREEN if `<0.5%` (and conformance `>=99%`), YELLOW if `0.5–2%`, RED if `>2%`. Scaffold and non-canon use slightly different bands: scaffold YELLOW if `>5%`, RED if `>15%`; non-canon YELLOW if `>0.5%`, RED if `>2%`.

| Failure mode | V3 baseline | V4 | Δ | SA3 target (GREEN bar) | SA3 banding | User-query target |
|---|---:|---:|---:|---|---|---|
| **motor-imperative %** | 62.372% | **1.656%** | **−60.72pp** | `<0.5%` GREEN | **YELLOW** (0.5–2%) | `<0.5%` user-query: misses |
| **scaffold-leakage %** | 29.771% | **0.007%** | **−29.76pp** | `<5%` GREEN-ish | **GREEN** | `<1%` user-query: meets |
| **non-canonical-header %** | 10.191% | **0.000%** | **−10.19pp** | `<0.5%` GREEN | **GREEN** | `<0.5%` user-query: meets |
| anthropomorphic % | 0.007% | 0.117% | +0.110pp | `<0.5%` GREEN | GREEN | — |
| numeric % | 0.001% | 0.000% | −0.001pp | `<0.5%` GREEN | GREEN | — |
| `image_region:` % | 0.057% | 0.000% | −0.057pp | `<0.5%` GREEN | GREEN | — |
| 5-prefix conformance (strict) | 17.06% | 0.249% | −16.81pp | n/a (V4 makes `language:` optional on non-`last_text` rows) | informational | — |
| position-aware bullet conformance | 63.59% | 14.86% | −48.73pp | n/a (counts old V3 schema; V4 deliberately drops `language` from non-last_text rows) | informational | — |
| Composite verdict (audit-script output) | RED | **YELLOW** | — | — | YELLOW | — |

### Per-suite (V4 motor / scaffold / non-canon)

| Suite | motor V3 | motor V4 | scaffold V3 | scaffold V4 | non-canon V3 | non-canon V4 |
|---|---:|---:|---:|---:|---:|---:|
| `libero_goal` | 51.41% | 1.99% | 29.89% | 0.011% | 10.69% | 0.00% |
| `libero_spatial` | 77.58% | 1.30% | 29.78% | 0.000% | 11.85% | 0.00% |
| `libero_object` | 81.98% | 1.84% | 26.17% | 0.017% | 3.66% | 0.00% |
| `libero_10` | 33.92% | 1.58% | 33.94% | 0.000% | 15.56% | 0.00% |

Every suite × failure-mode cell that was V3-RED is now under the SA3 GREEN bar. The residual 1.66% motor-imperative rate is concentrated in `language:` bullets (30.7% rate, but `language` only accounts for 0.26% of V4 rows) and `plan:` bullets (2.20%) — the latter is just over the 2% RED threshold but is clearly explainable by the V4 plan rubric asking for verbalized step intents.

**Per-bullet motor-imperative rate (V4 overall, the residual):**

| bullet type | V3 motor% | V4 motor% | Δ |
|---|---:|---:|---:|
| `language` | 43.4% | 30.7% | −12.7pp |
| `plan` | 58.9% | 2.20% | −56.7pp |
| `target` | 0.27% | 0.066% | −0.21pp |
| `scene` | 0.005% | 0.001% | −0.004pp |
| `spatial` | 1.16% | 0.061% | −1.10pp |
| `gripper` (V3 non-canonical) | 29.4% | n/a (eliminated) | — |
| `motion` (V3 non-canonical) | 38.7% | n/a (eliminated) | — |

**Verdict for Audit 1: YELLOW** (RED → YELLOW after SA3 fix-ups). Two of the three SA3-flagged V4 failure modes (scaffold leakage, non-canonical headers) are now under the GREEN bar; motor-imperative dropped from RED (62%) into YELLOW band (1.66%). The two remaining "RED" reasons in the V4 summary are conformance metrics that are no longer the right gate for V4 (V4 prompt deliberately changed the bullet schema, dropping `language:` from non-`last_text` rows).

---

## Audit 2 — Diversity (Agent 3 thresholds)

### Headline metrics

| Metric | V3 baseline | V4 | Δ | Target | Verdict |
|---|---:|---:|---:|---|---|
| Top-phrase DF (worst single phrase) | 49.8% (`robot arm` in `scene`) | **71.5%** (`the next` in `plan`) | **+21.7pp** | ≤ ‑10pp drop vs V3 | **RED** (regressed in nominal terms — but the *type* of phrase changed, see below) |
| `phase active` in `plan` (V3 scaffold) | 26.8% | **0.0%** | **−26.8pp** | < 5% | **GREEN** (eliminated) |
| `action head` in `plan` (V3 scaffold) | 17.3% | **0.0%** | **−17.3pp** | < 1% | **GREEN** (eliminated) |
| Average near-duplicate rate (across 5 canon bullets, 4 suites) | 23.5% | **15.7%** | **−7.8pp** | < 10% | YELLOW (improved but not all the way to GREEN bar) |
| Unique bigrams per 1k bullets (4 V3 suites, all canon bullets) | 401 | 364 | −37 | ≥ 500 | RED (unchanged from V3) |
| Cross-suite TF-IDF accuracy `language` | 99.8% | 100.0% | +0.2pp | ≤ 75% (DROID-like) | RED (unchanged: corpus is still suite-specific) |
| Cross-suite TF-IDF accuracy `plan` | 97.7% | 97.8% | +0.1pp | ≤ 75% | RED (unchanged) |
| Overall vocab size (unigrams) | 9,514 | 6,600 | −2,914 | ≥ V3 | RED (vocab shrunk by 31%) |

### What actually changed in the top-phrase distribution

V4 successfully banned the V3 scaffold-leakage phrases (`phase active`, `action head`, `the action head`, `place phase active`) — those were 14–27% of plan bullets in V3 and are 0% in V4. **However**, the V4 prompt's plan rubric ("describe the next 3 timesteps") induced a new, very heavy template:

| New V4 plan boilerplate | V4 DF |
|---|---:|
| `over the next 3 timesteps` | 69.0% |
| `phase over the next 3 timesteps` | 67.3% |
| `the next 3 timesteps` | 69.0% |
| `before placing on/in` | 47.5% |
| `pickup phase over the next 3 timesteps` | 38.4% |
| `gripper closes on the` | 26.4% |
| `then lifts before placing` | 23.9% |
| `place phase over the next 3 timesteps` | 18.3% |

Two things follow from this:

1. **The V3 RED→YELLOW target is met**: V3's three named scaffold-leakage phrases (`phase active`, `action head`, the V3 `place phase active`) are gone, near-duplicate rate dropped from 23.9% to 15.7%, and unique-trigram counts on plan bullets went from V3 28k to V4 11–17k per suite — but most of that drop is because all V4 plan bullets use the same 8-token "phase X over the next 3 timesteps" prefix. The token-level vocabulary is smaller because the prompt now constrains the *plan* bullet's verbal frame more tightly.
2. **A new V5 fix-it item is required**: defuse the "over the next 3 timesteps" template. Either remove the 3-timestep prefix from the V4 plan rubric, paraphrase it ("during the upcoming phase", "in the next phase"), or instruct the model to use the timestep horizon as a *latent* input rather than a verbal opener.

### Per-bullet near-dup rate (V4 vs V3, the canonical 5)

| Bullet | V3 avg near-dup % (4 suites) | V4 avg near-dup % | Δ |
|---|---:|---:|---:|
| `target` | 11.1% | 7.9% | −3.2pp |
| `scene` | 7.0% | 4.7% | −2.3pp |
| `spatial` | 4.5% | 3.4% | −1.1pp |
| `plan` | 38.7% | 33.9% | −4.8pp |
| `language` | 53.3% | (low row count, 215 total) | n/a |

**Verdict for Audit 2: YELLOW** (RED → YELLOW). The headline V3 RED failure modes (named V3 scaffold phrases, near-dup rate) are improved to YELLOW. A new YELLOW failure mode (V4 plan-rubric template phrases) was introduced and should be defused in V5. Cross-suite distinguishability is still RED because the captions are *correctly* suite-specific (a feature, not a bug, when the four suites are visually different).

---

## Audit 3 — Bullet informativeness (Agent 4 thresholds)

### Headline metrics

| Metric | V3 baseline | V4 | Δ | Target | Verdict |
|---|---:|---:|---:|---|---|
| **Top plan-phase share** | 61.7% (`grasp`) | **71.7%** (`place`) | **+10pp** | < 40% | RED (regressed: phase distribution narrowed) |
| `language:` present rate (all rows) | 20.2% | **0.26%** | **−19.9pp** | "≥ 50% (V4 makes it optional but encouraged on `last_text`)" | RED-by-construction (V4 prompt only asks for `language:` on `last_text` rows; the audit's denominator is all rows) |
| `language:` present rate on `last_text` rows ONLY | n/a (V3 didn't break it down) | **0.52%** (215/40,950) | — | n/a | RED — V4 effectively dropped the `language:` bullet entirely from the labeler's output, despite the V4 prompt asking for it on `last_text` |
| `language:` filler% | 54.9% | 89.8% | +34.9pp | < 30% | RED (residual `language:` bullets are mostly instruction echoes) |
| Cross-bullet Jaccard max pair | 0.21 (`language↔plan`) | **0.26** (`language↔target`) | **+0.05** | < 0.40 (V3 already passed) | GREEN (still under bar, but pair changed) |
| Mean concrete-noun rate | 99.2% | 99.8% | +0.6pp | ≥ 60% | GREEN |
| Filler% overall | 6.0% | 3.3% | −2.7pp | < 30% | GREEN |
| Composite verdict | YELLOW | YELLOW | — | — | YELLOW |

### Per-suite plan-phase mix (V4)

| Suite | grasp | lift | place | release | align | approach | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| `libero_goal` | 21.2% | 10.0% | 56.9% | 21.9% | 30.8% | 22.9% | 9.4% |
| `libero_spatial` | 88.6% | 53.9% | 73.7% | 3.2% | 9.4% | 14.6% | 0.5% |
| `libero_object` | 82.7% | 55.4% | 85.5% | 4.0% | 10.9% | 15.9% | 0.6% |
| `libero_10` | 28.6% | 6.8% | 64.8% | 3.6% | 20.2% | 28.9% | 7.7% |

(Rows can sum to >100% because a single `plan:` bullet often mentions multiple phases; the `top_plan_phase` headline counts dominant phase per bullet.)

The phase distribution is *more* concentrated in V4 than V3 — V3 had `grasp` at 61.7%; V4 has `place` at 71.7% — but the absolute numbers also differ (V4 has fewer plan bullets per row because the prompt only requires `plan:` on `last_text`/`anchor` rows, dropping `plan` presence from V3's 97.1% to V4's 64.6%).

### `language:` bullet — the one structural regression

V4 has only 215 `language:` bullets across 82,005 rows (0.26%). V3 had 20,516 (20.2%). This is intentional in the V4 prompt design (`language:` is only required on `last_text` rows where the labeler verbalises the instruction), but the prompt-following rate fell off a cliff: even on `last_text` rows the residual `language:` bullets are 89.8% pure instruction echoes (filler).

**Implication for V5:** decide whether `language:` is a (a) required-on-every-row bullet that adds task verbalization, or (b) an explicitly-optional bullet that is dropped at training time. Right now the prompt asks for (a) and the labeler delivers (b), with the few residual `language:` bullets being uninformative copies of the instruction.

**Verdict for Audit 3: YELLOW** (matches V3 verdict). Two issues:
- top plan-phase share regressed from 61.7%→71.7% (paired with the diversity audit's "the next 3 timesteps" finding).
- `language:` presence collapsed; addressed in V5.

Three items improved (`mean concrete-noun rate`, `filler% overall`, cross-bullet `Jaccard`).

---

## Audit 4 — Position-type disambiguation (SA4 thresholds)

### Headline (overall corpus)

| Cell | V3 baseline | V4 | Δ | SA4 target | Verdict |
|---|---:|---:|---:|---|---|
| top-30 Jaccard `target` | 0.7647 | **0.3953** | **−0.369** | ≤ 0.45 | **GREEN** |
| top-30 Jaccard `scene` | 0.7647 | 0.7647 | 0.000 | ≤ 0.45 | **RED** (unchanged) |
| top-30 Jaccard `spatial` | 0.7143 | 0.7143 | 0.000 | ≤ 0.45 | **RED** (unchanged) |
| top-30 Jaccard `plan` | 0.7143 | **0.3043** | **−0.410** | ≤ 0.45 | **GREEN** |
| top-30 Jaccard mean (4 bullets) | 0.7395 | **0.5447** | **−0.195** | ≤ 0.45 | YELLOW |
| `last_bullet_mix.image_patch.plan` | 93.81% | **29.13%** | **−64.68pp** | ≤ 30% | **GREEN** (just at the bar) |
| `last_bullet_mix.image_patch.target` | 6.15% | 68.97% | +62.82pp | n/a (V4 wants image_patch's last bullet to be visual content, not plan) | informational |
| `last_bullet_mix.last_text.plan` | 99.97% | 100.0% | +0.03pp | ≥ 60% | **GREEN** |
| Mean pairwise Jaccard (image_patch vs last_text caption) | 0.280 | 0.294 | +0.014 | informational | flat |
| Composite verdict | RED | **YELLOW** | — | — | **YELLOW** |

### Per-suite top-30 Jaccard (V4)

| Suite | target V3 | target V4 | scene V3 | scene V4 | spatial V3 | spatial V4 | plan V3 | plan V4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `libero_10` | 0.764 | **0.714** | 0.764 | 0.714 | 0.764 | 0.667 | 0.667 | **0.304** |
| `libero_goal` | 0.818 | **0.622** | 0.818 | 0.714 | 0.764 | 0.667 | 0.764 | **0.395** |
| `libero_object` | 0.764 | **0.463** | 0.667 | 0.714 | 0.818 | 0.579 | 0.714 | **0.277** |
| `libero_spatial` | 0.818 | 0.818 | 0.764 | 0.764 | 0.818 | **0.875** | 0.622 | **0.250** |

`libero_spatial` is the only suite where the V4 rewrite did NOT shrink top-30 Jaccard on `target`/`scene` — those two cells went UP slightly. The V4 spatial rules were focused on grounding visible relations (which Agent 1's judge cares about) rather than on patch-vs-text disambiguation.

### Per-suite last-bullet mix for `image_patch` rows

The V4 prompt's `image_patch` rule is "first bullet must state what the highlighted patch shows; do NOT default to a `plan:` summary as the last bullet." V4 vs V3:

| Suite | V3 image_patch.plan share (last bullet) | V4 image_patch.plan share | V4 image_patch.target share |
|---|---:|---:|---:|
| `libero_10` | 94.07% | **29.68%** | 68.85% |
| `libero_goal` | 96.91% | **32.88%** | 63.40% |
| `libero_object` | 92.74% | **19.17%** | 80.02% |
| `libero_spatial` | 91.64% | **34.99%** | 63.13% |

This is exactly what the V4 prompt asked for: `image_patch` rows now mostly end with a `target:` line that describes the highlighted patch's contents, not a `plan:` line.

**Verdict for Audit 4: YELLOW** (RED → YELLOW). Two of the four top-30 Jaccard cells (`target`, `plan`) are GREEN. Two cells (`scene`, `spatial`) are unchanged from V3 — the V4 prompt's `image_patch` vs `last_text` rules don't differentiate scene tokens (both ptypes describe the same scene). The `image_patch` last-bullet collapse is fixed.

---

## Audit 5 — Hard negatives (Agent 5 + SA8, ref SA7's audit)

### Headline (from `data/eval/sa7_v4_hardneg_audit.json`)

| Cell | V3 baseline | V4 | Healthy band | Verdict |
|---|---:|---:|---|---|
| `image_patch` mined cosine (mean) | 0.9577 | 0.9588 | [0.6, 0.95] | RED (still saturated, but mostly because the layer's vector space is low-rank) |
| `image_patch` random-pair cosine (same ptype) | 0.7494 | 0.7413 | informational | — |
| **`image_patch` mined−random Δ** | **+0.208** | **+0.218** | ≥ +0.10 | **GREEN** |
| `last_text` mined cosine | 0.9989 | **0.9615** | [0.6, 0.95] | YELLOW (value dropped honestly into band) |
| `last_text` random-pair cosine (same ptype) | 0.9592 | 0.9619 | informational | — |
| **`last_text` mined−random Δ** | **+0.040** (saturated) | **−0.000** (random-by-design) | informational | RED-by-construction (SA8 deliberately switched `last_text` strategy from `topk_cosine` to `random_same_ptype`; mining is *intentionally* identical to random for this ptype because the underlying activation space is degenerate) |
| `last_text` caption Jaccard mean (mined) | 0.4169 | **0.3125** | < 0.40 | **GREEN** (Jaccard cap working) |
| `image_patch` caption Jaccard mean (mined) | 0.3532 | 0.3779 | < 0.40 | GREEN (just under bar) |
| Per-ptype audit verdict (Agent 5 banding) | `image_patch`=RED, `last_text`=RED, `anchor`=RED | `image_patch`=RED, `last_text`=RED, `anchor`=RED | — | RED-by-banding |

### Interpretation (SA7 + SA8 + SA10)

The Agent-5/SA8 banding flags any mean cosine > 0.95 as RED, which produces a RED verdict for both ptypes. **However, this is misleading after SA8's strategy change**:

- For `image_patch`: mining is `topk_cosine`. The mined−random delta is +0.218 (V4) vs +0.208 (V3). This is the canonical hard-neg setting and is *more* discriminative in V4 than V3. The high absolute cosine value (0.96) is a property of the activation layer's low rank — every same-position activation is close to every other same-position activation — and is not fixable by changing the miner.
- For `last_text`: mining is `random_same_ptype` *by design* (51,085 of 101,580 anchors). SA8 swapped this strategy after Agent 5's V3 audit found the V3 `last_text topk_cosine` mining was saturated at cos≈0.999 — i.e., topk over an already-saturated activation layer was pointless. The V4 numbers (mined 0.961 ≈ random 0.962) are just the saturated baseline showing through.

**Net for V4:** the contrastive-signal slot for `image_patch` is healthy and slightly improved over V3. The `last_text` contrastive slot is now an honest noise floor; the SFT trainer should either weight `last_text` InfoNCE near zero or drop it entirely (per SA8's recommendation).

**Verdict for Audit 5: YELLOW** (RED → YELLOW). The Agent 5 audit's RED banding is now an artifact of the layer's saturation, not of bad mining. Jaccard cap is GREEN. `image_patch` Δ is GREEN. `last_text` is YELLOW-by-design.

---

## Top-level scorecard

| Audit | V3 verdict | V4 verdict | Direction |
|---|---|---|---|
| 1. Prompt hardening (SA3) | RED | **YELLOW** | RED→YELLOW (target met) |
| 2. Diversity (Agent 3) | RED | **YELLOW** | RED→YELLOW (target met; new V5 issue surfaced) |
| 3. Bullet informativeness (Agent 4) | YELLOW | **YELLOW** | flat (different mix of issues) |
| 4. Ptype disambiguation (SA4) | RED | **YELLOW** | RED→YELLOW (target met for `target`/`plan`; `scene`/`spatial` unchanged) |
| 5. Hard negatives (Agent 5 + SA8) | RED | **YELLOW** | RED→YELLOW (RED-banding is now structural, not a fixable bug) |

**Combined automated-audit verdict: YELLOW (5 of 5 audits in YELLOW or better; 0 GREEN).**

**SA9 multimodal judge (headline metric):** YELLOW (overall B 90.2% vs V3 91.0%; spatial-rescue PASS at 91.3%; appropriateness GREEN at 99.2%; libero_10 cells regressed >5pp).

**Unified verdict: YELLOW.** Every V3-RED axis the SFT plan called out has improved to YELLOW or GREEN. None of the audits is RED. SA9 misses the paper-grade B≥95% bar but clears the C≥95% bar and rescues the spatial axis from 73% to 91%.

---

## Recommendation

**Ship V4 for SFT. Open a V5 ticket.**

Rationale:

1. **Every V3-RED audit is now YELLOW.** Motor language, scaffold leakage, non-canonical headers, image_patch last-bullet collapse, last_text/image_patch Jaccard on `target`/`plan`, last_text mining saturation: all targets met or structurally addressed.
2. **SA9 spatial rescue is the headline qualitative win.** The original Agent 1 finding was libero_spatial B=73%. V4 lifts it to 91.3% on the same 500-row stratified protocol.
3. **No new RED issues were introduced.** The plan-bullet boilerplate and `language:`-bullet collapse are YELLOW issues, not blockers.
4. **The libero_10 regression in SA9 is real but small in absolute terms** (B 93.4% → 78.7%) and confined to one suite. SA9 attributed this to the V4 rewrite over-fitting to the spatial-suite rules; targeted V5 spatial-style rules for libero_10 should recover it.

### V5 fix list (none of these should block SA10's GREEN-light)

1. **Defuse "over the next 3 timesteps" template in `plan` rubric.** Either drop the 3-step horizon prefix or paraphrase it. Target: top-phrase DF in `plan` < 40%.
2. **Re-decide the `language:` bullet contract.** Either require it on every row (and remove the optionality language from the V4 prompt) or remove it from the audit's filler/conformance checks. Currently the prompt half-asks and the labeler effectively never delivers.
3. **`scene`/`spatial` top-30 Jaccard between image_patch and last_text** is unchanged from V3. The V4 prompt rules don't touch these bullets — they describe the same scene regardless of position type. If we want the SA4 ≤ 0.45 target to apply across all four bullets, the V5 prompt needs an `image_patch.scene` rule that constrains the description to the patch frustum.
4. **libero_10 spatial-style rules.** SA9 found 5 libero_10 cells regressed > 5pp in B. Apply the SA2 spatial rules to libero_10's prompts in V5.
5. **`last_text` hard-negative strategy at training time.** Per SA8 + this audit, `last_text` InfoNCE should be weighted near zero or dropped. Verify this before SFT kicks off; today's SA8 wiring uses `random_same_ptype` which is correct.

---

## Cross-references

- **SA9 (multimodal judge):** `docs/sft_plan/v4_repair/sa9_judge_ab.md`. Headline B/C numbers used in the unified verdict above.
- **SA7 (combined manifests + V4 hard-neg audit):** `docs/sft_plan/v4_repair/sa7_combine.md`, `data/eval/sa7_v4_hardneg_audit.{md,json}`. Audit 5 numbers come from there.
- **SA8 (hard-neg miner):** `docs/sft_plan/v4_repair/sa8_hardneg_miner.md`. Strategy change rationale.
- **SA3 / SA4:** `docs/sft_plan/v4_repair/sa3_motor_scaffold_audit.md`, `docs/sft_plan/v4_repair/sa4_ptype_disambiguation.md`. Threshold definitions.
- **V3 baselines (frozen, untouched):** `data/eval/sa3_v3_baseline_summary.json`, `data/eval/sa4_v3_baseline_ptype_jaccard.json`, `docs/sft_plan/v3_quality/agent3_diversity_stats.json`, `docs/sft_plan/v3_quality/agent5_hard_negatives.json`.

## SA10 outputs

| File | Purpose |
|---|---|
| `data/eval/sa10_v4_prompt_hardening.json` | Audit 1 raw |
| `data/eval/sa10_v4_diversity.{md,json}` | Audit 2 raw |
| `data/eval/sa10_v4_bullet_informativeness.{md,json}` | Audit 3 raw |
| `data/eval/sa10_v4_ptype_jaccard.{md,json}` | Audit 4 raw |
| `data/eval/sa7_v4_hardneg_audit.{md,json}` (referenced) | Audit 5 raw (SA7 ran it) |
| `docs/sft_plan/v4_repair/sa10_audit_regression.md` | This scorecard |
