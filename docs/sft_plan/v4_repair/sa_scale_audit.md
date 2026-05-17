# SA Scale Audit — V3 SFT FVE / Cosine Diagnosis

**Audit run:** 2026-05-16
**Checkpoint under inspection:** `data/sft/libero_4suite_v3/`
**α (P75 ‖h‖₂ from extraction stats):** `203.977134872893`
**Probe script:** `/tmp/probe_scale_bug.py` (output: `/tmp/probe_scale_bug.out`)
**Renorm probe:** `/tmp/probe_renorm.out` (64 rows, by position type)

---

## Verdict

**`SCALE_BUG_RULED_OUT`** — α is applied consistently between training and eval; FVE / MSE / cosine on the held-out eval are *numerically reproducible* from the same code path the trainer uses.

There is **no sign flip** (`cos(h, -ĥ) = -cos(h, +ĥ)` everywhere, as expected) and **no hidden-dim transpose** (`cos(h, ĥ.flip(-1)) ≈ 0`). Position-key alignment is correct: the dataset row that yields a given `h` returns the matching gold caption (verified end-to-end on row 0; round-tripped through `labels.jsonl`).

But the AR head has converged to a **systematically over-magnified prediction**:

```
mean ‖ĥ‖ / ‖h‖   = 5.327×   (eval-side, after pred_scaled * α)
mean cos(h, ĥ)   = +0.353
mean ‖pred_scaled‖ = 4.71  (raw AR.forward output)
mean ‖target_scaled = h/α‖ ≈ 178/204 ≈ 0.87
```

`pred_scaled` is **~5.4× too large** versus the target it was supposed to regress to in α-scaled space. With a directional accuracy of only `cos ≈ 0.35`, that magnitude inflation is precisely what produces FVE = −24 and MSE ≈ 395 in unscaled space. Renormalizing predictions to `‖h‖` (per-row) eliminates the catastrophe (see "Renorm reference numbers" below): FVE jumps from −24.0 → −0.31 with the *same* directional accuracy.

This is not a one-line patch in the eval / training code: α is applied correctly. It is a **training pathology** (weak directional signal × NCE-on-cosine that ignores magnitude × MSE that didn't pull magnitude down hard enough). The parent agent's V4 re-label is the right primary remedy; magnitude regularization in SFT is a complementary mitigation discussed at the end.

---

## Numerical reproduction of the V3 metrics

Probe over **64 random held-out rows**, episode-stratified (`split_by="episode"`, `held_out_fraction=0.05`, seed=0):

| pipeline                              |   cos    |   FVE    |   MSE   | ‖ĥ‖/‖h‖ |
| ------------------------------------- | :------: | :------: | :-----: | :-----: |
| `+ĥ` (current eval `pred_scaled * α`) | +0.3530  | **−23.99** | **394.94** |  5.33×  |
| `+ĥ` rescaled per-row to `‖h‖`        | +0.3530  | −0.310   |  20.70  |  1.00×  |
| predict `mean h` (per-dim baseline)    | +0.8451  | +0.6921  |   4.87  |  ≈0.85× |
| `−ĥ` (sign flip)                       | −0.3530  | −113.2   | 538.66  |  5.33×  |
| `ĥ.flip(-1)` (hidden reversed)         | −0.0096  | −101.2   | 482.30  |  5.33×  |

V3 final eval (in `metrics.jsonl`, step 15000):

```
overall: FVE=-23.59 MSE=373.50 cos=0.364
image_patch: FVE=-20.91 MSE=401.45 cos=0.327
last_text:   FVE=-27.57 MSE=345.75 cos=0.402
anchor:      FVE=-32.83 MSE=405.59 cos=0.361
```

Probe by position-type (n=64):

```
last_text   n=31  cos=+0.387  FVE_curr=-28.97  FVE_renorm=-0.23  ‖ĥ‖/‖h‖=5.76
image_patch n=32  cos=+0.323  FVE_curr=-20.86  FVE_renorm=-0.36  ‖ĥ‖/‖h‖=4.89
anchor      n= 1  cos=+0.262  FVE_curr=-33.10  FVE_renorm=-0.48  ‖ĥ‖/‖h‖=6.02
```

These line up to within sampling noise of the 1000-row eval, confirming the path the eval ran is the same one the probe runs. (Note FVE in `_StreamingFve.compute()` uses a **scalar** mean, not per-dim; that's why the per-dim mean baseline scores +0.69 in the probe but the canonical FVE-vs-scalar-mean is the V3 number.)

---

## Code paths traced (every α touch on the gradient path)

### Targets are divided by α exactly once (training only)

`src/nla/models/ar.py` `forward_sft` — the only place that scales targets:

```222:227:src/nla/models/ar.py
        pred_scaled = self.forward(explanations, device=target_activations.device)
        target_scaled = (target_activations / self.cfg.alpha).to(pred_scaled.dtype)
        if self.cfg.clip_target_scaled is not None:
            clip = float(self.cfg.clip_target_scaled)
            target_scaled = target_scaled.clamp(-clip, clip)
        mse = nn.functional.mse_loss(pred_scaled, target_scaled)
```

`src/nla/training/grpo.py` `compute_grpo_loss` — same convention for RL (not used in V3 SFT, but checked for consistency):

```533:536:src/nla/training/grpo.py
        ar.train()
        pred_scaled = ar(rollout_texts, device=device)                   # (B*K, H_act)  WITH grad
        target_scaled = (acts_rep / ar.cfg.alpha).to(pred_scaled.dtype)
        rewards = -((pred_scaled.detach() - target_scaled.detach()) ** 2).mean(dim=-1).float()
        ar_mse = ((pred_scaled - target_scaled) ** 2).mean()             # scalar, with grad
```

### Predictions are multiplied by α exactly once (eval only)

`src/nla/training/sft.py` teacher-forced eval:

```411:414:src/nla/training/sft.py
        ce_n += acts.shape[0]
        pred_scaled = ar(batch["description"], device=device)
        pred_unscaled = pred_scaled.detach().float() * alpha
        fve_acc.update(acts.float(), pred_unscaled, batch["position_type"])
```

`src/nla/training/sft.py` closed-loop eval:

```375:378:src/nla/training/sft.py
        texts = gen_out["text"]
        pred_scaled = ar(texts, device=device)
        pred_unscaled = pred_scaled.detach().float() * ar.cfg.alpha
        fve_acc.update(acts.float(), pred_unscaled, batch["position_type"])
```

`scripts/eval/closed_loop_retrieval.py` — same convention for the retrieval-margin eval:

```190:201:scripts/eval/closed_loop_retrieval.py
        with torch.no_grad():
            pred_scaled = ar(chunk, device=args.device).float()
        ar_out_scaled.append(pred_scaled.detach().to("cpu"))
    H_hat_scaled = torch.cat(ar_out_scaled, dim=0)            # (N, D), scaled
    H_hat_raw = H_hat_scaled * alpha                          # back to unscaled space
    assert H_hat_raw.shape == (N, D)

    # Pairwise cosine. Cosine is scale-invariant so it doesn't matter whether
    # both sides are scaled or unscaled, but be consistent.
```

### `predict()` returns unscaled by default (used by SAE / steering downstream)

```175:178:src/nla/models/ar.py
    def predict(self, explanations: list[str], *, unscale: bool = True) -> torch.Tensor:
        with torch.no_grad():
            pred_scaled = self.forward(explanations)
        return pred_scaled * self.cfg.alpha if unscale else pred_scaled
```

### Activations are stored *unscaled* (raw extraction)

`src/nla/extraction/storage.py` writes `features.contiguous()` directly into the safetensors buffer (no α touch); the `LabeledPositionDataset.__getitem__` returns `features[pos]` without scaling, so the trainer always sees raw `h`:

```549:561:src/nla/training/dataset.py
    def __getitem__(self, i):
        entry = self.labels[i]
        global_idx = self._index_by_id[entry.source_example_id]
        rec = self.reader.records[global_idx]
        item = self.reader[global_idx]
        features = item["features"]
        pos = entry.position_index
        if pos >= features.shape[0]:
            raise IndexError(
                f"Label position {pos} >= seq_len {features.shape[0]} for "
                f"example {entry.source_example_id}"
            )
        vec = features[pos].contiguous().to(torch.float32)
```

**Net round-trip:** training divides by α once, eval multiplies by α once. They cancel. The α value (`203.977…`) is identical across `config.json`, `ar/ar_config.json`, and `av/av_config.json`. **No double-α, no missing-α, no inverted-α.**

---

## What the probe ruled out

| hypothesis                                                | check                                                                                                  | verdict                |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ---------------------- |
| AR predicts `h * α` and eval also `* α`                   | `cos(h, pred_scaled / α)` is the same 0.353 (cosine is scale-invariant)                                | **ruled out** — magnitudes are 5×, not α=204×. A double-α bug would give ‖ĥ‖/‖h‖ ≈ α, not 5.4. |
| Sign flip somewhere                                       | `cos(h, -ĥ) = -0.353`                                                                                  | **ruled out** — exact negation, not anywhere near +0.96 |
| Hidden-dim reversal (`flip(-1)`)                          | `cos(h, ĥ.flip(-1)) = -0.010`                                                                          | **ruled out** — random (no transpose). |
| `min_bullet_lines` filter / position-key misalignment     | row 0: ds-fed `(example_id, pos, ptype) == (object__traj000272_step000034, 147, last_text)` matches `labels.jsonl` `meta`; gold caption matches the description fed to AR. | **ruled out** — alignment correct. |

---

## What the probe found instead

The AR head has converged to predicting `pred_scaled` with **norm ~4.71** when the target is `h/α` with norm **~0.87**.

Per-row breakdown (from `/tmp/probe_scale_bug.out`, 16 rows; full table in that file):

```
i  ptype          ‖h‖     ‖p_uns‖  ratio  cos(h,+p_uns)
0  last_text     155.10   894.22   5.77   +0.341
1  last_text     154.33   834.01   5.40   +0.422
2  last_text     156.20   900.17   5.76   +0.385
…                                  …        …
15 image_patch   203.46   979.94   4.82   +0.278
```

Why this fails to fix itself during training, **even though the MSE term in scaled space is supposed to pull `‖pred_scaled‖` down toward `‖h/α‖ ≈ 0.87`**:

1. **Contrastive-cosine loss + temperature 0.1 dominates the AR loss in late training.** From `metrics.jsonl` step 14000–15000, `ar_mse` ≈ 0.003–0.011 (in scaled space) while `ar_nce` ≈ 0.5–1.5; with `ar_contrastive_weight = 0.5`, the NCE term contributes ~0.25–0.75 to the AR objective vs. ~0.005 from MSE — a **50–150× ratio favoring NCE**. The cosine-NCE Jacobian is **purely tangential** to `pred_scaled` (`d cos(p,t)/d p` is normal to `p`), so its magnitude pressure on `pred_scaled` is **zero**. MSE is the only loss that pins magnitude, but its weight is overwhelmed.
2. **The MSE is "happy enough" at large magnitudes.** At cos≈0.35, with `‖pred_scaled‖ = 4.71` and `‖target_scaled‖ = 0.87`, per-element MSE is ≈ `(4.71² + 0.87² − 2·4.71·0.87·0.35) / 2048 ≈ 0.0098` — the value `metrics.jsonl` reports. Shrinking `‖pred_scaled‖` to 0.87 with the same cos=0.35 would reduce per-element MSE to ≈ `2·0.87²·(1−0.35)/2048 ≈ 0.00048` — a ~20× drop in MSE, but only a 0.0094 decrease in the *AR objective*, while NCE gradient is much larger. The gradient field is pinned in a flat-direction valley.
3. **`clip_target_scaled = 5.0` is ineffective as a magnitude regularizer.** It clamps the *target* per-element to ±5 (per-element targets are ~0.019 in this corpus, nowhere near the clip), so it never bites. It does nothing to bound prediction magnitude.

The AR head weight grew from Xavier-init norm ≈ 0.026 per row to ≈ 1.06 per row (40× growth), which is consistent with the optimizer driving AR to maximize cosine separation under NCE without magnitude pinning.

---

## Estimated impact of fixes

If the parent agent layered a magnitude regularizer on top of V4's relabel-only fix:

| intervention                                                                | expected `cos` | expected FVE | expected MSE |
| --------------------------------------------------------------------------- | :------------: | :----------: | :----------: |
| V4 relabel only (assuming new captions raise `cos` from 0.35 → 0.55)         |     ~0.55      |     ~0.4     |     ~12      |
| V4 relabel + magnitude calibration (renorm `pred_unscaled` to `‖h‖_train_p75`) |     ~0.55      |    ~0.55     |     ~10      |
| V3-as-is, eval just renormalizes ĥ to ‖h‖                                    |     0.353      |    −0.31     |    20.7      |
| V3-as-is current eval                                                        |     0.353      |    −24.0     |    394.9     |

The **eval-time renormalization alone** would close the FVE gap by ~98% (−24 → −0.3) without touching training, but it **does not improve cosine** and so cannot fix the underlying steering signal (which is what `closed_greedy_cosine = 0.39 < 0.55` measures and what the LIBERO sim-A/B numbers depend on).

So:
- **The headline FVE = −23.59 number is mostly cosmetic** — once magnitude calibration is fixed at *eval time*, FVE recovers, but the *actual reconstruction quality is bad* (cosine 0.35 < mean-baseline cosine 0.85 means AR is *worse than predicting the dataset mean*, which is what V2 postmortem identified as the AR-shortcut failure).
- **The real problem is data**: AV-generated captions are not informative enough about `h` to drive `cos(h, AR(AV(h)))` beyond ~0.35. That's what V4's relabel-and-rerank pipeline is designed to fix.

---

## Suggested patches (parent agent decides)

These are not bugs; they are *training/eval mitigations* that would make the V3 numbers less misleading and possibly also improve V4 outcomes.

### Patch 1 (eval-only, 1 line, zero risk): renormalize at eval time

In `src/nla/training/sft.py::_evaluate` (and `_closed_loop_eval`), replace:

```python
pred_unscaled = pred_scaled.detach().float() * alpha
```

with:

```python
pred_unscaled = pred_scaled.detach().float() * alpha
# Magnitude calibration: ‖pred_scaled‖ is unconstrained by NCE; project it
# onto the dataset's empirical norm scale so FVE/MSE reflect direction error,
# not magnitude inflation. Cosine and retrieval are unaffected (scale-invariant).
target_norm = acts.float().norm(dim=-1, keepdim=True)
pred_norm = pred_unscaled.norm(dim=-1, keepdim=True).clamp_min(1e-6)
pred_unscaled = pred_unscaled * (target_norm / pred_norm)
```

This **does not change cos / retrieval / closed-loop**. It does turn FVE = −24 into FVE ≈ −0.3 and MSE 395 → 21. **Use only as a diagnostic tool**, not as a replacement for fixing AR. Recommend gating behind a CLI flag `--rescale-pred-magnitude` so the original (uncalibrated) FVE is still preserved as the headline.

### Patch 2 (training-side, 5 lines, moderate risk): explicit magnitude term in AR loss

In `src/nla/models/ar.py::forward_sft`, before returning, add a magnitude penalty:

```python
# Magnitude calibration: cosine-NCE leaves ‖pred_scaled‖ unconstrained.
# Pin it to ‖target_scaled‖ explicitly so MSE in scaled space reflects
# direction error, not magnitude inflation.
mag_pred = pred_scaled.norm(dim=-1)
mag_tgt  = target_scaled.norm(dim=-1)
mag_term = ((mag_pred - mag_tgt) ** 2).mean()
mse = mse + 0.1 * mag_term  # weight tuned so it doesn't crush directional learning
```

Plumb a config knob `ar_magnitude_weight: float = 0.1` through `ARConfig` so it can be ablated. **Has not been tested by this audit; the parent agent should A/B vs no-mag-term on a 500-step smoke run before committing.**

### Patch 3 (training-side, 1 line, lowest risk): drop `ar_contrastive_weight` to 0.1

The current `0.5` lets NCE dominate AR's gradient (50–150× MSE). At 0.1, MSE has more authority over magnitude. **Side-effect:** weaker template-collapse regularization, which V3 specifically increased to fight V2's failure mode. Not recommended without simultaneously tightening the data signal (V4 relabel) so cos can climb past 0.5.

---

## Files read for this audit

- `src/nla/training/sft.py` (entire)
- `src/nla/training/dataset.py` (entire)
- `src/nla/models/ar.py` (entire)
- `src/nla/models/av.py` (α use only — confirmed unscaled storage + α-multiplied projection slot)
- `src/nla/training/checkpoint.py` (entire)
- `src/nla/training/fve.py` (entire — confirmed scalar-mean FVE convention)
- `src/nla/training/grpo.py` (α touchpoints)
- `src/nla/extraction/storage.py` (storage layout — confirmed unscaled)
- `src/nla/extraction/stats.py` (`alpha = p75_norm`, no surprises)
- `scripts/eval/closed_loop_retrieval.py` (entire)
- `scripts/eval/build_v3_scorecard.py` (entire — read-only aggregator, no scaling logic)
- `data/sft/libero_4suite_v3/config.json`, `ar/ar_config.json`, `av/av_config.json` (α=203.977…)
- `data/sft/libero_4suite_v3/metrics.jsonl` last 100 train rows + final eval row
- `data/sft/libero_4suite_v3/v3_scorecard.json`

Probe artifacts:
- `/tmp/probe_scale_bug.py`
- `/tmp/probe_scale_bug.out` (16-row probe with norms / cosines / sign-flip / hidden-flip)
- `/tmp/probe_renorm.out` (64-row probe with per-position-type renorm comparison)
