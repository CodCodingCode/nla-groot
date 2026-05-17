# LIBERO Goal NLA Steering Pilot — Closed-Loop Sim Results

End-to-end validation that an NLA backbone-feature steer trained on a tiny
LIBERO Goal labeled set can produce **measurable behavioral change** in
closed-loop simulation against the official `nvidia/GR00T-N1.7-LIBERO/libero_goal`
checkpoint, served through `scripts/eval/run_gr00t_server_nla_steer.py`.

## Setup

| Component | Value |
|-----------|-------|
| Policy   | `checkpoints/GR00T-N1.7-LIBERO/libero_goal` (LIBERO_PANDA embodiment) |
| AR       | `data/sft/libero_goal_pilot/ar` (243 LIBERO Goal labels, ~300 SFT steps) |
| Sim env  | `libero_sim/put_the_bowl_on_the_plate` |
| Sim venv | `third_party/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv` |
| Renderer | `MUJOCO_GL=osmesa` (system EGL is Mesa, not NVIDIA) |
| Seed     | `0` (deterministic; identical reset across runs) |
| n_action_steps | 8 |
| max_episode_steps | 120 (single-shot runs) / 200 (3-episode runs) |

## Steer prompts

Both prompts are 5–6 bullets in the same format the AR was trained on.

- `/tmp/steer_text_a.txt` – matches the env: bowl on plate, plate to right of bowl.
- `/tmp/steer_text_b.txt` – contradicts the env: wine bottle, wooden cabinet, wine rack.

## Conditions and results

| Condition | AR | Placement | Blend | Episodes | Successes | Avg chunks |
|-----------|----|-----------|-------|----------|-----------|------------|
| Baseline (vanilla policy)            | – | – | –   | 3 | 3/3 | 9.7  |
| Light A (matching prompt)            | ✓ | `image_patch`     | 1.0 | 1 | 1/1 | 10   |
| Light B (mismatched prompt)          | ✓ | `image_patch`     | 1.0 | 1 | 1/1 | 10   |
| **Aggressive B (mismatched + all-tokens)** | ✓ | `image_patch_all` | 1.5 | 3 | **0/3** | 25 (capped) |

`Avg chunks * n_action_steps` gives sim steps. Aggressive condition consistently
runs out the budget without solving — a robust behavioral failure rather than a
single bad seed.

The single-image-token (`image_patch`) steer at `blend=1.0` is too weak to
overcome the strong domain-tuned policy on this task, even with a contradictory
prompt — exactly the same finding as the offline action-delta probe in
`scripts/eval/nla_steer_quant_probe.py`. Spreading the steer across **every**
image patch token at `blend=1.5` is what flips the policy from "always solves" to
"never solves".

## Artifacts

- Servers logs: `logs/libero_goal_pilot/server_{baseline,steer_a,steer_b,steer_b_agg}.log`
- Rollout logs: `logs/libero_goal_pilot/rollout_{baseline,steer_a,steer_b,steer_b_agg}.log`
- Videos (file suffix `_s1` = success, `_s0` = failure):
  - `data/sim_rollouts/libero_goal_pilot/baseline/*.mp4` — 3× `_s1`
  - `data/sim_rollouts/libero_goal_pilot/steer_a/*.mp4` — 1× `_s1`
  - `data/sim_rollouts/libero_goal_pilot/steer_b/*.mp4` — 1× `_s1`
  - `data/sim_rollouts/libero_goal_pilot/steer_b_agg/*.mp4` — 3× `_s0`

## Reproducing

Two terminals.

### Terminal 1 — steered policy server (main `.venv`)

```bash
cd /home/ubuntu/nla-groot
source .venv/bin/activate
PYTHONPATH=src python scripts/eval/run_gr00t_server_nla_steer.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_goal \
  --embodiment-tag LIBERO_PANDA \
  --use-sim-policy-wrapper \
  --ar-dir data/sft/libero_goal_pilot/ar \
  --steer-text-file /tmp/steer_text_b.txt \
  --placement image_patch_all \
  --blend 1.5 \
  --port 5555
```

Wait until `0.0.0.0:5555` is listening (≈90–120 s; nohup output may be buffered,
check `ss -ltn`).

### Terminal 2 — LIBERO sim client (`libero_uv` venv)

```bash
cd /home/ubuntu/nla-groot/third_party/Isaac-GR00T
source gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/activate
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python gr00t/eval/rollout_policy.py \
  --policy-client-host localhost --policy-client-port 5555 \
  --env-name libero_sim/put_the_bowl_on_the_plate \
  --n-episodes 3 --n-envs 1 --max-episode-steps 200 --n-action-steps 8 \
  --seed 0 \
  --video-dir /home/ubuntu/nla-groot/data/sim_rollouts/libero_goal_pilot/steer_b_agg
```

For the baseline, drop `--ar-dir` / `--steer-text-file` from the server command
and point `--video-dir` at `…/baseline`.

## Caveats

- The AR is trained on 243 labels for ~300 steps; final val cosine is ~0.44.
  This is a *plumbing pilot*, not a quantitative claim about NLA quality.
- Only one task tried so far (`put_the_bowl_on_the_plate`). The behavior shift
  shows the steer can overpower the policy; it does **not** yet show that the
  policy was steered toward something semantically tied to the wine-bottle
  prompt. To make that claim we need a multi-task suite (e.g. LIBERO 10 / Long)
  and per-prompt success/affordance scoring, not just on-task success rate.
- We rely on `MUJOCO_GL=osmesa` (CPU rendering). Each rollout is ≈1 min/episode;
  GPU EGL would speed this up roughly 10×.
