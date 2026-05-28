# CLAUDE.md — nla-groot

Repo-wide guidance loaded into every conversation. Keep it short; expand the per-area docs (`docs/sft_plan/`, `docs/grpo/`, etc.) for depth.

## Fast inspection iteration

For "look at what the checkpoint does" iteration — viewing rollouts, comparing matched vs mismatched intent on the same scene, eyeballing init states — use the warm-REPL + long-lived steer-server stack. It does **not** speed up training; it speeds up the post-training inspection loop (drops "look at 5 init states" from ~50s to ~0.5s, "one rollout" from ~50s cold to ~20s warm).

### Components

| What | Where | Purpose |
|------|-------|---------|
| Warm REPL | [scripts/eval/play.sh](scripts/eval/play.sh) → [scripts/eval/play_repl.py](scripts/eval/play_repl.py) | `ipython -i` in the libero venv with pre-imports + cached `LiberoEnv` per task. Auto-detects the live policy server and opens a `PolicyClient`. |
| Long-lived steer server | [scripts/eval/launch_steer_server.sh](scripts/eval/launch_steer_server.sh) | Launches `NlaSteerGr00tPolicy` as a ZMQ daemon under nohup. Server cost (model load + bind) is paid once; rollouts then cost only sim wall time. PID + log at `data/sft/<run>/steer_server_logs/`. |
| Server status / cleanup | [scripts/eval/steer_server_status.sh](scripts/eval/steer_server_status.sh) | Lists all servers from pidfiles, marks live/stale. `--check` pings; `--clean` removes stale pidfiles. |
| LIBERO init-state cache | [data/libero_cache/](data/libero_cache/) (built by [scripts/eval/cache_libero_init_states.py](scripts/eval/cache_libero_init_states.py)) | Per task: `init_states.npy` (LIBERO's deterministic pool), `meta.json` (bddl path, language), `previews/init_<id>.png`. 40 tasks across goal/object/spatial/10 ≈ 142 MB. |
| Cache loader | [src/nla/eval/steerability/state_cache.py](src/nla/eval/steerability/state_cache.py) | `load_task_meta`, `load_init_states`, `apply_init_state(env, state)`, `preview_path`, `list_cached_tasks`. |

### REPL helpers exposed by `play_repl.py`

```python
view(task, init_id=0)                                # (256, 256, 3) uint8 — from cached PNG when available
show(arr_or_path, name=None)                         # save to data/play_out/<name>.png
play(task, init_id=0, steer_text=None,
     max_steps=200, save_video=True,
     steer_disabled=False)                           # rollout vs live server; writes MP4
info()                                               # cheatsheet + live server port
tasks                                                # list of cached tasks
client                                               # PolicyClient → auto-detected live server
```

### Quick reference

```bash
# Check server state, clean stale pidfiles
scripts/eval/steer_server_status.sh --check --clean

# Bring a server up (if none live)
scripts/eval/launch_steer_server.sh --sft-dir data/sft/<run> -- \
    --embodiment-tag LIBERO_PANDA --steer-text-file <bullets.txt>

# Open the REPL
scripts/eval/play.sh
# Or one-shot
scripts/eval/play.sh -c "play('put_the_bowl_on_the_plate')"

# (Re)build the init-state cache (CPU only via osmesa, won't touch the GPU)
LIBERO_PY=third_party/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python
PYTHONPATH=src "$LIBERO_PY" scripts/eval/cache_libero_init_states.py \
    --suite libero_goal --suite libero_object \
    --suite libero_spatial --suite libero_10 \
    --render-previews
```

## LIBERO render backend — osmesa pinning

Any script that imports `from gr00t.eval.sim.LIBERO.libero_env import LiberoEnv` (or anything transitively under `libero.libero.envs`) must set **both** environment variables before the import:

```python
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
```

Why: EGL device-display init fails on this host. [third_party/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_env.py](third_party/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_env.py) calls `os.environ.setdefault("MUJOCO_GL", "egl")` and `setdefault("PYOPENGL_PLATFORM", "egl")` at module load. Setting only `MUJOCO_GL=osmesa` leaves `PYOPENGL_PLATFORM` for libero_env to silently pin to `egl`; rendering then fails inside robosuite's mujoco OSMesa context with `ImportError: Cannot use OSMesa rendering platform. The PYOPENGL_PLATFORM environment variable is set to 'egl'`. Popping `PYOPENGL_PLATFORM` is also not enough — libero_env will `setdefault` it back to `egl`.

Reference implementations that get this right: [scripts/eval/play.sh](scripts/eval/play.sh), [scripts/eval/play_repl.py](scripts/eval/play_repl.py), [scripts/eval/cache_libero_init_states.py](scripts/eval/cache_libero_init_states.py).

## Training cadence (current v7)

v7 SFT runs at **~6.7 s/step on a single H100 PCIe** → 4000 steps ≈ **7.5 hours**. The slow knobs are `action_consistency_every_n_steps=1` (frozen GR00T forward every step) and `eval_closed_loop=True` (teacher-free closed-loop in the validation pass).

If iteration on the SFT recipe itself is the bottleneck (not inspection):
- Screening configs first: lower `total_steps`, `action_consistency_every_n_steps≥4`, no closed-loop eval. Only run the full v7 budget when a screening run looks promising.
- Surrogate eval: cache GR00T policy forwards on a fixed held-out batch, use action-MSE against them as in-loop signal. Replaces sim rollouts inside the loop (sim eval is otherwise ~50 GPU-hr).
- FSDP across 2–4 H100s gets ~3× on this loss stack (the `action_consistency` forward shards cleanly).
- Skip GRPO until SFT alone moves axis-2 (judge grounding). Stage 0 showed AR injection is inert at trained α, so GRPO won't rescue an SFT that didn't shift grounding.
