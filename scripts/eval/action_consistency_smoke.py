"""One-shot smoke test for the action-head consistency kernel.

Phase-1 deliverable for the V4-with-consistency overnight plan
(`docs/sft_plan/09_action_head_lora_phase1.md`).

What it does:

1. Loads a frozen ``Gr00tPolicy`` from a checkpoint (default
   ``checkpoints/GR00T-N1.7-LIBERO/libero_goal``).
2. Loads a V3 ``ActivationReconstructor`` (default
   ``data/sft/libero_4suite_v3/ar``).
3. Builds a ``ReplayManifest`` against the V4 combined activations and the
   LeRobot dataset roots.
4. Picks a single labeled ``image_patch`` row whose suite is ``goal`` from
   ``data/labels/libero_4suite_v4_combined/labels.jsonl``.
5. Runs ``ActionConsistencyKernel.consistency_loss(...)`` once, asserts the
   loss is finite, ``n_rows == 1`` and AR grads are non-None after a single
   ``loss.backward()``.

Pass criterion: prints ``SMOKE_PASS`` and exits 0. On failure prints
``SMOKE_FAIL: <reason>`` and exits 1.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def _fail(msg: str) -> "Never":
    print(f"SMOKE_FAIL: {msg}", flush=True)
    sys.exit(1)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy-path", default="checkpoints/GR00T-N1.7-LIBERO/libero_goal")
    p.add_argument("--embodiment-tag", default="LIBERO_PANDA")
    p.add_argument("--ar-dir", default="data/sft/libero_4suite_v3/ar")
    p.add_argument("--labels-jsonl", default="data/labels/libero_4suite_v4_combined/labels.jsonl")
    p.add_argument("--activations-root", default="data/activations/libero_4suite_v4_combined")
    p.add_argument("--suite", default="goal", help="Suite key matching example_id prefix.")
    p.add_argument(
        "--dataset-root",
        default="third_party/Isaac-GR00T/examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot",
        help="LeRobot dataset root for the picked suite.",
    )
    p.add_argument("--manifest-cache", default=None)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def _pick_row(labels_jsonl: Path, suite: str) -> dict:
    """Walk the labels file and return the first image_patch row in ``suite``."""
    with labels_jsonl.open() as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta = row.get("meta") or {}
            if meta.get("suite") != suite:
                continue
            if meta.get("position_type") != "image_patch":
                continue
            return row
    raise RuntimeError(
        f"No image_patch row with suite={suite!r} found in {labels_jsonl}."
    )


def main() -> None:
    args = _parse_args()

    repo = Path.cwd()
    labels = (repo / args.labels_jsonl).resolve()
    if not labels.exists():
        _fail(f"labels jsonl missing: {labels}")
    ar_dir = (repo / args.ar_dir).resolve()
    if not (ar_dir / "ar_config.json").exists():
        _fail(f"AR checkpoint missing ar_config.json: {ar_dir}")
    activations_root = (repo / args.activations_root).resolve()
    if not (activations_root / "index.jsonl").exists():
        _fail(f"activations root missing index.jsonl: {activations_root}")
    dataset_root = (repo / args.dataset_root).resolve()
    if not dataset_root.exists():
        _fail(f"lerobot dataset root missing: {dataset_root}")

    print(f"[smoke] loading AR checkpoint: {ar_dir}", flush=True)
    from nla.training.checkpoint import load_ar_from_sft

    ar = load_ar_from_sft(ar_dir, device=args.device, freeze=False)
    # We need AR grads (the test verifies they flow), so re-enable
    # requires_grad on the trainable LoRA + head params.
    for p in ar.parameters():
        if p.requires_grad is False and (
            "lora" in p.__class__.__name__.lower() or True
        ):
            # Re-enable everything that load_ar_from_sft might have frozen
            # if freeze=True was used elsewhere; freeze=False above already
            # leaves trainables on, but keep this defensive.
            pass
    n_trainable = sum(p.numel() for p in ar.parameters() if p.requires_grad)
    if n_trainable == 0:
        _fail("AR has 0 trainable parameters; expected LoRA + head to be trainable.")
    print(f"[smoke] AR trainable params: {n_trainable:,}", flush=True)

    print(f"[smoke] building replay manifest from {activations_root}", flush=True)
    from nla.training.replay_manifest import build_replay_manifest

    manifest_cache = (
        Path(args.manifest_cache).resolve()
        if args.manifest_cache
        else activations_root / "aux" / "replay_manifest_smoke.jsonl"
    )
    manifest = build_replay_manifest(
        str(activations_root),
        {args.suite: str(dataset_root)},
        cache_path=manifest_cache,
    )
    if len(manifest) == 0:
        _fail("Replay manifest is empty after filtering to suite.")
    print(f"[smoke] manifest entries: {len(manifest)}", flush=True)

    print(f"[smoke] picking labeled image_patch row from {labels}", flush=True)
    row = _pick_row(labels, args.suite)
    # Walk to a row whose source_example_id is in the manifest (some labeled
    # rows may reference activations the manifest dropped).
    src_id = row["meta"].get("source_example_id") or row["example_id"]
    desc = row["description"]
    pos_type = row["meta"]["position_type"]
    print(
        f"[smoke] picked label_example_id={row['example_id']} "
        f"source_example_id={src_id} position_type={pos_type}",
        flush=True,
    )
    if manifest.get(src_id) is None:
        # Try a few more rows before giving up.
        found = False
        with labels.open() as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                m = r.get("meta") or {}
                if m.get("suite") != args.suite or m.get("position_type") != "image_patch":
                    continue
                cand = m.get("source_example_id") or r["example_id"]
                if manifest.get(cand) is not None:
                    src_id = cand
                    desc = r["description"]
                    pos_type = m["position_type"]
                    found = True
                    print(
                        f"[smoke] resolved to source_example_id={src_id}",
                        flush=True,
                    )
                    break
        if not found:
            _fail(f"no labeled image_patch row in suite {args.suite!r} present in manifest")
    example_id = src_id

    print(f"[smoke] loading Gr00tPolicy from {args.policy_path}", flush=True)
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=str((repo / args.policy_path).resolve()),
        device=args.device,
    )
    policy.model.eval()

    from nla.training.action_head_consistency import (
        ActionConsistencyConfig,
        ActionConsistencyKernel,
        make_lerobot_obs_builder,
    )

    print("[smoke] constructing kernel + obs_builder_factory", flush=True)
    kernel = ActionConsistencyKernel(
        ActionConsistencyConfig(
            weight=1.0,
            every_n_steps=1,
            max_microbatch_per_step=1,
            image_patch_rows_only=True,
        ),
        manifest=manifest,
        policy_loader=lambda: policy,
        obs_builder_factory=lambda p: make_lerobot_obs_builder(
            p, {args.suite: str(dataset_root)}, args.embodiment_tag,
        ),
        ar_module=ar,
        device=args.device,
    )

    print("[smoke] running ensure_loaded() (resolves obs_builder)", flush=True)
    kernel.ensure_loaded()

    print("[smoke] computing consistency_loss(...) on a single row", flush=True)
    loss, diag = kernel.consistency_loss(
        descriptions=[desc],
        example_ids=[example_id],
        position_types=[pos_type],
    )
    print(
        f"[smoke] loss={float(loss.detach().item()):.6f} "
        f"n_rows={diag.n_rows} "
        f"delta_norm={diag.delta_action_norm:.6f} "
        f"cache_hits={diag.baseline_cache_hits} "
        f"cache_misses={diag.baseline_cache_misses}",
        flush=True,
    )

    if diag.n_rows != 1:
        _fail(f"expected n_rows==1, got {diag.n_rows}")
    if not torch.isfinite(loss):
        _fail(f"loss is not finite: {loss}")

    print("[smoke] running loss.backward() to verify AR grads", flush=True)
    loss.backward()
    grad_norm = 0.0
    n_grad_params = 0
    for p in ar.parameters():
        if p.requires_grad and p.grad is not None:
            grad_norm += float(p.grad.detach().float().norm().item()) ** 2
            n_grad_params += 1
    grad_norm = grad_norm ** 0.5
    print(
        f"[smoke] AR grad norm = {grad_norm:.6f} over {n_grad_params} param tensors",
        flush=True,
    )
    if n_grad_params == 0:
        _fail("no AR parameters received a gradient")
    if grad_norm == 0.0:
        _fail("AR grad norm is exactly zero (no signal flowed)")
    if not (grad_norm == grad_norm):  # NaN check
        _fail("AR grad norm is NaN")

    print("SMOKE_PASS", flush=True)


if __name__ == "__main__":
    main()
