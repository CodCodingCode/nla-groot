"""Policy server with batched ``get_action`` for parallel LIBERO rollouts."""

from __future__ import annotations

from typing import Any

import numpy as np

from gr00t.policy.policy import BasePolicy
from gr00t.policy.server_client import EndpointHandler, PolicyServer

from nla.eval.steerability.obs_batching import stack_nested_observations
def _build_batched_options(
    options_list: list[dict[str, Any] | None],
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Merge per-row options into one dict for :class:`NlaSteerGr00tPolicy`."""
    merged: dict[str, Any] = {}
    steer_rows: list[np.ndarray] = []
    spec = None
    for opt in options_list:
        opt = opt or {}
        h = None
        for k in ("steer_h", b"steer_h"):
            if k in opt and opt[k] is not None:
                raw = opt[k]
                if isinstance(raw, np.ndarray):
                    h = raw.astype(np.float32, copy=False)
                else:
                    h = np.asarray(raw, dtype=np.float32)
                if h.ndim == 2 and h.shape[0] == 1:
                    h = h.squeeze(0)
                break
        if h is None:
            raise ValueError(
                "get_action_batch requires steer_h in each row's options"
            )
        steer_rows.append(h)
        if spec is None:
            for sk in ("steer_spec", b"steer_spec"):
                if sk in opt and opt[sk] is not None:
                    spec = opt[sk]
                    break
    if len(steer_rows) != batch_size:
        raise ValueError(
            f"options_list length {len(steer_rows)} != batch {batch_size}"
        )
    merged["steer_h_batch"] = np.stack(steer_rows, axis=0)
    if spec is not None:
        merged["steer_spec"] = spec
    return merged


class NlaPolicyServer(PolicyServer):
    """Extends :class:`PolicyServer` with ``get_action_batch`` for sim-GRPO."""

    def __init__(self, policy: BasePolicy, **kwargs: Any) -> None:
        super().__init__(policy, **kwargs)
        self.register_endpoint(
            "get_action_batch",
            self._handle_get_action_batch,
            requires_input=True,
        )

    def _handle_get_action_batch(
        self,
        *,
        observations: list[dict[str, Any]],
        options_list: list[dict[str, Any] | None] | None = None,
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        if not observations:
            raise ValueError("observations must be non-empty")
        n = len(observations)
        opts = options_list if options_list is not None else [None] * n
        if len(opts) != n:
            raise ValueError(
                f"len(options_list)={len(opts)} != len(observations)={n}"
            )
        if n == 1:
            action, info = self.policy.get_action(observations[0], opts[0])
            return [(action, info)]

        batched_obs = stack_nested_observations(observations)
        batched_opts = _build_batched_options(opts, batch_size=n)
        action, info = self.policy.get_action(batched_obs, batched_opts)
        return _unbatch_action_results(action, info, batch_size=n)


def _unbatch_action_results(
    action: dict[str, Any],
    info: dict[str, Any],
    *,
    batch_size: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Split batched action dict into per-env (action, info) pairs."""
    per_action: list[dict[str, Any]] = [
        {} for _ in range(batch_size)
    ]
    for key, val in action.items():
        arr = np.asarray(val)
        if arr.ndim < 1:
            raise ValueError(f"action[{key!r}] has no batch dimension")
        if arr.shape[0] != batch_size:
            raise ValueError(
                f"action[{key!r}] batch {arr.shape[0]} != {batch_size}"
            )
        for i in range(batch_size):
            per_action[i][key] = arr[i]
    info_list = info if isinstance(info, list) else [info] * batch_size
    if len(info_list) != batch_size:
        info_list = [info] * batch_size
    return list(zip(per_action, info_list))
