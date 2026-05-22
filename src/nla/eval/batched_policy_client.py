"""ZMQ client with batched ``get_action`` for parallel LIBERO rollouts."""

from __future__ import annotations

from typing import Any

from gr00t.policy.server_client import PolicyClient


class BatchedPolicyClient(PolicyClient):
    """Policy client that can call ``get_action_batch`` on :class:`NlaPolicyServer`."""

    def get_action_batch(
        self,
        observations: list[dict[str, Any]],
        options_list: list[dict[str, Any] | None] | None = None,
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Run one GPU forward for ``len(observations)`` parallel envs."""
        if not observations:
            return []
        if len(observations) == 1:
            return [
                self.get_action(observations[0], (options_list or [None])[0]),
            ]
        n = len(observations)
        opts = options_list if options_list is not None else [None] * n
        response = self.call_endpoint(
            "get_action_batch",
            {"observations": observations, "options_list": opts},
        )
        if not isinstance(response, list):
            raise RuntimeError(
                f"get_action_batch expected list response, got {type(response)!r}"
            )
        out: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for item in response:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                out.append((item[0], item[1]))
            else:
                raise RuntimeError(
                    f"unexpected batch item shape: {type(item)!r}"
                )
        if len(out) != n:
            raise RuntimeError(
                f"get_action_batch returned {len(out)} items, expected {n}"
            )
        return out
