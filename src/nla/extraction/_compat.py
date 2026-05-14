"""Runtime compatibility patches for the vendored Isaac-GR00T code.

These patches handle drift between the released GR00T-N1.7-3B checkpoint
and the current Isaac-GR00T source tree. Each patch is small, targeted,
and idempotent — call ``apply_all()`` once before instantiating Gr00tPolicy.

Patches currently applied
-------------------------
1. ``backbone_model_type`` -> ``model_type`` rename.
   The shipped checkpoint serialises ``backbone_model_type`` in its
   ``processor_config.json``, but ``Gr00tN1d7Processor.__init__`` accepts
   only ``model_type``. We wrap ``__init__`` to rename the kwarg if both
   are present we trust the legacy key.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCHED = False


def apply_all() -> None:
    """Apply every compatibility patch. Safe to call multiple times."""
    global _PATCHED
    if _PATCHED:
        return
    _patch_processor_legacy_kwargs()
    _PATCHED = True


def _patch_processor_legacy_kwargs() -> None:
    """Rename ``backbone_model_type`` -> ``model_type`` at processor init."""
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    original_init = Gr00tN1d7Processor.__init__

    def patched_init(self, *args, **kwargs):
        if "backbone_model_type" in kwargs:
            legacy = kwargs.pop("backbone_model_type")
            # Don't clobber an explicitly-set model_type with a legacy value.
            kwargs.setdefault("model_type", legacy)
            logger.debug(
                "Renamed legacy processor kwarg 'backbone_model_type' -> 'model_type' (=%s)",
                kwargs["model_type"],
            )
        return original_init(self, *args, **kwargs)

    # Avoid double-wrapping if apply_all is called twice through different paths.
    if getattr(Gr00tN1d7Processor.__init__, "__nla_patched__", False):
        return
    patched_init.__nla_patched__ = True  # type: ignore[attr-defined]
    Gr00tN1d7Processor.__init__ = patched_init  # type: ignore[method-assign]
