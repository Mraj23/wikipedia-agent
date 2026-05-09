"""Best-effort Weights & Biases logging for the faithfulness GRPO trainer.

Design constraints:
    - Safe to import even when `wandb` is not installed.
    - Safe to call when `WANDB_API_KEY` is unset or `--no-wandb` is passed.
    - Never raises into the training loop; failures degrade to a logger
      warning and a no-op handle.

The trainer keeps writing `train_log.jsonl` and `eval_log.jsonl` regardless,
so wandb is purely a viewer. JSONL stays the source of truth.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class WandbHandle:
    enabled: bool
    run: Any = None

    def log(self, payload: Dict[str, Any], step: Optional[int] = None) -> None:
        if not self.enabled or self.run is None:
            return
        try:
            self.run.log(payload, step=step)
        except Exception as exc:  # noqa: BLE001
            logger.warning("wandb.log failed: %s", exc)

    def finish(self) -> None:
        if not self.enabled or self.run is None:
            return
        try:
            self.run.finish()
        except Exception as exc:  # noqa: BLE001
            logger.warning("wandb.finish failed: %s", exc)


def init_wandb(
    *,
    enabled: bool,
    project: Optional[str],
    run_name: Optional[str],
    config: Dict[str, Any],
    log_dir: Optional[str] = None,
) -> WandbHandle:
    if not enabled:
        return WandbHandle(enabled=False)
    if not os.environ.get("WANDB_API_KEY"):
        logger.info("WANDB_API_KEY not set; wandb logging disabled")
        return WandbHandle(enabled=False)
    try:
        import wandb  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        logger.info("wandb not installed (%s); wandb logging disabled", exc)
        return WandbHandle(enabled=False)
    try:
        run = wandb.init(
            project=project or "faithfulness",
            name=run_name,
            config=config,
            dir=log_dir,
            reinit=True,
        )
        return WandbHandle(enabled=True, run=run)
    except Exception as exc:  # noqa: BLE001
        logger.warning("wandb.init failed: %s; wandb logging disabled", exc)
        return WandbHandle(enabled=False)


_SCALAR_TYPES = (int, float, bool)


def _flatten(payload: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten one level of nested dicts, preserve scalars, drop unloggable values."""
    out: Dict[str, Any] = {}
    for k, v in payload.items():
        key = f"{prefix}{k}"
        if isinstance(v, _SCALAR_TYPES) or v is None:
            out[key] = v
        elif isinstance(v, dict):
            for sk, sv in v.items():
                if isinstance(sv, _SCALAR_TYPES):
                    out[f"{key}/{sk}"] = sv
        # silently skip strings, lists, and other non-scalar entries; the
        # JSONL log keeps them.
    return out


def log_train_step(handle: WandbHandle, entry: Dict[str, Any]) -> None:
    if not handle.enabled:
        return
    step = entry.get("step")
    payload = _flatten({k: v for k, v in entry.items() if k != "sample"}, prefix="train/")
    handle.log(payload, step=step)


def log_fast_eval(handle: WandbHandle, entry: Dict[str, Any]) -> None:
    if not handle.enabled:
        return
    step = entry.get("step")
    payload = _flatten(entry, prefix="eval/")
    handle.log(payload, step=step)
