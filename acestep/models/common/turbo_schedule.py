"""Turbo-variant timestep schedule construction (#1156 PR-D).

Turbo / XL-turbo DiT variants are CFG-distilled and use a fixed
``fix_nfe=8`` table mapped onto discrete shift values (1.0, 2.0, 3.0).
The regular ``generate_audio`` path on those variants computes the
schedule inline using the constants here; the new flow-edit delegate
needs the same logic, so we centralise it.

The shared :func:`flow_edit_helpers.build_timestep_schedule` accepts a
``timesteps`` tensor and pads a trailing zero, so callers just hand it
the list this module returns.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

from loguru import logger

# Discrete shift values supported by the turbo distillation.
VALID_TURBO_SHIFTS: List[float] = [1.0, 2.0, 3.0]

# Union of the 8-step schedules for shift ∈ {1, 2, 3} — 20 unique values.
# Used to snap user-supplied custom timesteps to the nearest trained
# value (training only saw these specific schedules).
VALID_TURBO_TIMESTEPS: List[float] = [
    1.0, 0.9545454545454546, 0.9333333333333333, 0.9, 0.875,
    0.8571428571428571, 0.8333333333333334, 0.7692307692307693, 0.75,
    0.6666666666666666, 0.6428571428571429, 0.625, 0.5454545454545454,
    0.5, 0.4, 0.375, 0.3, 0.25, 0.2222222222222222, 0.125,
]

# Fixed 8-step schedules per shift (excluding the trailing 0 — callers
# add it themselves where needed).
TURBO_SHIFT_TIMESTEPS = {
    1.0: [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
    2.0: [
        1.0, 0.9333333333333333, 0.8571428571428571, 0.7692307692307693,
        0.6666666666666666, 0.5454545454545454, 0.4, 0.2222222222222222,
    ],
    3.0: [
        1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75,
        0.6428571428571429, 0.5, 0.3,
    ],
}


def _snap_timesteps(values: Sequence[float]) -> List[float]:
    """Map each value to the nearest entry in ``VALID_TURBO_TIMESTEPS``."""
    snapped = [min(VALID_TURBO_TIMESTEPS, key=lambda x: abs(x - t)) for t in values]
    if list(values) != snapped:
        logger.warning(
            "[turbo_schedule] timesteps mapped to nearest valid values: %s -> %s",
            list(values), snapped,
        )
    return snapped


def build_turbo_timestep_schedule(
    *,
    infer_steps: Optional[int] = None,
    shift: float = 3.0,
    timesteps: Optional[Union[Sequence[float], "torch.Tensor"]] = None,
) -> List[float]:
    """Return the turbo timestep schedule (without the trailing 0).

    Mirrors the inline logic in turbo / xl_turbo ``generate_audio``:

    1. If ``timesteps`` is supplied, drop trailing zeros, clamp length
       to 20, and snap each value to ``VALID_TURBO_TIMESTEPS``.
    2. Else if ``infer_steps`` is set, build a linspace-style schedule
       of length ``min(infer_steps, 20)`` and apply the shift transform.
    3. Else use the fixed ``TURBO_SHIFT_TIMESTEPS[round_to_valid(shift)]``.
    """
    # Case 1: explicit timesteps.
    if timesteps is not None:
        try:
            ts_list = timesteps.tolist()  # type: ignore[union-attr]
        except AttributeError:
            ts_list = list(timesteps)
        # Drop trailing zeros (the loop appends one itself).
        while ts_list and ts_list[-1] == 0:
            ts_list.pop()
        if not ts_list:
            logger.warning(
                "[turbo_schedule] empty timesteps after stripping zeros; "
                "falling back to shift=%.1f schedule",
                shift,
            )
        else:
            if len(ts_list) > 20:
                logger.warning(
                    "[turbo_schedule] timesteps length=%d exceeds maximum 20; "
                    "truncating.",
                    len(ts_list),
                )
                ts_list = ts_list[:20]
            return _snap_timesteps(ts_list)

    # Case 2: variable-step linspace + shift transform (matches the
    # per-step schedule in turbo's ``generate_audio``).
    if infer_steps is not None and infer_steps > 0:
        n = min(int(infer_steps), 20)
        if n != int(infer_steps):
            logger.warning(
                "[turbo_schedule] infer_steps=%d exceeds maximum 20, clamping.",
                infer_steps,
            )
        raw = [1.0 - i / n for i in range(n)]
        if shift != 1.0:
            raw = [shift * t / (1.0 + (shift - 1.0) * t) for t in raw]
        return raw

    # Case 3: fixed lookup by shift.
    snapped_shift = min(VALID_TURBO_SHIFTS, key=lambda x: abs(x - shift))
    if snapped_shift != shift:
        logger.warning(
            "[turbo_schedule] shift=%.2f not supported, rounded to %.1f.",
            shift, snapped_shift,
        )
    return list(TURBO_SHIFT_TIMESTEPS[snapped_shift])
