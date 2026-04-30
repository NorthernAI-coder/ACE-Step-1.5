"""Unit tests for the turbo timestep-schedule helper (#1156 PR-D).

Pins the three branches of ``build_turbo_timestep_schedule``:
* explicit ``timesteps`` → snap to nearest valid value, drop trailing
  zeros, clamp length to 20
* ``infer_steps`` set → linspace + shift transform
* both unset → fixed shift table (snapped to a valid shift)
"""

import unittest

from acestep.models.common.turbo_schedule import (
    TURBO_SHIFT_TIMESTEPS,
    VALID_TURBO_SHIFTS,
    VALID_TURBO_TIMESTEPS,
    build_turbo_timestep_schedule,
)


class BuildTurboTimestepScheduleTests(unittest.TestCase):

    # -- Case 1: explicit timesteps --------------------------------------

    def test_explicit_timesteps_passthrough_when_already_valid(self):
        ts = [1.0, 0.875, 0.75, 0.125]
        out = build_turbo_timestep_schedule(timesteps=ts)
        self.assertEqual(out, ts)

    def test_explicit_timesteps_drops_trailing_zeros(self):
        out = build_turbo_timestep_schedule(timesteps=[1.0, 0.875, 0.0, 0.0])
        self.assertEqual(out, [1.0, 0.875])

    def test_explicit_timesteps_snaps_to_valid(self):
        # 0.5001 should snap to the nearest VALID_TURBO_TIMESTEPS entry.
        out = build_turbo_timestep_schedule(timesteps=[1.0, 0.5001, 0.13])
        self.assertEqual(len(out), 3)
        for v in out:
            self.assertIn(v, VALID_TURBO_TIMESTEPS)

    def test_explicit_timesteps_clamped_to_20(self):
        ts = [1.0 - i * 0.04 for i in range(30)]  # 30 entries
        out = build_turbo_timestep_schedule(timesteps=ts)
        self.assertEqual(len(out), 20)

    def test_empty_timesteps_falls_through_to_shift_table(self):
        # All zeros → empty after stripping → shift fallback.
        out = build_turbo_timestep_schedule(timesteps=[0.0, 0.0], shift=3.0)
        self.assertEqual(out, list(TURBO_SHIFT_TIMESTEPS[3.0]))

    # -- Case 2: variable infer_steps ------------------------------------

    def test_infer_steps_linspace_no_shift(self):
        out = build_turbo_timestep_schedule(infer_steps=4, shift=1.0)
        # 1.0, 0.75, 0.5, 0.25 (no shift transform).
        self.assertEqual(out, [1.0, 0.75, 0.5, 0.25])

    def test_infer_steps_with_shift_transforms(self):
        out = build_turbo_timestep_schedule(infer_steps=4, shift=3.0)
        # Shift transform: t' = 3t / (1 + 2t) — strictly less than t for t<1
        # except at t=1.0.  All values must be in [0, 1].
        self.assertEqual(len(out), 4)
        self.assertAlmostEqual(out[0], 1.0)
        for v in out:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_infer_steps_clamped_to_20(self):
        out = build_turbo_timestep_schedule(infer_steps=30, shift=1.0)
        self.assertEqual(len(out), 20)

    # -- Case 3: shift fallback ------------------------------------------

    def test_default_shift_uses_lookup_table(self):
        out = build_turbo_timestep_schedule(shift=3.0)
        self.assertEqual(out, list(TURBO_SHIFT_TIMESTEPS[3.0]))

    def test_invalid_shift_snaps_to_nearest(self):
        out = build_turbo_timestep_schedule(shift=2.7)
        # 2.7 nearest to 3.0 in VALID_TURBO_SHIFTS.
        self.assertEqual(out, list(TURBO_SHIFT_TIMESTEPS[3.0]))

    def test_all_valid_shifts_produce_8_step_schedules(self):
        for s in VALID_TURBO_SHIFTS:
            out = build_turbo_timestep_schedule(shift=s)
            self.assertEqual(len(out), 8, f"shift={s} should give 8 steps")
            self.assertAlmostEqual(out[0], 1.0)


if __name__ == "__main__":
    unittest.main()
