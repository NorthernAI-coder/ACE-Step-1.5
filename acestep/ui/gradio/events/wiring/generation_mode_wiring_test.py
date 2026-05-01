"""Tests for generation mode repaint UI wiring helpers."""

import tempfile
import unittest

from acestep.ui.gradio.events.wiring.generation_mode_wiring import (
    _on_source_session_dir_change,
)


class RepaintModeChoiceTests(unittest.TestCase):
    """Verify repaint mode choices remain stable."""

    def test_unknown_repaint_mode_value_resets_without_session_folder(self):
        """Unsupported repaint mode values should reset to auto."""
        update = _on_source_session_dir_change("", "unsupported")

        self.assertEqual(["auto", "conservative", "balanced", "aggressive"], update["choices"])
        self.assertEqual("auto", update["value"])

    def test_session_folder_keeps_standard_repaint_choices(self):
        """Generated sessions are hidden state, not a separate user-facing mode."""
        with tempfile.TemporaryDirectory() as tmp:
            update = _on_source_session_dir_change(tmp, "balanced")

        self.assertEqual(["auto", "conservative", "balanced", "aggressive"], update["choices"])
        self.assertEqual("balanced", update["value"])


if __name__ == "__main__":
    unittest.main()
