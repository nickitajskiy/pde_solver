"""Дополнительная проверка GUI: PDE_GUI_TESTS=1, требуется графический сеанс."""

import os
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get("PDE_GUI_TESTS") == "1", "GUI проверяется по запросу")
class GuiTests(unittest.TestCase):
    def test_calculations_and_error_recovery(self):
        from main import create_app

        root = create_app()
        try:
            root.update()
            with patch("main.messagebox.showerror") as errors:
                root.heat_tab.calculate(compare=True)
                root.update()
                self.assertEqual(len(root.heat_tab.solutions), 3)
                root.heat_tab.layer.set(0)
                root.update()
                self.assertIn("Точное", root.heat_tab.plot.axes.get_legend_handles_labels()[1])
                root.heat_tab.study()
                root.wave_tab.calculate()
                root.laplace_tab.calculate()
                root.laplace_tab.calculate()
                root.update()
                self.assertEqual(len(root.laplace_tab.plot.figure.axes), 2)
                errors.assert_not_called()
                entry = root.heat_tab.entries["h"]
                entry.delete(0, "end")
                entry.insert(0, "0")
                root.heat_tab.calculate()
                errors.assert_called_once()
                self.assertFalse(root.heat_tab.solutions)
                entry.delete(0, "end")
                entry.insert(0, "0.05")
                root.heat_tab.calculate()
                self.assertEqual(len(root.heat_tab.solutions), 1)
        finally:
            root.destroy()
