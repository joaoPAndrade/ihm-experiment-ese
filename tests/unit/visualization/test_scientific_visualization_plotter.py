import os
import tempfile

import numpy as np

from src.visualization.scientific_visualization_plotter import (
    ScientificVisualizationPlotter,
)


class TestScientificVisualizationPlotter:
    def _test_plotting_generation(self) -> None:
        plotter = ScientificVisualizationPlotter(dpi=80)
        with tempfile.TemporaryDirectory() as temp_dir:
            ih = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
            ihm = {"kDN": np.array([0.1, 0.2, 0.3, 0.4, 0.5]), "N2": np.array([0.2, 0.3, 0.4, 0.5, 0.6])}
            corr = {"kDN": (0.85, 0.01, True), "N2": (0.72, 0.04, True)}
            paths = plotter.execute(ih, ihm, corr, temp_dir)
            assert len(paths) == 2
            for path in paths:
                assert os.path.isfile(path)
                assert os.path.getsize(path) > 0

    def _test_directory_creation(self) -> None:
        plotter = ScientificVisualizationPlotter(dpi=80)
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = os.path.join(temp_dir, "nested", "plots")
            ih = np.zeros(3)
            ihm = {"kDN": np.zeros(3)}
            corr = {"kDN": (0.0, 1.0, False)}
            paths = plotter.execute(ih, ihm, corr, nested_dir)
            assert os.path.isdir(nested_dir)
            assert len(paths) == 2

    def execute(self) -> None:
        self._test_plotting_generation()
        self._test_directory_creation()

    def test_scientific_visualization_plotter(self) -> None:
        self.execute()
