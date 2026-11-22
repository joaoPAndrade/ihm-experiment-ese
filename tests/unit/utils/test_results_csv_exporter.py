import os
import tempfile

import numpy as np

from src.utils.results_csv_exporter import ResultsCsvExporter


class TestResultsCsvExporter:
    def _test_csv_export_content(self) -> None:
        exporter = ResultsCsvExporter(delimiter=";")
        with tempfile.TemporaryDirectory() as temp_dir:
            dummy_data = {
                "ih": np.array([0.1, 0.9]),
                "ihm": {"kDN": np.array([0.2, 0.8]), "TD": np.array([0.3, 0.7])},
                "labels": np.array([0, 1]),
                "correlations": {"kDN": (0.85, 0.01, True), "TD": (0.65, 0.04, True)},
                "model_names": ["ModelA", "ModelB"],
                "base_f1": np.array([0.70, 0.75]),
                "filt_f1": np.array([0.80, 0.82]),
                "hypothesis_res": {
                    "p_value": 0.002,
                    "is_normal": 1.0,
                    "shapiro_p": 0.45,
                    "effect_size": 1.5,
                    "significant": 1.0,
                },
            }
            exported_paths = exporter.execute(dummy_data, temp_dir)
            assert len(exported_paths) == 4
            for path in exported_paths:
                assert os.path.isfile(path)
                with open(path, "r", encoding="utf-8") as stream:
                    lines = stream.readlines()
                    assert len(lines) >= 2
                    assert ";" in lines[0]

    def _test_output_directory_creation(self) -> None:
        exporter = ResultsCsvExporter()
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = os.path.join(temp_dir, "csv_results", "run_1")
            dummy_data = {
                "ih": np.array([0.5]),
                "ihm": {"kDN": np.array([0.5])},
                "labels": np.array([0]),
                "correlations": {"kDN": (0.0, 1.0, False)},
                "base_f1": np.array([0.5]),
                "filt_f1": np.array([0.5]),
                "hypothesis_res": {
                    "p_value": 1.0,
                    "is_normal": 0.0,
                    "shapiro_p": 0.0,
                    "effect_size": 0.5,
                    "significant": 0.0,
                },
            }
            paths = exporter.execute(dummy_data, nested_dir)
            assert os.path.isdir(nested_dir)
            assert len(paths) == 4

    def execute(self) -> None:
        self._test_csv_export_content()
        self._test_output_directory_creation()

    def test_results_csv_exporter(self) -> None:
        self.execute()
