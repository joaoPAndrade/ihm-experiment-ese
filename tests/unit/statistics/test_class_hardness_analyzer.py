import os
import tempfile

import numpy as np

from src.statistics.class_hardness_analyzer import ClassHardnessAnalyzer


class TestClassHardnessAnalyzer:
    def _test_class_records_and_brackets(self) -> None:
        analyzer = ClassHardnessAnalyzer()
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int64)
        ih = np.array([0.2, 0.4, 0.3, 0.5, 0.1, 0.8, 0.9, 0.7, 0.6, 0.85, 0.75], dtype=np.float64)
        ihm = {"kDN": np.array([0.1] * 5 + [0.8] * 6), "DCP": np.array([0.0] * 5 + [0.5] * 6), "N2_IHM": np.array([0.2] * 5 + [0.7] * 6)}
        mapping = {0: "Species_Easy", 1: "Species_Hard"}
        records = analyzer._compute_class_records(labels, ih, ihm, mapping)
        assert len(records) == 2
        assert records[0]["quantidade_amostras"] == 5
        assert records[1]["quantidade_amostras"] == 6
        assert records[0]["nome_especie"] == "Species_Easy"
        assert records[1]["nome_especie"] == "Species_Hard"
        assert records[0]["media_ground_truth_ih"] < records[1]["media_ground_truth_ih"]
        brackets = analyzer._compute_size_brackets(records)
        assert len(brackets) == 1
        assert brackets[0]["quantidade_classes"] == 2

    def _test_execute_files_creation(self) -> None:
        analyzer = ClassHardnessAnalyzer()
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int64)
        ih = np.array([0.2, 0.4, 0.3, 0.5, 0.1, 0.8, 0.9, 0.7, 0.6, 0.85, 0.75], dtype=np.float64)
        ihm = {"kDN": np.array([0.1] * 5 + [0.8] * 6), "DCP": np.array([0.0] * 5 + [0.5] * 6), "N2_IHM": np.array([0.2] * 5 + [0.7] * 6)}
        mapping = {0: "Species_Easy", 1: "Species_Hard"}
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = analyzer.execute(labels, ih, ihm, mapping, temp_dir)
            assert os.path.exists(paths["class_summary_csv"])
            assert os.path.exists(paths["sample_size_analysis_csv"])

    def execute(self) -> None:
        self._test_class_records_and_brackets()
        self._test_execute_files_creation()

    def test_class_hardness_analyzer(self) -> None:
        self.execute()
