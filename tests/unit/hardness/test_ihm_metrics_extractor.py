import numpy as np
import pytest

from src.hardness.ihm_metrics_extractor import IhmMetricsExtractor


class TestIhmMetricsExtractor:
    def _test_metrics_extraction(self) -> None:
        extractor = IhmMetricsExtractor(k_neighbors=2)
        features = np.array([[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]], dtype=np.float64)
        labels = np.array([0, 0, 1, 1], dtype=np.int64)
        metrics = extractor.execute(features, labels)
        expected_keys = {
            "kDN",
            "DCP",
            "TD",
            "CLD",
            "F1_IHM",
            "N1_IHM",
            "N2_IHM",
            "LSC_IHM",
            "LSR_IHM",
            "U",
            "H",
            "Degree_IHM",
            "Closeness_IHM",
        }
        assert set(metrics.keys()) == expected_keys
        for arr in metrics.values():
            assert arr.shape == (4,)
            assert np.all(arr >= 0.0) and np.all(arr <= 1.0)

    def _test_exceptions(self) -> None:
        extractor = IhmMetricsExtractor()
        with pytest.raises(ValueError):
            extractor.execute(np.zeros((1, 2)), np.array([0]))
        with pytest.raises(ValueError):
            extractor.execute(np.zeros((3, 2)), np.array([0, 1]))

    def execute(self) -> None:
        self._test_metrics_extraction()
        self._test_exceptions()

    def test_ihm_metrics_extractor(self) -> None:
        self.execute()
