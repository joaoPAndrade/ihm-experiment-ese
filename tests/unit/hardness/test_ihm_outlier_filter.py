import numpy as np
import pytest

from src.hardness.ihm_outlier_filter import IhmOutlierFilter


class TestIhmOutlierFilter:
    def _test_outlier_filtering(self) -> None:
        filter_obj = IhmOutlierFilter(metric_name="kDN", cutoff_percentile=60.0)
        features = np.array([[1], [2], [3], [4], [5]], dtype=np.float32)
        labels = np.array([0, 0, 1, 1, 1], dtype=np.int64)
        ihm = {"kDN": np.array([0.1, 0.2, 0.3, 0.9, 1.0])}
        f_feat, f_labels, mask = filter_obj.execute(features, labels, ihm)
        assert len(f_feat) == 3
        assert len(f_labels) == 3
        assert np.array_equal(mask, np.array([True, True, True, False, False]))

    def _test_exceptions_and_fallbacks(self) -> None:
        filter_obj = IhmOutlierFilter(metric_name="unknown")
        ihm = {"m1": np.array([0.1, 0.9]), "m2": np.array([0.2, 0.8])}
        f_feat, _f_labels, _mask = filter_obj.execute(np.zeros((2, 1)), np.array([0, 1]), ihm)
        assert len(f_feat) >= 1
        with pytest.raises(ValueError):
            filter_obj.execute(np.zeros((2, 1)), np.array([0]), ihm)

    def execute(self) -> None:
        self._test_outlier_filtering()
        self._test_exceptions_and_fallbacks()

    def test_ihm_outlier_filter(self) -> None:
        self.execute()
