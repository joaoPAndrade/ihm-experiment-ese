import numpy as np
import pytest

from src.data.data_standard_scaler import DataStandardScaler


class TestDataStandardScaler:
    def _test_zscore_scaling_properties(self) -> None:
        scaler = DataStandardScaler()
        train = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]], dtype=np.float32)
        test = np.array([[3.0, 20.0], [7.0, 40.0]], dtype=np.float32)
        s_train, s_test = scaler.execute(train, test)
        assert np.allclose(np.mean(s_train, axis=0), 0.0, atol=1e-5)
        assert np.allclose(np.std(s_train, axis=0), 1.0, atol=1e-5)
        assert np.allclose(s_test[0], np.array([0.0, 0.0], dtype=np.float32), atol=1e-5)

    def _test_exceptions_and_dimensions(self) -> None:
        scaler = DataStandardScaler()
        with pytest.raises(ValueError):
            scaler.execute(np.ones((5,)), np.ones((2, 2)))
        with pytest.raises(ValueError):
            scaler.execute(np.ones((5, 3)), np.ones((2, 2)))

    def execute(self) -> None:
        self._test_zscore_scaling_properties()
        self._test_exceptions_and_dimensions()

    def test_data_standard_scaler(self) -> None:
        self.execute()
