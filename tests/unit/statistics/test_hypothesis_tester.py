import numpy as np
import pytest

from src.statistics.hypothesis_tester import HypothesisTester


class TestHypothesisTester:
    def _test_hypothesis_testing(self) -> None:
        tester = HypothesisTester(alpha=0.05)
        baseline = np.array([0.70, 0.72, 0.71, 0.69, 0.73, 0.70, 0.72], dtype=np.float64)
        filtered = np.array([0.80, 0.82, 0.81, 0.79, 0.83, 0.80, 0.82], dtype=np.float64)
        result = tester.execute(baseline, filtered)
        assert "p_value" in result
        assert "effect_size" in result
        assert "significant" in result
        assert result["effect_size"] > 0.0

    def _test_exceptions(self) -> None:
        tester = HypothesisTester()
        with pytest.raises(ValueError):
            tester.execute(np.ones(1), np.ones(1))
        with pytest.raises(ValueError):
            tester.execute(np.ones(5), np.ones(3))

    def execute(self) -> None:
        self._test_hypothesis_testing()
        self._test_exceptions()

    def test_hypothesis_tester(self) -> None:
        self.execute()
