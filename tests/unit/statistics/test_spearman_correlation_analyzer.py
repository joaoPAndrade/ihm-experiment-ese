import numpy as np
import pytest

from src.statistics.spearman_correlation_analyzer import SpearmanCorrelationAnalyzer


class TestSpearmanCorrelationAnalyzer:
    def _test_correlation_calculations(self) -> None:
        analyzer = SpearmanCorrelationAnalyzer(alpha=0.05)
        ih = np.array([0.1, 0.2, 0.3, 0.8, 0.9], dtype=np.float64)
        m_pos = np.array([0.1, 0.2, 0.3, 0.8, 0.9], dtype=np.float64)
        m_neg = np.array([0.9, 0.8, 0.7, 0.2, 0.1], dtype=np.float64)
        m_const = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float64)
        metrics = {"pos": m_pos, "neg": m_neg, "const": m_const}
        results = analyzer.execute(ih, metrics)
        assert results["pos"][0] == pytest.approx(1.0)
        assert results["neg"][0] == pytest.approx(-1.0)
        assert results["const"][0] == 0.0

    def _test_exceptions(self) -> None:
        analyzer = SpearmanCorrelationAnalyzer()
        with pytest.raises(ValueError):
            analyzer.execute(np.ones(5), {})
        with pytest.raises(ValueError):
            analyzer.execute(np.ones(5), {"m": np.ones(3)})

    def execute(self) -> None:
        self._test_correlation_calculations()
        self._test_exceptions()

    def test_spearman_correlation_analyzer(self) -> None:
        self.execute()
