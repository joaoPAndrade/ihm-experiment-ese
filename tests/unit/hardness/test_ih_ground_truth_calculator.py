import numpy as np
import pytest
from src.hardness.ih_ground_truth_calculator import IhGroundTruthCalculator


class TestIhGroundTruthCalculator:
    def _test_nominal_and_extreme_cases(self) -> None:
        calculator = IhGroundTruthCalculator()
        labels = np.array([0, 1, 2])
        perfect_probs = np.zeros((3, 7, 3), dtype=np.float64)
        perfect_probs[0, :, 0] = 1.0
        perfect_probs[1, :, 1] = 1.0
        perfect_probs[2, :, 2] = 1.0
        assert np.allclose(calculator.execute(perfect_probs, labels), np.array([0.0, 0.0, 0.0]))
        worst_probs = np.zeros((3, 7, 3), dtype=np.float64)
        worst_probs[0, :, 1] = 1.0
        worst_probs[1, :, 2] = 1.0
        worst_probs[2, :, 0] = 1.0
        assert np.allclose(calculator.execute(worst_probs, labels), np.array([1.0, 1.0, 1.0]))
        mixed_probs = np.full((1, 4, 2), 0.5, dtype=np.float64)
        mixed_probs[0, :2, 0] = 0.8
        mixed_probs[0, 2:, 0] = 0.4
        expected_ih = 1.0 - (0.8 + 0.8 + 0.4 + 0.4) / 4.0
        assert calculator.execute(mixed_probs, np.array([0]))[0] == pytest.approx(expected_ih)

    def _test_dimensions_and_exceptions(self) -> None:
        calculator = IhGroundTruthCalculator()
        single_model_probs = np.array([[0.9, 0.1], [0.2, 0.8]])
        labels = np.array([0, 1])
        result = calculator.execute(single_model_probs, labels)
        assert np.allclose(result, np.array([0.1, 0.2]))
        with pytest.raises(ValueError):
            calculator.execute(np.zeros((2, 3, 2)), np.array([0]))
        with pytest.raises(ValueError):
            calculator.execute(np.zeros((2, 3, 2)), np.array([0, 5]))
        with pytest.raises(ValueError):
            calculator.execute(np.zeros((2,)), np.array([0, 1]))

    def execute(self) -> None:
        self._test_nominal_and_extreme_cases()
        self._test_dimensions_and_exceptions()

    def test_ih_ground_truth_calculator(self) -> None:
        self.execute()
