import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.classification.classifier_pool_evaluator import ClassifierPoolEvaluator


class TestClassifierPoolEvaluator:
    def _test_evaluator_execution_and_shapes(self) -> None:
        evaluator = ClassifierPoolEvaluator()
        features = np.array([[1.0, 0.0], [1.1, 0.1], [0.0, 1.0], [0.1, 1.1]], dtype=np.float32)
        labels = np.array([0, 0, 1, 1], dtype=np.int64)
        folds = [(np.array([1, 2, 3]), np.array([0])), (np.array([0, 2, 3]), np.array([1])), (np.array([0, 1, 3]), np.array([2])), (np.array([0, 1, 2]), np.array([3]))]
        models = {"LR": LogisticRegression(), "DT": DecisionTreeClassifier(random_state=42)}
        result = evaluator.execute(models, folds, features, labels)
        assert "out_of_fold_probabilities" in result
        assert "model_f1_scores" in result
        assert result["out_of_fold_probabilities"].shape == (4, 2, 2)
        assert result["model_f1_scores"].shape == (2,)

    def _test_exceptions(self) -> None:
        evaluator = ClassifierPoolEvaluator()
        with pytest.raises(ValueError):
            evaluator.execute({}, [], np.zeros((2, 2)), np.zeros((2,)))

    def execute(self) -> None:
        self._test_evaluator_execution_and_shapes()
        self._test_exceptions()

    def test_classifier_pool_evaluator(self) -> None:
        self.execute()
