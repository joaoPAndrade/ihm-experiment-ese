from typing import Any

from sklearn.ensemble import (
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


class ClassifierPoolBuilder:
    def __init__(self, seed: int = 42) -> None:
        self.seed: int = seed

    def _create_linear_and_kernel_models(self) -> dict[str, Any]:
        return {
            "SVM_Linear": SVC(kernel="linear", probability=True, random_state=self.seed),
            "SVM_RBF": SVC(kernel="rbf", probability=True, random_state=self.seed),
            "Logistic_Regression": LogisticRegression(max_iter=500, random_state=self.seed),
        }

    def _create_ensemble_and_neural_models(self) -> dict[str, Any]:
        return {
            "Random_Forest": RandomForestClassifier(n_estimators=50, random_state=self.seed),
            "Gradient_Boosting": GradientBoostingClassifier(n_estimators=50, random_state=self.seed),
            "Bagging": BaggingClassifier(random_state=self.seed),
            "MLP": MLPClassifier(hidden_layer_sizes=(50,), max_iter=200, random_state=self.seed),
        }

    def execute(self) -> dict[str, Any]:
        models = self._create_linear_and_kernel_models()
        models.update(self._create_ensemble_and_neural_models())
        return models
