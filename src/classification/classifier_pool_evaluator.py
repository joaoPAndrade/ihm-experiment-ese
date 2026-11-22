from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score

from src.data.data_standard_scaler import DataStandardScaler


class ClassifierPoolEvaluator:
    def __init__(self, average_f1: str = "macro") -> None:
        self.average_f1: str = average_f1

    def _evaluate_model_on_folds(self, model: Any, folds: list[tuple[np.ndarray, np.ndarray]], features: np.ndarray, labels: np.ndarray, num_classes: int) -> tuple[np.ndarray, float]:
        oof_probs = np.zeros((len(labels), num_classes), dtype=np.float64)
        fold_f1_scores: list[float] = []
        scaler = DataStandardScaler()
        for train_idx, test_idx in folds:
            x_train, x_test = scaler.execute(features[train_idx], features[test_idx])
            clf = clone(model)
            clf.fit(x_train, labels[train_idx])
            preds = clf.predict(x_test)
            fold_f1_scores.append(float(f1_score(labels[test_idx], preds, average=self.average_f1, zero_division=0.0)))
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(x_test)
                for col_idx, cls in enumerate(getattr(clf, "classes_", range(probs.shape[1]))):
                    if int(cls) < num_classes:
                        oof_probs[test_idx, int(cls)] = probs[:, col_idx]
            else:
                for idx_in_test, pred in enumerate(preds):
                    if int(pred) < num_classes:
                        oof_probs[test_idx[idx_in_test], int(pred)] = 1.0
        return oof_probs, float(np.mean(fold_f1_scores))

    def _aggregate_evaluation_results(self, model_outputs: list[tuple[np.ndarray, float]], sample_count: int, num_models: int, num_classes: int) -> dict[str, np.ndarray]:
        probabilities_tensor = np.zeros((sample_count, num_models, num_classes), dtype=np.float64)
        f1_list: list[float] = []
        for model_idx, (oof_probs, f1) in enumerate(model_outputs):
            probabilities_tensor[:, model_idx, :] = oof_probs
            f1_list.append(f1)
        return {"out_of_fold_probabilities": probabilities_tensor, "model_f1_scores": np.array(f1_list, dtype=np.float64)}

    def execute(self, classifiers: dict[str, Any], folds: list[tuple[np.ndarray, np.ndarray]], features: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
        if not classifiers or not folds:
            raise ValueError("Classifiers and folds cannot be empty")
        num_classes = int(np.max(labels)) + 1
        model_results = [self._evaluate_model_on_folds(clf, folds, features, labels, num_classes) for clf in classifiers.values()]
        return self._aggregate_evaluation_results(model_results, len(labels), len(classifiers), num_classes)
