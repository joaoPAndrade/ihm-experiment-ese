import numpy as np


class IhGroundTruthCalculator:
    def __init__(self, clip_bounds: bool = True) -> None:
        self.clip_bounds: bool = clip_bounds

    def _extract_true_class_probabilities(self, probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
        if probabilities.ndim == 2:
            probabilities = probabilities[:, np.newaxis, :]
        if probabilities.ndim != 3:
            raise ValueError("Probabilities must be 2D (N, C) or 3D (N, L, C)")
        if labels.ndim != 1 or probabilities.shape[0] != labels.shape[0]:
            raise ValueError("Sample counts in probabilities and labels must match")
        if np.any(labels < 0) or np.any(labels >= probabilities.shape[2]):
            raise ValueError("Label indices must be within [0, num_classes - 1]")
        sample_indices = np.arange(probabilities.shape[0])[:, None]
        model_indices = np.arange(probabilities.shape[1])[None, :]
        target_labels = labels[:, None]
        return probabilities[sample_indices, model_indices, target_labels]

    def _compute_hardness_scores(self, true_class_probs: np.ndarray) -> np.ndarray:
        mean_confidence = np.mean(true_class_probs, axis=1)
        hardness_scores = 1.0 - mean_confidence
        if self.clip_bounds:
            hardness_scores = np.clip(hardness_scores, 0.0, 1.0)
        return hardness_scores.astype(np.float64)

    def execute(self, predicted_probabilities: np.ndarray, true_labels: np.ndarray) -> np.ndarray:
        true_probs = self._extract_true_class_probabilities(predicted_probabilities, true_labels)
        return self._compute_hardness_scores(true_probs)
