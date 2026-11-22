from collections import Counter
from typing import Any

import numpy as np

from src.classification.classifier_pool_builder import ClassifierPoolBuilder
from src.classification.classifier_pool_evaluator import ClassifierPoolEvaluator
from src.data.botanical_dataset_loader import BotanicalDatasetLoader
from src.data.data_standard_scaler import DataStandardScaler
from src.data.image_feature_extractor import ImageFeatureExtractor
from src.hardness.ih_ground_truth_calculator import IhGroundTruthCalculator
from src.hardness.ihm_metrics_extractor import IhmMetricsExtractor
from src.hardness.ihm_outlier_filter import IhmOutlierFilter
from src.statistics.hypothesis_tester import HypothesisTester
from src.statistics.spearman_correlation_analyzer import SpearmanCorrelationAnalyzer
from src.utils.results_csv_exporter import ResultsCsvExporter
from src.validation.stratified_fold_splitter import StratifiedFoldSplitter
from src.visualization.scientific_visualization_plotter import (
    ScientificVisualizationPlotter,
)


class ExperimentPipeline:
    def __init__(self, random_state: int = 42, n_splits: int = 5, target_size: tuple[int, int] = (64, 64)) -> None:
        self.random_state: int = random_state
        self.n_splits: int = n_splits
        self.target_size: tuple[int, int] = target_size

    def _run_baseline_and_hardness_stage(self, features: np.ndarray, labels: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
        pool = ClassifierPoolBuilder(seed=self.random_state).execute()
        eval_res = ClassifierPoolEvaluator().execute(pool, folds, features, labels)
        ih = IhGroundTruthCalculator().execute(eval_res["out_of_fold_probabilities"], labels)
        scaled_global, _ = DataStandardScaler().execute(features, features)
        ihm = IhmMetricsExtractor(seed=self.random_state).execute(scaled_global, labels)
        correlations = SpearmanCorrelationAnalyzer().execute(ih, ihm)
        return {"pool": pool, "eval_res": eval_res, "ih": ih, "ihm": ihm, "correlations": correlations}

    def _run_statistical_and_visualization_stage(self, baseline_stage: dict[str, Any], features: np.ndarray, labels: np.ndarray, output_dir: str) -> dict[str, Any]:
        f_feat, f_labels, _mask = IhmOutlierFilter().execute(features, labels, baseline_stage["ihm"])
        counts = Counter(f_labels.tolist())
        valid_mask = np.array([counts[int(lbl)] >= self.n_splits for lbl in f_labels], dtype=bool)
        if np.sum(valid_mask) >= self.n_splits and len(np.unique(f_labels[valid_mask])) >= 2:
            f_feat, f_labels = f_feat[valid_mask], f_labels[valid_mask]
            _, f_labels = np.unique(f_labels, return_inverse=True)
        filt_folds = StratifiedFoldSplitter(n_splits=self.n_splits, random_state=self.random_state).execute(len(f_labels), f_labels)
        filt_res = ClassifierPoolEvaluator().execute(baseline_stage["pool"], filt_folds, f_feat, f_labels)
        base_f1 = baseline_stage["eval_res"]["model_f1_scores"]
        filt_f1 = filt_res["model_f1_scores"]
        hypothesis_res = HypothesisTester().execute(base_f1, filt_f1)
        plot_paths = ScientificVisualizationPlotter().execute(baseline_stage["ih"], baseline_stage["ihm"], baseline_stage["correlations"], output_dir)
        csv_payload = {"ih": baseline_stage["ih"], "ihm": baseline_stage["ihm"], "labels": labels, "correlations": baseline_stage["correlations"], "base_f1": base_f1, "filt_f1": filt_f1, "hypothesis_res": hypothesis_res, "model_names": list(baseline_stage["pool"].keys())}
        csv_paths = ResultsCsvExporter().execute(csv_payload, output_dir)
        return {"hypothesis_test": hypothesis_res, "plot_paths": plot_paths, "csv_paths": csv_paths, "filtered_sample_count": len(f_labels)}

    def execute(self, csv_path: str, images_dir: str, output_dir: str, max_samples: int | None = None) -> dict[str, Any]:
        paths, labels, _mapping = BotanicalDatasetLoader(min_samples_per_class=self.n_splits).execute(csv_path, images_dir)
        if max_samples is not None and max_samples < len(paths):
            sel_indices: list[int] = []
            for cls_idx in np.unique(labels):
                cls_indices = np.where(labels == cls_idx)[0].tolist()
                take = min(len(cls_indices), max(self.n_splits, max_samples // 4))
                sel_indices.extend(cls_indices[:take])
                if len(sel_indices) >= max_samples and len(np.unique(labels[sel_indices])) >= 2:
                    break
            paths = [paths[i] for i in sel_indices]
            _, labels = np.unique(labels[sel_indices], return_inverse=True)
        features = ImageFeatureExtractor(target_size=self.target_size).execute(paths, batch_size=32)
        folds = StratifiedFoldSplitter(n_splits=self.n_splits, random_state=self.random_state).execute(len(labels), labels)
        baseline_res = self._run_baseline_and_hardness_stage(features, labels, folds)
        stats_res = self._run_statistical_and_visualization_stage(baseline_res, features, labels, output_dir)
        return {"dataset_stats": {"total_samples": len(labels), "classes": len(np.unique(labels))}, "q1_spearman_correlations": baseline_res["correlations"], "q2_hypothesis_test": stats_res["hypothesis_test"], "plot_paths": stats_res["plot_paths"], "csv_paths": stats_res["csv_paths"]}
