import math
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class ScientificVisualizationPlotter:
    def __init__(self, dpi: int = 150) -> None:
        self.dpi: int = dpi

    def _plot_ihm_boxplots(self, ih_scores: np.ndarray, ihm_metrics: dict[str, np.ndarray], output_path: str) -> str:
        metric_names = list(ihm_metrics.keys())
        cols = min(4, max(1, len(metric_names)))
        rows = max(1, math.ceil(len(metric_names) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows), squeeze=False)
        flat_axes = axes.flatten()
        for idx, name in enumerate(metric_names):
            vals = np.asarray(ihm_metrics[name], dtype=np.float64)
            binned = np.round(vals, 1)
            sns.boxplot(x=binned, y=ih_scores, ax=flat_axes[idx], color="#4C72B0")
            flat_axes[idx].set_title(f"IHM: {name}")
            flat_axes[idx].set_xlabel("Faixa de Dificuldade")
            flat_axes[idx].set_ylabel("Instance Hardness (IH)")
        for idx in range(len(metric_names), len(flat_axes)):
            flat_axes[idx].set_visible(False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi)
        plt.close(fig)
        return output_path

    def _plot_correlation_heatmap(self, correlation_map: dict[str, tuple[float, float, bool]], output_path: str) -> str:
        names = list(correlation_map.keys())
        rhos = np.array([[correlation_map[k][0]] for k in names])
        fig, ax = plt.subplots(figsize=(6, max(4, len(names) * 0.5)))
        sns.heatmap(rhos, annot=True, fmt=".3f", yticklabels=names, xticklabels=["Spearman Rho"], cmap="coolwarm", vmin=-1.0, vmax=1.0, ax=ax)
        ax.set_title("Correlação de Spearman com IH")
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi)
        plt.close(fig)
        return output_path

    def execute(self, ih_scores: np.ndarray, ihm_metrics: dict[str, np.ndarray], correlation_map: dict[str, tuple[float, float, bool]], output_directory: str) -> list[str]:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory, exist_ok=True)
        boxplot_path = os.path.join(output_directory, "ihm_boxplots.png")
        heatmap_path = os.path.join(output_directory, "spearman_heatmap.png")
        p1 = self._plot_ihm_boxplots(ih_scores, ihm_metrics, boxplot_path)
        p2 = self._plot_correlation_heatmap(correlation_map, heatmap_path)
        return [p1, p2]
