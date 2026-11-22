import argparse
import sys
from typing import Any

from src.pipelines.experiment_pipeline import ExperimentPipeline


class Main:
    def __init__(self) -> None:
        self.parser: argparse.ArgumentParser = argparse.ArgumentParser(description="IHM Experiment Pipeline Runner")
        self.parser.add_argument("--csv-path", type=str, default="datasets/final.csv")
        self.parser.add_argument("--images-dir", type=str, default="datasets/original")
        self.parser.add_argument("--output-dir", type=str, default="outputs")
        self.parser.add_argument("--max-samples", type=int, default=None)
        self.parser.add_argument("--n-splits", type=int, default=5)
        self.parser.add_argument("--seed", type=int, default=42)

    def _display_summary(self, results: dict[str, Any]) -> None:
        print("\n=== EXPERIMENTO IHM CONCLUIDO COM SUCESSO ===")
        print(f"Total de Amostras: {results['dataset_stats']['total_samples']}")
        print(f"Total de Classes: {results['dataset_stats']['classes']}")
        print("\n--- Correlacoes de Spearman (Q1) ---")
        for metric, (rho, p_val, is_sig) in results["q1_spearman_correlations"].items():
            sig = "SIM" if is_sig else "NAO"
            print(f"  {metric:15s}: rho={rho:+.4f}, p={p_val:.4e}, sig={sig}")
        hyp = results["q2_hypothesis_test"]
        print("\n--- Teste de Hipotese (Q2) ---")
        print(f"  Normalidade (Shapiro-Wilk p): {hyp['shapiro_p']:.4e}")
        print(f"  p-valor do Teste Pareado:     {hyp['p_value']:.4e}")
        print(f"  Significante (alpha=0.05):    {'SIM' if hyp['significant'] > 0.5 else 'NAO'}")
        print(f"  Tamanho de Efeito:            {hyp['effect_size']:.4f}")
        print("\n--- Graficos Gerados ---")
        for path in results["plot_paths"]:
            print(f"  Grafico: {path}")
        print("\n--- Arquivos CSV Exportados ---")
        for path in results["csv_paths"]:
            print(f"  CSV:     {path}")

    def execute(self, arguments: list[str] | None = None) -> int:
        args = self.parser.parse_args(arguments)
        pipeline = ExperimentPipeline(random_state=args.seed, n_splits=args.n_splits)
        results = pipeline.execute(csv_path=args.csv_path, images_dir=args.images_dir, output_dir=args.output_dir, max_samples=args.max_samples)
        self._display_summary(results)
        return 0


if __name__ == "__main__":
    runner = Main()
    sys.exit(runner.execute(sys.argv[1:]))
