import argparse
import csv
import sys

import numpy as np

from src.statistics.class_hardness_analyzer import ClassHardnessAnalyzer


class AnalyzeClassesRunner:
    def __init__(self) -> None:
        self.parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Class Hardness and Sample Size Performance Analyzer")
        self.parser.add_argument("--results-dir", type=str, default="outputs_teste", help="Directory with instance_hardness_and_ihm.csv")
        self.parser.add_argument("--dataset-csv", type=str, default="datasets/final.csv", help="Metadata CSV to resolve species names")
        self.parser.add_argument("--output-dir", type=str, default=None, help="Output directory (defaults to results-dir)")

    def _load_data_and_mapping(self, results_dir: str, dataset_csv: str) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[int, str]]:
        instances_csv = f"{results_dir}/instance_hardness_and_ihm.csv"
        labels_list, ih_list, mapping = [], [], {}
        ihm_dict: dict[str, list[float]] = {}
        with open(instances_csv, "r", encoding="utf-8") as stream:
            reader = csv.DictReader(stream, delimiter=";")
            ihm_cols = [c for c in (reader.fieldnames or []) if c.startswith("ihm_")]
            for col in ihm_cols:
                ihm_dict[col.replace("ihm_", "")] = []
            for row in reader:
                cls_id = int(row["rotulo_classe"])
                labels_list.append(cls_id)
                ih_list.append(float(row["ground_truth_ih"]))
                if row.get("nome_especie"):
                    mapping[cls_id] = row["nome_especie"]
                for col in ihm_cols:
                    ihm_dict[col.replace("ihm_", "")].append(float(row[col]))
        if not mapping:
            with open(dataset_csv, "r", encoding="utf-8", errors="ignore") as stream:
                r = csv.DictReader(stream, delimiter=";" if ";" in stream.readline() else ",")
                stream.seek(0)
                species_set = sorted({row["specific_epithet"].strip() for row in r if row.get("specific_epithet")})
                mapping = {idx: sp for idx, sp in enumerate(species_set)}
        return np.array(labels_list, dtype=np.int64), np.array(ih_list, dtype=np.float64), {k: np.array(v, dtype=np.float64) for k, v in ihm_dict.items()}, mapping

    def _display_analysis_summary(self, paths: dict[str, str], results_dir: str) -> None:
        print("\n=== ANALISE POR CLASSE E TAMANHO DE AMOSTRAS CONCLUIDA ===")
        print(f"Resumo por Classe salvo em:             {paths['class_summary_csv']}")
        print(f"Analise por Tamanho de Amostras salva em: {paths['sample_size_analysis_csv']}")
        print("\n--- Desempenho Medio por Faixa de Amostras ---")
        with open(paths["sample_size_analysis_csv"], "r", encoding="utf-8") as stream:
            reader = csv.DictReader(stream, delimiter=";")
            for row in reader:
                print(f"  [{row['faixa_quantidade_amostras']}]")
                print(f"    Classes: {row['quantidade_classes']} | Total Amostras: {row['total_amostras']}")
                print(f"    IH Medio (Taxa Erro): {float(row['media_ih_erro']):.4f} | Taxa Acerto: {float(row['taxa_acerto_media']):.4f}")
                print(f"    kDN Medio: {float(row['media_kdn']):.4f} | DCP Medio: {float(row['media_dcp']):.4f} | N2: {float(row['media_n2']):.4f}")
                print(f"    Avaliacao: {row['impacto_dificuldade']}\n")

    def execute(self, arguments: list[str] | None = None) -> int:
        args = self.parser.parse_args(arguments)
        out_dir = args.output_dir or args.results_dir
        labels, ih, ihm, mapping = self._load_data_and_mapping(args.results_dir, args.dataset_csv)
        paths = ClassHardnessAnalyzer().execute(labels, ih, ihm, mapping, out_dir)
        self._display_analysis_summary(paths, out_dir)
        return 0


if __name__ == "__main__":
    runner = AnalyzeClassesRunner()
    sys.exit(runner.execute(sys.argv[1:]))
