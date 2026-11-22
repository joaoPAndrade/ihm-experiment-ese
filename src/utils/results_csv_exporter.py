import csv
import os
from typing import Any


class ResultsCsvExporter:
    def __init__(self, delimiter: str = ";") -> None:
        self.delimiter: str = delimiter

    def _export_instances_and_correlations(self, data: dict[str, Any], output_dir: str) -> list[str]:
        p1 = os.path.join(output_dir, "instance_hardness_and_ihm.csv")
        ih, ihm, labels = data["ih"], data["ihm"], data["labels"]
        metric_keys = list(ihm.keys())
        with open(p1, "w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter=self.delimiter)
            writer.writerow(["amostra_id", "rotulo_classe", "ground_truth_ih"] + [f"ihm_{k}" for k in metric_keys])
            for idx in range(len(ih)):
                row = [idx, int(labels[idx]), f"{float(ih[idx]):.6f}"] + [f"{float(ihm[k][idx]):.6f}" for k in metric_keys]
                writer.writerow(row)
        p2 = os.path.join(output_dir, "q1_spearman_correlations.csv")
        corrs = data["correlations"]
        with open(p2, "w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter=self.delimiter)
            writer.writerow(["metrica_ihm", "spearman_rho", "p_valor", "significante_alpha_0_05"])
            for name, (rho, p_val, is_sig) in corrs.items():
                writer.writerow([name, f"{float(rho):.6f}", f"{float(p_val):.6e}", "SIM" if is_sig else "NAO"])
        return [p1, p2]

    def _export_q2_comparison_and_hypothesis(self, data: dict[str, Any], output_dir: str) -> list[str]:
        p3 = os.path.join(output_dir, "q2_classification_comparison.csv")
        models = data.get("model_names", [f"Model_{i}" for i in range(len(data["base_f1"]))])
        with open(p3, "w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter=self.delimiter)
            writer.writerow(["classificador", "f1_score_baseline", "f1_score_filtrado_ihm", "diferenca_ganho_f1"])
            for i, name in enumerate(models):
                b_f1, f_f1 = float(data["base_f1"][i]), float(data["filt_f1"][i])
                writer.writerow([name, f"{b_f1:.6f}", f"{f_f1:.6f}", f"{(f_f1 - b_f1):.6f}"])
        p4 = os.path.join(output_dir, "q2_statistical_hypothesis_summary.csv")
        hyp = data["hypothesis_res"]
        with open(p4, "w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter=self.delimiter)
            writer.writerow(["metrica_avaliacao", "valor", "interpretacao"])
            norm_str = "Normal" if hyp["is_normal"] > 0.5 else "Nao-Normal"
            writer.writerow(["Normalidade (Shapiro-Wilk)", f"{hyp['shapiro_p']:.6e}", norm_str])
            test_name = "t-Student Pareado" if hyp["is_normal"] > 0.5 else "Wilcoxon Pareado"
            writer.writerow(["Teste de Hipotese Pareado", test_name, f"p-valor: {hyp['p_value']:.6e}"])
            sig_str = "SIM (Rejeita H0)" if hyp["significant"] > 0.5 else "NAO (Nao Rejeita H0)"
            writer.writerow(["Significante a alpha=0.05", sig_str, "Impacto Estatistico Confirmado" if hyp["significant"] > 0.5 else "Sem diferenca estatistica"])
            eff_name = "d de Cohen" if hyp["is_normal"] > 0.5 else "Vargha-Delaney A12"
            writer.writerow([f"Tamanho de Efeito ({eff_name})", f"{hyp['effect_size']:.6f}", "Magnitude do Efeito da Filtragem"])
        return [p3, p4]

    def execute(self, experiment_results: dict[str, Any], output_dir: str) -> list[str]:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        paths = self._export_instances_and_correlations(experiment_results, output_dir)
        paths.extend(self._export_q2_comparison_and_hypothesis(experiment_results, output_dir))
        return paths
