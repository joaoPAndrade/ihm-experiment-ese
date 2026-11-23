import csv
import os
from typing import Any

import numpy as np


class ClassHardnessAnalyzer:
    def __init__(self, delimiter: str = ";") -> None:
        self.delimiter: str = delimiter

    def _compute_class_records(self, labels: np.ndarray, ih: np.ndarray, ihm: dict[str, np.ndarray], mapping: dict[int, str] | None) -> list[dict[str, Any]]:
        metric_keys = list(ihm.keys())
        records: list[dict[str, Any]] = []
        for cls_id in sorted(np.unique(labels)):
            mask = labels == cls_id
            c_ih = ih[mask]
            sp_name = mapping.get(int(cls_id), f"Class_{cls_id}") if mapping else f"Class_{cls_id}"
            rec: dict[str, Any] = {
                "classe_id": int(cls_id),
                "nome_especie": sp_name,
                "quantidade_amostras": int(np.sum(mask)),
                "media_ground_truth_ih": float(np.mean(c_ih)),
                "desvio_ih": float(np.std(c_ih)),
                "taxa_acerto_media": 1.0 - float(np.mean(c_ih)),
            }
            for k in metric_keys:
                rec[f"media_ihm_{k}"] = float(np.mean(ihm[k][mask]))
            records.append(rec)
        return records

    def _compute_size_brackets(self, class_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        brackets = [
            ("5 a 10 amostras (Raras)", 5, 10),
            ("11 a 25 amostras (Baixo Suporte)", 11, 25),
            ("26 a 50 amostras (Medio Suporte)", 26, 50),
            ("51 a 100 amostras (Alto Suporte)", 51, 100),
            ("> 100 amostras (Abundantes)", 101, 10000000),
        ]
        results: list[dict[str, Any]] = []
        for label, low, high in brackets:
            matched = [r for r in class_records if low <= r["quantidade_amostras"] <= high]
            if not matched:
                continue
            m_ih = float(np.mean([r["media_ground_truth_ih"] for r in matched]))
            imp = "Alta Dureza / Risco de Erro Elevado" if m_ih > 0.55 else "Dureza Moderada / Desempenho Estavel"
            results.append({
                "faixa_quantidade_amostras": label,
                "quantidade_classes": len(matched),
                "total_amostras": sum(r["quantidade_amostras"] for r in matched),
                "media_ih_erro": m_ih,
                "taxa_acerto_media": 1.0 - m_ih,
                "media_kdn": float(np.mean([r.get("media_ihm_kDN", 0.0) for r in matched])),
                "media_dcp": float(np.mean([r.get("media_ihm_DCP", 0.0) for r in matched])),
                "media_n2": float(np.mean([r.get("media_ihm_N2_IHM", 0.0) for r in matched])),
                "impacto_dificuldade": imp,
            })
        return results

    def execute(self, labels: np.ndarray, ih: np.ndarray, ihm: dict[str, np.ndarray], mapping: dict[int, str] | None, output_dir: str) -> dict[str, str]:
        if len(labels) == 0:
            raise ValueError("Labels array cannot be empty")
        os.makedirs(output_dir, exist_ok=True)
        class_records = self._compute_class_records(labels, ih, ihm, mapping)
        p1 = os.path.join(output_dir, "class_hardness_summary.csv")
        metric_keys = [k for k in class_records[0] if k.startswith("media_ihm_")]
        with open(p1, "w", encoding="utf-8", newline="") as s1:
            w1 = csv.writer(s1, delimiter=self.delimiter)
            w1.writerow(["classe_id", "nome_especie", "quantidade_amostras", "media_ground_truth_ih", "desvio_ih", "taxa_acerto_media"] + metric_keys)
            for r in class_records:
                w1.writerow([r["classe_id"], r["nome_especie"], r["quantidade_amostras"], f"{r['media_ground_truth_ih']:.6f}", f"{r['desvio_ih']:.6f}", f"{r['taxa_acerto_media']:.6f}"] + [f"{r[k]:.6f}" for k in metric_keys])
        p2 = os.path.join(output_dir, "sample_size_performance_analysis.csv")
        bracket_records = self._compute_size_brackets(class_records)
        with open(p2, "w", encoding="utf-8", newline="") as s2:
            w2 = csv.writer(s2, delimiter=self.delimiter)
            w2.writerow(["faixa_quantidade_amostras", "quantidade_classes", "total_amostras", "media_ih_erro", "taxa_acerto_media", "media_kdn", "media_dcp", "media_n2", "impacto_dificuldade"])
            for b in bracket_records:
                w2.writerow([b["faixa_quantidade_amostras"], b["quantidade_classes"], b["total_amostras"], f"{b['media_ih_erro']:.6f}", f"{b['taxa_acerto_media']:.6f}", f"{b['media_kdn']:.6f}", f"{b['media_dcp']:.6f}", f"{b['media_n2']:.6f}", b["impacto_dificuldade"]])
        return {"class_summary_csv": p1, "sample_size_analysis_csv": p2}
