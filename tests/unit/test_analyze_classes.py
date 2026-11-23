import os
import tempfile

from analyze_classes import AnalyzeClassesRunner


class TestAnalyzeClasses:
    def _test_runner_execution(self) -> None:
        runner = AnalyzeClassesRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            res_dir = os.path.join(temp_dir, "results")
            os.makedirs(res_dir)
            inst_csv = os.path.join(res_dir, "instance_hardness_and_ihm.csv")
            with open(inst_csv, "w", encoding="utf-8") as s:
                s.write("amostra_id;rotulo_classe;ground_truth_ih;ihm_kDN;ihm_DCP;ihm_N2_IHM\n")
                for i in range(12):
                    cls = 0 if i < 6 else 1
                    ih_val = 0.2 if cls == 0 else 0.8
                    s.write(f"{i};{cls};{ih_val};0.1;0.0;0.2\n")
            meta_csv = os.path.join(temp_dir, "meta.csv")
            with open(meta_csv, "w", encoding="utf-8") as s:
                s.write("filename;specific_epithet\n")
                s.write("a.jpg;species_alpha\nb.jpg;species_beta\n")
            status = runner.execute(["--results-dir", res_dir, "--dataset-csv", meta_csv])
            assert status == 0
            assert os.path.exists(os.path.join(res_dir, "class_hardness_summary.csv"))
            assert os.path.exists(os.path.join(res_dir, "sample_size_performance_analysis.csv"))

    def execute(self) -> None:
        self._test_runner_execution()

    def test_analyze_classes(self) -> None:
        self.execute()
