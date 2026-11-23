# IHM Experiment - ESE

Projeto de pesquisa experimental em Engenharia de Software Experimental (ESE) com foco em Instance Hardness (IH) e Instance Hardness Measures (IHM) aplicadas à classificação botânica de exsicatas da família Piperaceae, reproduzindo e estendendo a metodologia de Torquette et al. (2024).

---

## 1. Configuração do Ambiente e Setup

O projeto utiliza **Python 3.14+** gerenciado via `venv`, com especificação declarativa de versão em `.python-version` e dependências padronizadas em `pyproject.toml` e `requirements.txt`.

```bash
chmod +x venv_setup.sh
./venv_setup.sh
source .venv/bin/activate
```

---

## 2. Como Executar o Projeto

Com o ambiente virtual ativado (`source .venv/bin/activate`), você pode executar o experimento de três formas principais:

### 2.1. Execução Rápida (Teste Amostral)
Ideal para verificar rapidamente todo o pipeline com uma amostragem reduzida:

```bash
python main.py --max-samples 50 --n-splits 3 --output-dir outputs_teste
```

### 2.2. Execução Completa de Todo o Dataset
Executa o pipeline científico ponta a ponta sobre **todas as 10.503 amostras e 236 classes botânicas**:

```bash
python main.py --csv-path datasets/final.csv --images-dir datasets/original --output-dir outputs --n-splits 5 --seed 42
```

### 2.3. Execução da Análise por Classe e Tamanho Amostral
Gera ou atualiza os relatórios de médias de IH/IHM por classe e a análise de desempenho por quantidade de amostras a partir de qualquer diretório de resultados:

```bash
python analyze_classes.py --results-dir outputs_teste --dataset-csv datasets/final.csv
```

---

## 3. Padrões de Engenharia e Arquitetura (.AGENTS)

- **Uma Classe por Arquivo**: Cada módulo `.py` encapsula uma única classe com o mesmo nome em `snake_case`.
- **Método `execute`**: O ponto de entrada e execução principal de cada classe é concentrado no método público `execute`.
- **Limite de Métodos e Linhas**: No máximo 4 métodos por classe e no máximo 30 linhas por método.
- **Zero Comentários Inline**: Ausência total de comentários (`#`) e docstrings no código-fonte Python; toda documentação técnica reside exclusivamente no `README.md`.
- **Tipagem Estática Completa**: 100% de parâmetros e retornos tipados estaticamente.
- **Quality Gate Obrigatório**: Validação via `scripts/quality_gate_validator.py`.

---

## 4. Módulos Implementados

### 4.1. Scripts de Execução CLI
- **`Main`** (`main.py`): Ponto de entrada CLI principal do experimento ponta a ponta.
- **`AnalyzeClassesRunner`** (`analyze_classes.py`): CLI para computar a agregação por classe botânica e faixas de quantidade de amostras.

### 4.2. Utilitários (`src/utils/`)
- **`VersionProvider`** (`src/utils/version_provider.py`): Recupera a versão semântica atual em `VERSION`.
- **`ResultsCsvExporter`** (`src/utils/results_csv_exporter.py`): Exporta os 4 CSVs estruturados de instâncias, correlações e hipótese.

### 4.3. Carregamento e Pré-processamento (`src/data/`)
- **`BotanicalDatasetLoader`** (`src/data/botanical_dataset_loader.py`): Ingestão das 10.503 exsicatas com auditoria e mapeamento de espécies.
- **`ImageFeatureExtractor`** (`src/data/image_feature_extractor.py`): Extração determinística de 45 atributos visuais (RGB, histogramas, gradientes espaciais e grid pooling).
- **`DataStandardScaler`** (`src/data/data_standard_scaler.py`): Padronizador Z-score ajustado no conjunto de treino dentro de cada fold (zero *data leakage*).

### 4.4. Validação e Particionamento (`src/validation/`)
- **`StratifiedFoldSplitter`** (`src/validation/stratified_fold_splitter.py`): Particionamento estratificado 5-fold preservando a proporção de classes raras.

### 4.5. Classificação (`src/classification/`)
- **`ClassifierPoolBuilder`** (`src/classification/classifier_pool_builder.py`): Constrói os 7 algoritmos heterogêneos (`SVM_Linear`, `SVM_RBF`, `Random_Forest`, `Gradient_Boosting`, `Bagging`, `Logistic_Regression`, `MLP`).
- **`ClassifierPoolEvaluator`** (`src/classification/classifier_pool_evaluator.py`): Executa validação cruzada gerando probabilidades *out-of-fold* e Macro F1-score.

### 4.6. Dificuldade de Instância (`src/hardness/`)
- **`IhGroundTruthCalculator`** (`src/hardness/ih_ground_truth_calculator.py`): Calcula o Ground Truth $IH(x_i, y_i)$ segundo a Equação 1 de Torquette et al. (2024).
- **`IhmMetricsExtractor`** (`src/hardness/ihm_metrics_extractor.py`) e **`GraphIhmCalculator`** (`src/hardness/graph_ihm_calculator.py`): Extraem as **13 medidas analíticas de classificação IHM**: $kDN$, $DCP$, $TD$, $CLD$, $F1_{IHM}$, $N1_{IHM}$, $N2_{IHM}$, $LSC_{IHM}$, $LSR_{IHM}$, $U$, $H$, $Degree_{IHM}$ e $Closeness_{IHM}$.
- **`IhmOutlierFilter`** (`src/hardness/ihm_outlier_filter.py`): Filtra instâncias com alto escore de ruído para o tratamento experimental de Q2.

### 4.7. Estatística e Análise por Classes (`src/statistics/`)
- **`SpearmanCorrelationAnalyzer`** (`src/statistics/spearman_correlation_analyzer.py`): Análise de correlação de postos de Spearman ($\rho$) e p-valores para Q1.
- **`HypothesisTester`** (`src/statistics/hypothesis_tester.py`): Shapiro-Wilk, teste pareado (t-Student ou Wilcoxon) e tamanho de efeito ($A_{12}$ ou $d$) para Q2.
- **`ClassHardnessAnalyzer`** (`src/statistics/class_hardness_analyzer.py`): Agrupa amostras por espécie botânica calculando a quantidade de amostras, IH médio, desvio padrão e médias das 13 IHMs, gerando também a análise de desempenho segmentada por quantidade de amostras.

### 4.8. Visualização Científica (`src/visualization/`)
- **`ScientificVisualizationPlotter`** (`src/visualization/scientific_visualization_plotter.py`): Boxplots dinâmicos (`ihm_boxplots.png`) e heatmap de Spearman (`spearman_heatmap.png`).

### 4.9. Pipeline Orquestrador (`src/pipelines/`)
- **`ExperimentPipeline`** (`src/pipelines/experiment_pipeline.py`): Orquestração ponta a ponta gerando todos os CSVs e gráficos no diretório de saída.

---

## 5. Estrutura dos Arquivos CSV Exportados

O pipeline exporta 6 arquivos no formato CSV (delimitador `;`, codificação `utf-8`):

1. **`instance_hardness_and_ihm.csv`**: Instâncias individuais com ground truth IH e as 13 métricas IHM.
2. **`q1_spearman_correlations.csv`**: Correlação de Spearman ($\rho$), p-valor e significância de cada métrica com o IH.
3. **`q2_classification_comparison.csv`**: Macro F1-score dos 7 modelos no baseline vs filtrado e ganho obtido.
4. **`q2_statistical_hypothesis_summary.csv`**: Teste de normalidade, teste pareado e tamanho de efeito.
5. **`class_hardness_summary.csv`**:
   - Cada linha representa uma classe botânica (espécie).
   - Colunas: `classe_id`, `nome_especie`, `quantidade_amostras`, `media_ground_truth_ih`, `desvio_ih`, `taxa_acerto_media`, seguidas pelas médias das 13 métricas IHM (`media_ihm_kDN`, `media_ihm_F1_IHM`, `media_ihm_TD`, etc.).
6. **`sample_size_performance_analysis.csv`**:
   - Cada linha representa uma faixa de tamanho de classe (ex.: 5 a 10 amostras, 11 a 25 amostras, etc.).
   - Colunas: `faixa_quantidade_amostras`, `quantidade_classes`, `total_amostras`, `media_ih_erro`, `taxa_acerto_media`, `media_kdn`, `media_dcp`, `media_n2`, `impacto_dificuldade`.
