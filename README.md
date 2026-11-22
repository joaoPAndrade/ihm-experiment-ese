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

### 2.2. Execução Completa do Experimento
Executa o pipeline científico ponta a ponta sobre as 10.503 amostras e 236 classes botânicas:

```bash
python main.py --csv-path datasets/final.csv --images-dir datasets/original --output-dir outputs --n-splits 5 --seed 42
```

#### Parâmetros Disponíveis da CLI (`main.py`):
| Parâmetro | Tipo | Padrão | Descrição |
| :--- | :--- | :--- | :--- |
| `--csv-path` | `str` | `datasets/final.csv` | Caminho para a planilha de metadados das amostras |
| `--images-dir` | `str` | `datasets/original` | Diretório raiz contendo as imagens JPEG das exsicatas |
| `--output-dir` | `str` | `outputs` | Diretório de destino para os relatórios CSV e gráficos PNG |
| `--max-samples` | `int` | `None` | Limite de amostras a carregar (útil para testes rápidos) |
| `--n-splits` | `int` | `5` | Quantidade de folds na validação cruzada estratificada |
| `--seed` | `int` | `42` | Semente pseudoaleatória para reprodutibilidade científica |

### 2.3. Execução da Suíte de Testes e Validação
Para rodar todos os testes unitários com cobertura completa:

```bash
pytest
```

Para auditar conformidade com as regras arquiteturais (.AGENTS):

```bash
python scripts/quality_gate_validator.py $(find src tests main.py -name "*.py" | sort)
ruff check src tests main.py
mypy src tests main.py --ignore-missing-imports
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

### 4.1. Ponto de Entrada CLI (`main.py`)
- **`Main`** (`main.py`): Ponto de entrada executável por linha de comando (`CLI`), faz o parsing dos argumentos, dispara o orquestrador `ExperimentPipeline` e exibe o sumário dos resultados no terminal.

### 4.2. Utilitários (`src/utils/`)
- **`VersionProvider`** (`src/utils/version_provider.py`): Recupera a versão semântica atual em `VERSION`.
- **`ResultsCsvExporter`** (`src/utils/results_csv_exporter.py`): Exporta todos os dados analíticos e estatísticos do experimento para 4 arquivos CSV estruturados (separador `;`).

### 4.3. Carregamento e Pré-processamento (`src/data/`)
- **`BotanicalDatasetLoader`** (`src/data/botanical_dataset_loader.py`): Ingestão e auditoria das 10.503 exsicatas com filtro de suporte mínimo por classe.
- **`ImageFeatureExtractor`** (`src/data/image_feature_extractor.py`): Extração de 45 atributos visuais (RGB, histogramas, gradientes espaciais e grid pooling).
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

### 4.7. Estatística e Validade Científica (`src/statistics/`)
- **`SpearmanCorrelationAnalyzer`** (`src/statistics/spearman_correlation_analyzer.py`): Análise de correlação de postos de Spearman ($\rho$) e p-valores para Q1.
- **`HypothesisTester`** (`src/statistics/hypothesis_tester.py`): Shapiro-Wilk, teste pareado (t-Student ou Wilcoxon) e tamanho de efeito ($A_{12}$ ou $d$) para Q2.

### 4.8. Visualização Científica (`src/visualization/`)
- **`ScientificVisualizationPlotter`** (`src/visualization/scientific_visualization_plotter.py`): Boxplots dinâmicos (`ihm_boxplots.png`) e heatmap de Spearman (`spearman_heatmap.png`).

### 4.9. Pipeline Orquestrador (`src/pipelines/`)
- **`ExperimentPipeline`** (`src/pipelines/experiment_pipeline.py`): Orquestração do experimento ponta a ponta com exportação em CSV e PNG.

---

## 5. Estrutura dos Arquivos CSV Exportados

O pipeline exporta 4 arquivos no formato CSV (delimitador `;`, codificação `utf-8`):

1. **`instance_hardness_and_ihm.csv`**:
   - Cada linha representa uma amostra do dataset.
   - Colunas: `amostra_id`, `rotulo_classe`, `ground_truth_ih`, seguidas por uma coluna para cada métrica IHM (`ihm_kDN`, `ihm_DCP`, `ihm_TD`, `ihm_CLD`, `ihm_F1_IHM`, `ihm_N1_IHM`, `ihm_N2_IHM`, `ihm_LSC_IHM`, `ihm_LSR_IHM`, `ihm_U`, `ihm_H`, `ihm_Degree_IHM`, `ihm_Closeness_IHM`).

2. **`q1_spearman_correlations.csv`**:
   - Cada linha representa uma métrica IHM analisada em relação ao Ground Truth IH ($r_s$).
   - Colunas: `metrica_ihm`, `spearman_rho`, `p_valor`, `significante_alpha_0_05` (`SIM` ou `NAO`).

3. **`q2_classification_comparison.csv`**:
   - Cada linha representa um classificador do pool de modelos avaliado antes e depois da filtragem.
   - Colunas: `classificador`, `f1_score_baseline`, `f1_score_filtrado_ihm`, `diferenca_ganho_f1`.

4. **`q2_statistical_hypothesis_summary.csv`**:
   - Tabela síntese dos testes formais de significância estatística.
   - Linhas: Teste de Normalidade (Shapiro-Wilk), Teste Pareado (t-Student ou Wilcoxon), Nível de Significância ($\alpha=0.05$) e Tamanho do Efeito (Vargha-Delaney $A_{12}$ ou $d$ de Cohen).
   - Colunas: `metrica_avaliacao`, `valor`, `interpretacao`.
