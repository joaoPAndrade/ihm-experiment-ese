# IHM Experiment - ESE

Projeto de pesquisa experimental em Engenharia de Software Experimental (ESE) e Interação Humano-Computador (IHM) com suporte a técnicas avançadas de Machine Learning e Deep Learning.

---

## 1. Configuração do Ambiente e Setup Rápido

O projeto utiliza **Python 3.14+** gerenciado via `venv`, com especificação declarativa de versão em `.python-version` e dependências padronizadas em `pyproject.toml` e `requirements.txt`.

### 1.1. Setup Automatizado do Ambiente Virtual
Para criar o ambiente virtual, instalar as dependências de desenvolvimento e registrar os Git Hooks do Quality Gate, execute:

```bash
chmod +x venv_setup.sh
./venv_setup.sh
```

### 1.2. Ativação do Ambiente Virtual
```bash
source .venv/bin/activate
```

### 1.3. Execução dos Testes Automatizados
```bash
pytest
```

---

## 2. Padrões de Engenharia e Arquitetura (.AGENTS)

O projeto adota diretrizes estritas de código limpo, modularidade e reprodutibilidade científica:

- **Uma Classe por Arquivo**: Cada módulo `.py` encapsula uma única classe com o mesmo nome em `snake_case`. Exemplo: `VersionProvider` reside exclusivamente em `src/utils/version_provider.py`.
- **Método `execute`**: O ponto de entrada e execução do algoritmo principal de cada classe é concentrado no método público `execute`.
- **Limite de Métodos e Linhas**: No máximo 4 métodos por classe (incluindo `__init__`, `execute` e auxiliares) e no máximo 30 linhas por método.
- **Zero Comentários Inline**: Ausência total de comentários (`#`) e docstrings (`\"\"\" \"\"\"`) no código-fonte Python; toda documentação técnica reside exclusivamente no `README.md`.
- **Tipagem Estática Completa**: 100% de parâmetros e retornos fortemente tipados (`typing`, primitivos, tipos de bibliotecas).
- **Quality Gate Obrigatório**: Cada alteração passa pela validação do script `scripts/quality_gate_validator.py` antes de qualquer commit.

---

## 3. Quality Gateway e Automação de Qualidade

O repositório possui validação contínua de padrões via Git Hooks e scripts automatizados:

- **Script Validador**: `scripts/quality_gate_validator.py`
  - Inspeciona a árvore sintática abstrata (AST) e tokens dos arquivos modificados.
  - Bloqueia comentários `#`, docstrings multilinhas, funções soltas, excesso de métodos (> 4) e métodos com mais de 30 linhas.
- **Git Hook de Pre-Commit**: `.githooks/pre-commit`
  - Acionado automaticamente a cada `git commit`.
  - Aborta a transação se qualquer diretriz for violada.
  - Para executar manualmente:
    ```bash
    python3 scripts/quality_gate_validator.py [arquivos...]
    ```

---

## 4. Skills e Automação de Agentes (`.agents/skills/`)

O repositório conta com skills especializadas para agentes autônomos (Antigravity):

1. **`planning` (`.agents/skills/planning/`)**:
   - Metodologia de decomposição de funcionalidades complexas em tarefas atômicas sequenciais (`.agents/tasks/task-N.md`).
   - Ciclo de validação "uma por vez" com checklist estrito de Definition of Done.
2. **`python-ml-dl-dev` (`.agents/skills/python-ml-dl-dev/`)**:
   - Diretrizes de desenvolvimento para pipelines de aprendizado de máquina e deep learning.
   - Padrões de reprodutibilidade com sementes e determinismo (`references/reproducibility_guide.md`).
   - Melhores práticas PyTorch com precisão mista AMP, `inference_mode` e gestão de memória (`references/pytorch_engineering.md`).
   - Prevenção de vazamento de dados (*Data Leakage*) e métricas balanceadas (`references/data_leakage_and_metrics.md`).
   - Testes de sanidade científica com verificação de tensores, gradientes e overfit em 1 batch (`references/testing_and_qa_ml.md`).

---

## 5. Módulos Implementados

### 5.1. Utilitários (`src/utils/`)

#### `VersionProvider` (`src/utils/version_provider.py`)
Componente responsável por recuperar a versão atual do projeto a partir do arquivo declarativo `VERSION`.

- **Inicialização (`__init__`)**:
  - `version_file: str = "VERSION"`: Caminho para o arquivo que contém a versão textual.
- **Método Principal (`execute`)**:
  - **Assinatura**: `execute(self) -> str`
  - **Retorno**: `str` contendo a versão semântica do projeto (ex.: `"0.1.0"`).
- **Método Auxiliar (`_read_from_file`)**:
  - **Assinatura**: `_read_from_file(self) -> str`
  - **Responsabilidade**: Lê e remove quebras de linha do arquivo de versão.
