#!/usr/bin/env bash
set -e

echo "=== Configuração do Ambiente Virtual (venv) - IHM Experiment ESE ==="

if ! command -v python3 &> /dev/null; then
    echo "❌ Erro: python3 não foi encontrado no sistema." >&2
    exit 1
fi

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Criando ambiente virtual em $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    echo "✅ Ambiente virtual criado com sucesso!"
else
    echo "ℹ️  Ambiente virtual $VENV_DIR já existe."
fi

if [ -f "requirements.txt" ]; then
    echo "📥 Instalando ferramentas de qualidade (requirements.txt)..."
    "$VENV_DIR/bin/pip" install -r requirements.txt || echo "⚠️ Aviso: Instalação offline ou sem rede."
fi

echo "🔧 Configurando Git Hooks para o Quality Gate..."
git config core.hooksPath .githooks || true
chmod +x .githooks/pre-commit || true

if [ -f "src/utils/version_provider.py" ]; then
    PROJECT_VERSION=$("$VENV_DIR/bin/python" src/utils/version_provider.py 2>/dev/null || python3 src/utils/version_provider.py || echo "0.1.0")
    echo "📌 Versão do projeto: v$PROJECT_VERSION"
fi

echo ""
echo "🎉 Setup concluído com sucesso!"
echo "Para ativar o ambiente virtual, execute:"
echo "    source .venv/bin/activate"
