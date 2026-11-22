import ast
import io
import os
import subprocess
import sys
import tokenize


class QualityGateValidator:
    def __init__(self, target_files: list[str] | None = None) -> None:
        self.target_files: list[str] = target_files or []
        self.errors: list[str] = []

    def _check_comments_and_docstrings(self, filepath: str, source_bytes: bytes) -> list[str]:
        file_errors: list[str] = []
        try:
            tokens = tokenize.tokenize(io.BytesIO(source_bytes).readline)
            for tok in tokens:
                if tok.type == tokenize.COMMENT:
                    file_errors.append(f"{filepath}:{tok.start[0]}: Proibido comentário com '#'")
        except tokenize.TokenError as token_error:
            file_errors.append(f"{filepath}: Erro de tokenizacao: {token_error}")
        try:
            tree = ast.parse(source_bytes)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    if ast.get_docstring(node, clean=False) is not None:
                        file_errors.append(f"{filepath}:{node.lineno}: Proibido docstring/comentario multilinha")
                elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    file_errors.append(f"{filepath}:{node.lineno}: Proibido string literal solta/comentario")
        except SyntaxError as syntax_error:
            file_errors.append(f"{filepath}: Erro de sintaxe: {syntax_error}")
        return file_errors

    def _check_class_structure(self, filepath: str, tree: ast.AST) -> list[str]:
        if os.path.basename(filepath) == "__init__.py":
            return []
        file_errors: list[str] = []
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if functions:
            file_errors.append(f"{filepath}: Proibido funcoes soltas. Use classes com metodo 'execute'")
        if len(classes) != 1:
            file_errors.append(f"{filepath}: Arquivo deve conter exatamente 1 classe (encontradas: {len(classes)})")
            return file_errors
        cls = classes[0]
        methods = [node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if len(methods) > 4:
            file_errors.append(f"{filepath}: Classe '{cls.name}' possui {len(methods)} metodos (maximo permitido: 4)")
        if not any(method.name == "execute" for method in methods):
            file_errors.append(f"{filepath}: Classe '{cls.name}' deve conter o metodo 'execute'")
        for method in methods:
            length = (method.end_lineno or method.lineno) - method.lineno + 1
            if length > 30:
                file_errors.append(f"{filepath}:{method.lineno}: Metodo '{method.name}' tem {length} linhas (maximo: 30)")
            missing = [a.arg for a in method.args.args if a.arg not in ("self", "cls") and a.annotation is None]
            if missing or method.returns is None:
                file_errors.append(f"{filepath}:{method.lineno}: Metodo '{method.name}' sem tipagem estrita completa")
        return file_errors

    def execute(self) -> int:
        files = list(self.target_files)
        if not files:
            cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"]
            output = subprocess.run(cmd, capture_output=True, text=True).stdout
            files = [line.strip() for line in output.splitlines() if line.strip().endswith(".py")]
        for filepath in files:
            if not os.path.isfile(filepath):
                continue
            with open(filepath, "rb") as stream:
                content = stream.read()
            self.errors.extend(self._check_comments_and_docstrings(filepath, content))
            try:
                tree = ast.parse(content)
                self.errors.extend(self._check_class_structure(filepath, tree))
            except SyntaxError:
                pass
        if self.errors:
            print("\033[91m[QUALITY GATE REPROVADO] Violações das regras do projeto:\033[0m", file=sys.stderr)
            for err in self.errors:
                print(f"  ❌ {err}", file=sys.stderr)
            return 1
        print("\033[92m[QUALITY GATE APROVADO] Todas as regras foram cumpridas com sucesso!\033[0m")
        return 0


if __name__ == "__main__":
    validator = QualityGateValidator(sys.argv[1:])
    sys.exit(validator.execute())
