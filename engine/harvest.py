import ast
import hashlib
import re


SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|token|password|passwd)\s*[=:]\s*['\"][^'\"]{8,}"),
    re.compile(r"-----BEGIN (RSA|EC|DSA|OPENSSH) PRIVATE KEY-----"),
    re.compile(r"(?i)aws_(access_key_id|secret_access_key)"),
    re.compile(r"sk[_-][A-Za-z0-9_]{10,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
]

def extract_python_units(source, path, repo):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines()
    units = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            units.append({
                "code": "\n".join(lines[node.lineno - 1:node.end_lineno]),
                "language": "python",
                "repo": repo,
                "path": path,
                "unit_type": "class" if isinstance(node, ast.ClassDef) else "function",
                "name": node.name
            })
    return units


def has_secret(code):
    return any(p.search(code) for p in SECRET_PATTERNS)


def passes_size(unit, min_lines, max_lines):
    n = unit["code"].count("\n") + 1
    return min_lines <= n <= max_lines


def _fingerprint(code):
    return hashlib.sha256("".join(code.split()).encode()).hexdigest()


def dedupe(units):
    seen, out = set(), []
    for u in units:
        fp = _fingerprint(u["code"])
        if fp not in seen:
            seen.add(fp)
            out.append(u)
    return out

