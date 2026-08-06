import ast


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