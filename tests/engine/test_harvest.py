from engine.harvest import extract_python_units, has_secret, passes_size, dedupe


SRC = '''import os

def top(a, b):
    """Add."""
    return a + b

class Thing:
    def method(self):
        return 1

async def atop():
    return 2
'''


def test_extracts_top_level_units():
    units = extract_python_units(SRC, path="pkg/m.py", repo="demo")
    names = [u["name"] for u in units]
    assert names == ["top", "Thing", "atop"]
    assert units[0]["code"].startswith("def top(a, b):")
    assert units[0]["unit_type"] == "function"
    assert units[1]["unit_type"] == "class"
    assert "def method" in units[1]["code"]
    assert all(u["repo"] == "demo" and u["language"] == "python" for u in units)


def test_methods_not_extracted_separately():
    units = extract_python_units(SRC, path="m.py", repo="demo")
    assert "method" not in [u["name"] for u in units]


def test_syntax_error_returns_empty():
    assert extract_python_units("def broken(:", path="m.py", repo="demo") == []


def _unit(code):
    return {
        "code": code,
        "language": "python",
        "repo": "r",
        "path": "p",
        "unit_type": "function",
        "name": "f"
    }


def test_secret_detection():
    assert has_secret('API_KEY = "sk_live_abcdefgh1234"')
    assert has_secret("token: 'ghp_abcdefghijklmnopqrstuv'")
    assert has_secret("-----BEGIN RSA PRIVATE KEY-----")
    assert not has_secret("def add(a, b):\n    return a + b")


def test_size_filter():
    small = _unit("def f():\n    pass")
    ok = _unit("def f():\n    a = 1\n    return a")
    assert not passes_size(small, 3, 120)
    assert passes_size(ok, 3, 120)


def test_dedupe_ignores_whitespace():
    a = _unit("def f():\n    return 1")
    b = _unit("def f():\n           return 1")
    c = _unit("def g():\n    return 2")
    out = dedupe([a, b, c])
    assert len(out) == 2
    assert out[0] is a and out[1] is c
