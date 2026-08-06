from engine.harvest import extract_python_units

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


