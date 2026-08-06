# Style-Tuning Engine Implementation Plan

> **Execution mode (overrides the default handoff):** This plan is executed in
> **code-by-hand sessions** (soltero-skills `code-by-hand`): the navigator (agent)
> presents each step's block; the owner types every line into the real files and runs
> every command. The agent never writes code files. Steps use checkbox (`- [ ]`)
> syntax for tracking; `.code-by-hand.md` at repo root tracks the live session position.

**Goal:** A fully-local, config-driven fine-tuning engine (harvest → dataset → tune → eval → report) that tunes Qwen2.5-Coder models toward the owner's coding style.

**Architecture:** Five independently runnable scripts under `engine/`, each reading YAML configs from `configs/` and producing inspectable file outputs under gitignored `data/` and `models/`. Stage 1 validates everything with Qwen2.5-Coder-3B bf16 LoRA on native Windows; Stage 2 flips a config to 7B QLoRA on WSL2.

**Tech Stack:** Python 3.12 (`modelEnvGpu`), PyTorch 2.9.1+rocm7.2.1, transformers, peft, datasets, pyyaml, pytest, ruff.

**Spec:** `Docs/superpowers/specs/2026-08-06-style-tuning-engine-design.md`

## Global Constraints

- All Python runs via the explicit venv path: `& "modelEnvGpu\Scripts\python.exe"` — never bare `python`.
- PowerShell 5.1: no `&&`; chain with `;` or `if ($?)`.
- The **owner runs all pip/training/benchmark commands** in his own terminal and pastes results back.
- Fully local: no paid API calls anywhere.
- All training uses bf16 autocast (`bf16=True`) — the experiment-02 confirmed win.
- `greystonedb` is denylisted at harvest; harvest must refuse it even if configured.
- Corpus and adapters live under `data/` and `models/` (both already gitignored). Verify nothing under them is ever committed.
- Working test command from repo root: `& "modelEnvGpu\Scripts\python.exe" -m pytest tests -v` (running from repo root puts `engine/` on `sys.path`).
- Commit after every green task; messages in the repo's existing imperative style.

---

### Task 1: Environment prerequisites (owner-run, no code)

**Files:** none.

**Interfaces:**
- Produces: `modelEnvGpu` with `transformers`, `peft`, `datasets`, `accelerate`, `pyyaml`, `pytest`, `ruff` importable.

- [ ] **Step 1: Install packages** (owner runs; ~2 GB of wheels)

```powershell
& "modelEnvGpu\Scripts\python.exe" -m pip install transformers peft datasets accelerate pyyaml pytest ruff
```

- [ ] **Step 2: Verify imports and GPU**

```powershell
& "modelEnvGpu\Scripts\python.exe" -c "import transformers, peft, datasets, yaml, pytest; import torch; print(transformers.__version__, peft.__version__, torch.cuda.is_available())"
```

Expected: three values ending in `True`. (Ignore the cosmetic `Francisco: Unknown command line argument` line — known ROCm path-with-space issue.)

---

### Task 2: Package skeleton + config loader

**Files:**
- Create: `engine/__init__.py` (empty), `engine/config.py`
- Test: `tests/engine/test_config.py`

**Interfaces:**
- Produces: `load_config(path, required=()) -> dict` — parses YAML, raises `ConfigError` (subclass of `Exception`) if the file is missing or any key in `required` is absent. Every later CLI consumes this.

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/test_config.py
import pytest
from engine.config import load_config, ConfigError


def test_loads_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("out: data/x.jsonl\nmin_lines: 3\n", encoding="utf-8")
    cfg = load_config(p, required=("out",))
    assert cfg["out"] == "data/x.jsonl"
    assert cfg["min_lines"] == 3


def test_missing_file_raises():
    with pytest.raises(ConfigError):
        load_config("no/such/file.yaml")


def test_missing_required_key_raises(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("out: x\n", encoding="utf-8")
    with pytest.raises(ConfigError) as e:
        load_config(p, required=("out", "repos"))
    assert "repos" in str(e.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& "modelEnvGpu\Scripts\python.exe" -m pytest tests\engine\test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'engine'` (or `engine.config`).

- [ ] **Step 3: Write minimal implementation**

```python
# engine/config.py
from pathlib import Path

import yaml


class ConfigError(Exception):
    pass


def load_config(path, required=()):
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ConfigError(f"{p} missing required keys: {missing}")
    return cfg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& "modelEnvGpu\Scripts\python.exe" -m pytest tests\engine\test_config.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```powershell
git add engine tests; git commit -m "Add engine package with YAML config loader"
```

---

### Task 3: Harvest — Python unit extraction

**Files:**
- Create: `engine/harvest.py`
- Test: `tests/engine/test_harvest.py`

**Interfaces:**
- Produces: `extract_python_units(source: str, path: str, repo: str) -> list[dict]` — one dict per **top-level** function/async function/class: `{"code", "language": "python", "repo", "path", "unit_type": "function"|"class", "name"}`. Unparseable source returns `[]`. Methods ship inside their class unit, not separately.

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/test_harvest.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& "modelEnvGpu\Scripts\python.exe" -m pytest tests\engine\test_harvest.py -v`
Expected: FAIL — `ImportError` (no `engine.harvest`).

- [ ] **Step 3: Write minimal implementation**

```python
# engine/harvest.py
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
                "name": node.name,
            })
    return units
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& "modelEnvGpu\Scripts\python.exe" -m pytest tests\engine\test_harvest.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```powershell
git add engine\harvest.py tests\engine\test_harvest.py; git commit -m "Harvest: extract top-level Python units via ast"
```

---

### Task 4: Harvest — secrets scan, size filter, dedupe

**Files:**
- Modify: `engine/harvest.py` (append)
- Test: `tests/engine/test_harvest.py` (append)

**Interfaces:**
- Produces: `has_secret(code: str) -> bool`; `passes_size(unit: dict, min_lines: int, max_lines: int) -> bool`; `dedupe(units: list[dict]) -> list[dict]` (keeps first occurrence, whitespace-insensitive fingerprint). The CLI (Task 6) counts secret-flagged units separately so they are *reported*, per spec.

- [ ] **Step 1: Write the failing tests** (append to `tests/engine/test_harvest.py`)

```python
from engine.harvest import has_secret, passes_size, dedupe


def _unit(code):
    return {"code": code, "language": "python", "repo": "r", "path": "p", "unit_type": "function", "name": "f"}


def test_secret_detection():
    assert has_secret('API_KEY = "sk_live_abcdefgh1234"')
    assert has_secret("token: 'ghp_abcdefghijklmnopqrstuv'")
    assert has_secret("-----BEGIN RSA PRIVATE KEY-----")
    assert not has_secret("def add(a, b):\n    return a + b")


def test_size_filter():
    small = _unit("def f():\n    pass")          # 2 lines
    ok = _unit("def f():\n    a = 1\n    return a")  # 3 lines
    assert not passes_size(small, 3, 120)
    assert passes_size(ok, 3, 120)


def test_dedupe_ignores_whitespace():
    a = _unit("def f():\n    return 1")
    b = _unit("def f():\n        return 1")
    c = _unit("def g():\n    return 2")
    out = dedupe([a, b, c])
    assert len(out) == 2
    assert out[0] is a and out[1] is c
```

- [ ] **Step 2: Run to verify failure**

Run: `& "modelEnvGpu\Scripts\python.exe" -m pytest tests\engine\test_harvest.py -v`
Expected: FAIL — `ImportError: cannot import name 'has_secret'`.

- [ ] **Step 3: Implement** (append to `engine/harvest.py`; add `import hashlib`, `import re` at top)

```python
SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|token|password|passwd)\s*[=:]\s*['\"][^'\"]{8,}"),
    re.compile(r"-----BEGIN (RSA|EC|DSA|OPENSSH) PRIVATE KEY-----"),
    re.compile(r"(?i)aws_(access_key_id|secret_access_key)"),
    re.compile(r"sk[_-][A-Za-z0-9_]{10,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
]


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
```

- [ ] **Step 4: Run to verify pass**

Run: `& "modelEnvGpu\Scripts\python.exe" -m pytest tests\engine\test_harvest.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```powershell
git add engine\harvest.py tests\engine\test_harvest.py; git commit -m "Harvest: secrets scan, size filter, whitespace-insensitive dedupe"
```

---

### Task 5: Harvest — TypeScript heuristic extraction

**Files:**
- Modify: `engine/harvest.py` (append)
- Test: `tests/engine/test_harvest.py` (append)

**Interfaces:**
- Produces: `extract_ts_units(source: str, path: str, repo: str) -> list[dict]` — same dict shape, `language: "typescript"`, `unit_type: "unit"`, `name` = first 60 chars of the declaration line. **Known heuristic:** brace-counting; braces inside string literals can mis-split — acceptable for corpus building (bad splits get filtered/deduped downstream).

- [ ] **Step 1: Write the failing tests** (append)

```python
from engine.harvest import extract_ts_units

TS_SRC = '''import { x } from "./x";

export function add(a: number, b: number): number {
  return a + b;
}

const mul = (a: number, b: number) => {
  return a * b;
};

export class Store {
  private items: string[] = [];
  add(item: string) {
    this.items.push(item);
  }
}
'''


def test_extracts_ts_units():
    units = extract_ts_units(TS_SRC, path="src/m.ts", repo="demo")
    assert len(units) == 3
    assert units[0]["code"].startswith("export function add")
    assert units[0]["code"].rstrip().endswith("}")
    assert "this.items.push" in units[2]["code"]
    assert all(u["language"] == "typescript" for u in units)


def test_ts_import_lines_skipped():
    units = extract_ts_units(TS_SRC, path="m.ts", repo="demo")
    assert not any("import {" in u["code"].splitlines()[0] for u in units)
```

- [ ] **Step 2: Run to verify failure**

Run: `& "modelEnvGpu\Scripts\python.exe" -m pytest tests\engine\test_harvest.py -v`
Expected: FAIL — `ImportError: cannot import name 'extract_ts_units'`.

- [ ] **Step 3: Implement** (append to `engine/harvest.py`)

```python
TS_DECL = re.compile(
    r"^(export\s+)?(default\s+)?(async\s+)?"
    r"(function\s+\w+|class\s+\w+|(const|let)\s+\w+\s*=\s*(async\s*)?\()"
)


def extract_ts_units(source, path, repo):
    lines = source.splitlines()
    units, i = [], 0
    while i < len(lines):
        if not TS_DECL.match(lines[i].strip()):
            i += 1
            continue
        depth, j, started = 0, i, False
        while j < len(lines):
            depth += lines[j].count("{") - lines[j].count("}")
            if "{" in lines[j]:
                started = True
            if started and depth <= 0:
                break
            j += 1
        if started and j < len(lines):
            units.append({
                "code": "\n".join(lines[i:j + 1]),
                "language": "typescript",
                "repo": repo,
                "path": path,
                "unit_type": "unit",
                "name": lines[i].strip()[:60],
            })
        i = j + 1
    return units
```

- [ ] **Step 4: Run to verify pass**

Run: `& "modelEnvGpu\Scripts\python.exe" -m pytest tests\engine\test_harvest.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```powershell
git add engine\harvest.py tests\engine\test_harvest.py; git commit -m "Harvest: heuristic TypeScript unit extraction"
```

---

### Task 6: Harvest — repo walker, denylist, CLI

**Files:**
- Modify: `engine/harvest.py` (append; add `import argparse, json, sys` and `from pathlib import Path` at top)
- Create: `configs/harvest.yaml`
- Test: `tests/engine/test_harvest.py` (append)

**Interfaces:**
- Consumes: `load_config` (Task 2), all Task 3–5 functions.
- Produces: `DENYLIST = {"greystonedb"}`; `iter_source_files(root: Path) -> Iterator[Path]` (yields `.py`/`.ts` files, skipping `SKIP_DIRS`); `harvest_repo(root: Path) -> tuple[list[dict], int]` returning (kept units, secret_flagged_count); CLI `python -m engine.harvest --config configs/harvest.yaml` writing JSONL to `cfg["out"]` and printing a per-repo summary. **Denylisted repo in config → `SystemExit` with non-zero code, nothing written.**

- [ ] **Step 1: Write the failing tests** (append)

```python
import json
import subprocess
import sys

import pytest

from engine.harvest import DENYLIST, iter_source_files, harvest_repo


def _mkrepo(tmp_path, name):
    r = tmp_path / name
    (r / "node_modules" / "junk").mkdir(parents=True)
    (r / "node_modules" / "junk" / "lib.py").write_text("def vendored():\n    return 0\n")
    (r / "src").mkdir()
    (r / "src" / "app.py").write_text(
        "def real(a):\n    b = a + 1\n    return b\n\n"
        'def leaky():\n    key = "sk_live_abcdefgh1234"\n    return key\n'
    )
    (r / "src" / "ui.ts").write_text(
        "export function go(n: number) {\n  return n + 1;\n}\n"
    )
    return r


def test_walker_skips_vendored_dirs(tmp_path):
    r = _mkrepo(tmp_path, "demo")
    files = sorted(p.name for p in iter_source_files(r))
    assert files == ["app.py", "ui.ts"]


def test_harvest_repo_drops_and_counts_secrets(tmp_path):
    r = _mkrepo(tmp_path, "demo")
    units, flagged = harvest_repo(r)
    assert flagged == 1
    assert sorted(u["name"] for u in units if u["language"] == "python") == ["real"]
    assert any(u["language"] == "typescript" for u in units)


def test_denylist_contains_greystonedb():
    assert "greystonedb" in DENYLIST


def test_cli_refuses_denylisted_repo(tmp_path):
    r = _mkrepo(tmp_path, "greystonedb")
    cfg = tmp_path / "h.yaml"
    out = tmp_path / "raw.jsonl"
    cfg.write_text(f"repos:\n  - {r}\nout: {out}\nmin_lines: 3\nmax_lines: 120\n")
    proc = subprocess.run(
        [sys.executable, "-m", "engine.harvest", "--config", str(cfg)],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert not out.exists()


def test_cli_writes_jsonl(tmp_path):
    r = _mkrepo(tmp_path, "demo")
    cfg = tmp_path / "h.yaml"
    out = tmp_path / "raw.jsonl"
    cfg.write_text(f"repos:\n  - {r}\nout: {out}\nmin_lines: 3\nmax_lines: 120\n")
    proc = subprocess.run(
        [sys.executable, "-m", "engine.harvest", "--config", str(cfg)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert {r_["repo"] for r_ in rows} == {"demo"}
    assert "flagged" in proc.stdout
```

- [ ] **Step 2: Run to verify failure**

Run: `& "modelEnvGpu\Scripts\python.exe" -m pytest tests\engine\test_harvest.py -v`
Expected: FAIL — `ImportError: cannot import name 'DENYLIST'`.

- [ ] **Step 3: Implement** (append to `engine/harvest.py`)

```python
SKIP_DIRS = {
    ".git", "node_modules", "dist", "build", "__pycache__", ".venv", "venv",
    "modelEnv", "modelEnvGpu", "data", "models", "vendor", ".next", "coverage",
}
DENYLIST = {"greystonedb"}
EXTRACTORS = {".py": extract_python_units, ".ts": extract_ts_units}


def iter_source_files(root):
    for p in sorted(Path(root).rglob("*")):
        if p.suffix in EXTRACTORS and not any(part in SKIP_DIRS for part in p.parts):
            yield p


def harvest_repo(root, min_lines=3, max_lines=120):
    root = Path(root)
    kept, flagged = [], 0
    for f in iter_source_files(root):
        try:
            source = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = f.relative_to(root).as_posix()
        for u in EXTRACTORS[f.suffix](source, path=rel, repo=root.name):
            if has_secret(u["code"]):
                flagged += 1
            elif passes_size(u, min_lines, max_lines):
                kept.append(u)
    return kept, flagged


def main(argv=None):
    from engine.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, required=("repos", "out"))

    for r in cfg["repos"]:
        if Path(r).name.lower() in DENYLIST:
            sys.exit(f"REFUSED: {r} is denylisted")

    all_units = []
    for r in cfg["repos"]:
        units, flagged = harvest_repo(
            r, cfg.get("min_lines", 3), cfg.get("max_lines", 120)
        )
        all_units.extend(units)
        print(f"{Path(r).name}: {len(units)} units kept, {flagged} flagged (secrets, dropped)")

    all_units = dedupe(all_units)
    out = Path(cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for u in all_units:
            f.write(json.dumps(u) + "\n")
    print(f"TOTAL: {len(all_units)} units -> {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `& "modelEnvGpu\Scripts\python.exe" -m pytest tests\engine\test_harvest.py -v`
Expected: 13 passed.

- [ ] **Step 5: Create `configs/harvest.yaml`** (owner adjusts the repo list to what's cloned locally; clone missing ones first)

```yaml
# Harvest sources. greystonedb is denylisted in code — do not add it.
repos:
  - C:\Users\Francisco Soltero\Desktop\LocalModels
  - C:\Users\Francisco Soltero\Desktop\LastCall
  - C:\Users\Francisco Soltero\Desktop\gamemaster
  - C:\Users\Francisco Soltero\Desktop\PostProject
  - C:\Users\Francisco Soltero\Desktop\quantum
out: data\style_corpus\raw.jsonl
min_lines: 3
max_lines: 120
```

- [ ] **Step 6: Owner smoke-run on the real repos**

```powershell
& "modelEnvGpu\Scripts\python.exe" -m engine.harvest --config configs\harvest.yaml
```

Expected: per-repo `N units kept, M flagged` lines + `TOTAL: ... -> data\style_corpus\raw.jsonl`. Paste the summary back — corpus size decides whether OSS augmentation is needed now or later.

- [ ] **Step 7: Commit**

```powershell
git add engine\harvest.py tests\engine\test_harvest.py configs\harvest.yaml; git commit -m "Harvest: repo walker with denylist and JSONL CLI"
```

---

### Task 7: Dataset — backtranslation core (model-free, injectable generator)

**Files:**
- Create: `engine/dataset.py`
- Test: `tests/engine/test_dataset.py`

**Interfaces:**
- Consumes: harvest row dicts.
- Produces: `build_instruction_prompt(unit: dict) -> str`; `clean_instruction(text: str) -> str` (strip whitespace/quotes); `keep_pair(instruction: str, code: str) -> bool` (10–400 chars, no code fences); `backtranslate(units, generate: Callable[[str], str]) -> list[dict]` returning rows `{"instruction", "code", "repo", "language", "source": "owner"}`. The generator is injected so tests never load a model.

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/test_dataset.py
from engine.dataset import backtranslate, build_instruction_prompt, clean_instruction, keep_pair

UNIT = {"code": "def add(a, b):\n    return a + b", "language": "python",
        "repo": "demo", "path": "m.py", "unit_type": "function", "name": "add"}


def test_prompt_contains_code_and_language():
    p = build_instruction_prompt(UNIT)
    assert "def add(a, b):" in p
    assert "python" in p


def test_clean_instruction():
    assert clean_instruction('  "Write an add function."  ') == "Write an add function."
    assert clean_instruction("") == ""


def test_keep_pair_rejects_junk():
    assert keep_pair("Write a function that adds two numbers.", UNIT["code"])
    assert not keep_pair("short", UNIT["code"])
    assert not keep_pair("x" * 500, UNIT["code"])
    assert not keep_pair("```python\ncode\n```", UNIT["code"])


def test_backtranslate_pairs_real_code_with_generated_instruction():
    fake = lambda prompt: "Write a Python function that adds two numbers."
    rows = backtranslate([UNIT], fake)
    assert rows == [{
        "instruction": "Write a Python function that adds two numbers.",
        "code": UNIT["code"], "repo": "demo", "language": "python", "source": "owner",
    }]


def test_backtranslate_drops_filtered_pairs():
    assert backtranslate([UNIT], lambda p: "no") == []
```

- [ ] **Step 2: Run to verify failure**

Run: `& "modelEnvGpu\Scripts\python.exe" -m pytest tests\engine\test_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'engine.dataset'`.

- [ ] **Step 3: Implement**

```python
# engine/dataset.py
INSTRUCTION_PROMPT = """Read the following {language} code and write the single, \
specific instruction a developer would have given to produce exactly this code. \
Reply with the instruction only - no preamble, no code.

```{language}
{code}
```"""


def build_instruction_prompt(unit):
    return INSTRUCTION_PROMPT.format(language=unit["language"], code=unit["code"])


def clean_instruction(text):
    return text.strip().strip('"').strip()


def keep_pair(instruction, code):
    return 10 <= len(instruction) <= 400 and "```" not in instruction


def backtranslate(units, generate, source="owner"):
    rows = []
    for u in units:
        instruction = clean_instruction(generate(build_instruction_prompt(u)))
        if keep_pair(instruction, u["code"]):
            rows.append({
                "instruction": instruction,
                "code": u["code"],
                "repo": u["repo"],
                "language": u["language"],
                "source": source,
            })
    return rows
```

- [ ] **Step 4: Run to verify pass**

Run: `& "modelEnvGpu\Scripts\python.exe" -m pytest tests\engine\test_dataset.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```powershell
git add engine\dataset.py tests\engine\test_dataset.py; git commit -m "Dataset: backtranslation core with injectable generator"
```

---

### Task 8: Dataset — splits, weights, stats, CLI with local generator

**Files:**
- Modify: `engine/dataset.py` (append; add `import argparse, json, random` and `from collections import Counter`, `from pathlib import Path` at top)
- Create: `configs/dataset.yaml`
- Test: `tests/engine/test_dataset.py` (append)

**Interfaces:**
- Consumes: `load_config`, `backtranslate`, harvest JSONL.
- Produces: `split_by_repo(rows, test_repos: list, val_repos: list) -> (train, val, test)`; `apply_source_weights(rows, weights: dict, seed=13) -> list` (deterministic downsampling of the **train** split only; weight 1.0 keeps everything); `corpus_stats(rows) -> dict[(repo, language), int]`; CLI `python -m engine.dataset --config configs/dataset.yaml` reading `raw`, writing `train.jsonl`/`val.jsonl`/`test.jsonl` into `out_dir`, printing stats. The real generator loads `generator_model` with transformers (greedy decode, `max_new_tokens=80`, bf16 on `cuda`).

- [ ] **Step 1: Write the failing tests** (append)

```python
from engine.dataset import apply_source_weights, corpus_stats, split_by_repo


def _row(repo, source="owner"):
    return {"instruction": "Write a thing that works properly.", "code": "def f():\n    return 1",
            "repo": repo, "language": "python", "source": source}


def test_split_is_by_whole_repo():
    rows = [_row("a"), _row("a"), _row("b"), _row("c"), _row("d")]
    train, val, test = split_by_repo(rows, test_repos=["c"], val_repos=["b"])
    assert [r["repo"] for r in train] == ["a", "a", "d"]
    assert [r["repo"] for r in val] == ["b"]
    assert [r["repo"] for r in test] == ["c"]


def test_source_weights_downsample_deterministically():
    rows = [_row("a", source="oss") for _ in range(200)] + [_row("a") for _ in range(50)]
    out1 = apply_source_weights(rows, {"owner": 1.0, "oss": 0.3}, seed=13)
    out2 = apply_source_weights(rows, {"owner": 1.0, "oss": 0.3}, seed=13)
    assert out1 == out2
    oss = [r for r in out1 if r["source"] == "oss"]
    assert 30 <= len(oss) <= 90                       # ~60 expected at 0.3
    assert len([r for r in out1 if r["source"] == "owner"]) == 50


def test_corpus_stats_counts_by_repo_and_language():
    stats = corpus_stats([_row("a"), _row("a"), _row("b")])
    assert stats[("a", "python")] == 2
    assert stats[("b", "python")] == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `& "modelEnvGpu\Scripts\python.exe" -m pytest tests\engine\test_dataset.py -v`
Expected: FAIL — `ImportError: cannot import name 'apply_source_weights'`.

- [ ] **Step 3: Implement** (append to `engine/dataset.py`)

```python
def split_by_repo(rows, test_repos, val_repos):
    train, val, test = [], [], []
    for r in rows:
        if r["repo"] in test_repos:
            test.append(r)
        elif r["repo"] in val_repos:
            val.append(r)
        else:
            train.append(r)
    return train, val, test


def apply_source_weights(rows, weights, seed=13):
    rng = random.Random(seed)
    return [r for r in rows if rng.random() < weights.get(r.get("source", "owner"), 1.0)]


def corpus_stats(rows):
    return Counter((r["repo"], r["language"]) for r in rows)


def make_generator(model_name, device="cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device)

    def generate(prompt):
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        ids = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=80, do_sample=False)
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

    return generate


def main(argv=None):
    from engine.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, required=("raw", "out_dir", "generator_model", "test_repos", "val_repos"))

    units = [json.loads(l) for l in Path(cfg["raw"]).read_text(encoding="utf-8").splitlines()]
    print(f"backtranslating {len(units)} units with {cfg['generator_model']} ...")
    rows = backtranslate(units, make_generator(cfg["generator_model"]))
    print(f"kept {len(rows)} pairs after filtering")

    train, val, test = split_by_repo(rows, cfg["test_repos"], cfg["val_repos"])
    train = apply_source_weights(train, cfg.get("source_weights", {"owner": 1.0}))

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in (("train", train), ("val", val), ("test", test)):
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in subset:
                f.write(json.dumps(r) + "\n")
        print(f"{name}: {len(subset)} pairs")
    for (repo, lang), n in sorted(corpus_stats(train).items()):
        print(f"  train {repo}/{lang}: {n}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `& "modelEnvGpu\Scripts\python.exe" -m pytest tests\engine\test_dataset.py -v`
Expected: 8 passed.

- [ ] **Step 5: Create `configs/dataset.yaml`** (owner picks final held-out repos once harvest stats from Task 6 are known)

```yaml
raw: data\style_corpus\raw.jsonl
out_dir: data\style_corpus
generator_model: Qwen/Qwen2.5-3B-Instruct   # already cached from inference/qwen_chat.py
test_repos: [sqlite-explorer-mcp]
val_repos: [PostProject]
source_weights:
  owner: 1.0
  oss: 0.3
```

- [ ] **Step 6: Owner runs backtranslation** (slow — the 3B generates one instruction per unit)

```powershell
& "modelEnvGpu\Scripts\python.exe" -m engine.dataset --config configs\dataset.yaml
```

Expected: kept/filtered counts, split sizes, per-repo stats. **Checkpoint (spec §4): owner and agent review ~20 random pairs from `train.jsonl` for instruction quality before any training run.**

- [ ] **Step 7: Commit**

```powershell
git add engine\dataset.py tests\engine\test_dataset.py configs\dataset.yaml; git commit -m "Dataset: repo-level splits, source weights, backtranslation CLI"
```

---

### Task 9: Tune — example formatting with prompt masking

**Files:**
- Create: `engine/tune.py`
- Test: `tests/engine/test_tune.py`

**Interfaces:**
- Produces: `PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"`; `format_example(row: dict, tokenizer, max_len=1024) -> {"input_ids": list[int], "labels": list[int]}` — labels are `-100` for every prompt token (loss on code tokens only; same masking reused by eval's held-out loss so tuned-vs-base numbers are comparable), code tokens + EOS supervised, both lists truncated to `max_len`. Tokenizer contract used: `tokenizer(text, add_special_tokens=False)["input_ids"]` and `tokenizer.eos_token`.

- [ ] **Step 1: Write the failing test** (uses a stub tokenizer — no downloads)

```python
# tests/engine/test_tune.py
from engine.tune import PROMPT_TEMPLATE, format_example


class StubTokenizer:
    eos_token = "<eos>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


ROW = {"instruction": "Add two numbers.", "code": "def add(a, b):\n    return a + b"}


def test_prompt_tokens_are_masked():
    tok = StubTokenizer()
    ex = format_example(ROW, tok, max_len=4096)
    prompt_len = len(PROMPT_TEMPLATE.format(instruction=ROW["instruction"]))
    assert ex["labels"][:prompt_len] == [-100] * prompt_len
    assert all(l != -100 for l in ex["labels"][prompt_len:])
    assert len(ex["input_ids"]) == len(ex["labels"])


def test_code_and_eos_are_supervised():
    tok = StubTokenizer()
    ex = format_example(ROW, tok, max_len=4096)
    code_and_eos = ROW["code"] + tok.eos_token
    assert ex["input_ids"][-len(code_and_eos):] == [ord(c) for c in code_and_eos]


def test_truncation():
    tok = StubTokenizer()
    ex = format_example(ROW, tok, max_len=10)
    assert len(ex["input_ids"]) == 10 and len(ex["labels"]) == 10
```

- [ ] **Step 2: Run to verify failure**

Run: `& "modelEnvGpu\Scripts\python.exe" -m pytest tests\engine\test_tune.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'engine.tune'`.

- [ ] **Step 3: Implement**

```python
# engine/tune.py
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def format_example(row, tokenizer, max_len=1024):
    prompt = PROMPT_TEMPLATE.format(instruction=row["instruction"])
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    code_ids = tokenizer(row["code"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    return {
        "input_ids": (prompt_ids + code_ids)[:max_len],
        "labels": ([-100] * len(prompt_ids) + code_ids)[:max_len],
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `& "modelEnvGpu\Scripts\python.exe" -m pytest tests\engine\test_tune.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```powershell
git add engine\tune.py tests\engine\test_tune.py; git commit -m "Tune: SFT example formatting with prompt-token masking"
```

---

### Task 10: Tune — config-driven LoRA trainer CLI

**Files:**
- Modify: `engine/tune.py` (append; add `import argparse, json, time` and `from pathlib import Path` at top)
- Create: `configs/coder3b-lora.yaml`, `configs/smoke-lora.yaml`
- Test: verification is a **owner-run smoke train** (loading a real model in pytest is not feasible; the pure logic was tested in Task 9).

**Interfaces:**
- Consumes: `load_config`, `format_example`, `data_dir/train.jsonl` + `val.jsonl` (Task 8 output).
- Produces: CLI `python -m engine.tune --config <yaml>` that trains a LoRA adapter and saves it to `models/adapters/<tag>/`, printing `wall_clock_s`, `tok_s` (supervised tokens/sec), and `vram_gb` (`torch.cuda.max_memory_allocated()/2**30`). Config keys (exact): `tag, base_model, data_dir, lora_r, lora_alpha, lora_dropout, target_modules, learning_rate, epochs, batch_size, grad_accum, max_len, grad_checkpointing, device`.

- [ ] **Step 1: Implement the trainer** (append to `engine/tune.py`)

```python
def load_rows(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()]


def main(argv=None):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments)

    from engine.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, required=("tag", "base_model", "data_dir"))

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16
    ).to(cfg.get("device", "cuda"))
    if cfg.get("grad_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model = get_peft_model(model, LoraConfig(
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg.get("target_modules",
                               ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"]),
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    data_dir = Path(cfg["data_dir"])
    fmt = lambda row: format_example(row, tok, cfg.get("max_len", 1024))
    train_ds = Dataset.from_list([fmt(r) for r in load_rows(data_dir / "train.jsonl")])
    val_ds = Dataset.from_list([fmt(r) for r in load_rows(data_dir / "val.jsonl")])

    out_dir = Path("models") / "adapters" / cfg["tag"]
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out_dir / "checkpoints"),
            per_device_train_batch_size=cfg.get("batch_size", 2),
            gradient_accumulation_steps=cfg.get("grad_accum", 8),
            num_train_epochs=cfg.get("epochs", 2),
            learning_rate=cfg.get("learning_rate", 2e-4),
            bf16=True,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="no",
            report_to=[],
        ),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    wall = time.time() - t0

    supervised = sum(sum(1 for l in ex["labels"] if l != -100) for ex in train_ds)
    model.save_pretrained(out_dir)
    print(f"tag={cfg['tag']} wall_clock_s={wall:.1f} "
          f"tok_s={supervised * cfg.get('epochs', 2) / wall:.0f} "
          f"vram_gb={torch.cuda.max_memory_allocated() / 2**30:.2f} "
          f"adapter={out_dir}")


if __name__ == "__main__":
    main()
```

Note: if the installed transformers rejects `eval_strategy`, the older spelling is `evaluation_strategy` — one-word fix, owner types it.

- [ ] **Step 2: Create `configs/smoke-lora.yaml`** (tiny model, proves the plumbing in minutes)

```yaml
tag: smoke-lora
base_model: Qwen/Qwen2.5-Coder-0.5B-Instruct
data_dir: data\style_corpus
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
learning_rate: 2.0e-4
epochs: 1
batch_size: 2
grad_accum: 4
max_len: 512
grad_checkpointing: false
device: cuda
```

- [ ] **Step 3: Create `configs/coder3b-lora.yaml`** (the Stage 1 real run)

```yaml
tag: coder3b-lora-r16
base_model: Qwen/Qwen2.5-Coder-3B-Instruct
data_dir: data\style_corpus
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
learning_rate: 2.0e-4
epochs: 2
batch_size: 2
grad_accum: 8
max_len: 1024
grad_checkpointing: true
device: cuda
```

- [ ] **Step 4: Owner smoke-runs the trainer** (downloads the 0.5B on first run)

```powershell
& "modelEnvGpu\Scripts\python.exe" -m engine.tune --config configs\smoke-lora.yaml
```

Expected: trainable-parameter printout (~0.5–2% of total), falling loss in the log, final line with `wall_clock_s / tok_s / vram_gb`, adapter files in `models\adapters\smoke-lora\`. Paste the output back.

- [ ] **Step 5: Commit**

```powershell
git add engine\tune.py configs\smoke-lora.yaml configs\coder3b-lora.yaml; git commit -m "Tune: config-driven LoRA trainer CLI with smoke and 3B configs"
```

---

### Task 11: Eval — held-out style loss

**Files:**
- Create: `engine/eval.py`
- Test: `tests/engine/test_eval.py`

**Interfaces:**
- Consumes: `format_example` (Task 9) — same masking as training, so base-vs-tuned numbers are apples-to-apples.
- Produces: `heldout_loss(model, tokenizer, rows, device="cuda", max_len=1024) -> float` — mean per-row causal-LM loss over supervised tokens. Testable with stub model/tokenizer.

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/test_eval.py
from engine.eval import heldout_loss


class StubTokenizer:
    eos_token = "<eos>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 100 for c in text]}


class StubLoss:
    def __init__(self, v):
        self.v = v

    def item(self):
        return self.v


class StubOut:
    def __init__(self, v):
        self.loss = StubLoss(v)


class StubModel:
    def __init__(self, losses):
        self.losses = list(losses)

    def eval(self):
        return self

    def __call__(self, input_ids=None, labels=None):
        return StubOut(self.losses.pop(0))


ROWS = [
    {"instruction": "Add two numbers together.", "code": "def add(a, b):\n    return a + b"},
    {"instruction": "Subtract two numbers now.", "code": "def sub(a, b):\n    return a - b",},
]


def test_heldout_loss_is_mean_over_rows():
    model = StubModel([2.0, 4.0])
    assert heldout_loss(model, StubTokenizer(), ROWS, device="cpu") == 3.0


def test_empty_rows_returns_zero():
    assert heldout_loss(StubModel([]), StubTokenizer(), [], device="cpu") == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `& "modelEnvGpu\Scripts\python.exe" -m pytest tests\engine\test_eval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'engine.eval'`.

- [ ] **Step 3: Implement**

```python
# engine/eval.py
import torch

from engine.tune import format_example


def heldout_loss(model, tokenizer, rows, device="cuda", max_len=1024):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for row in rows:
            ex = format_example(row, tokenizer, max_len)
            out = model(
                input_ids=torch.tensor([ex["input_ids"]], device=device),
                labels=torch.tensor([ex["labels"]], device=device),
            )
            total += out.loss.item()
    return total / len(rows) if rows else 0.0
```

- [ ] **Step 4: Run to verify pass**

Run: `& "modelEnvGpu\Scripts\python.exe" -m pytest tests\engine\test_eval.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```powershell
git add engine\eval.py tests\engine\test_eval.py; git commit -m "Eval: held-out style loss with training-identical masking"
```

---

### Task 12: Eval — regression benchmark (build + sandboxed pass@1)

**Files:**
- Modify: `engine/eval.py` (append; add `import json, subprocess, sys, tempfile` and `from pathlib import Path` at top)
- Test: `tests/engine/test_eval.py` (append)

**Interfaces:**
- Produces: `extract_code(text: str) -> str` (strips one Markdown fence if present); `run_problem(candidate: str, tests: str, timeout=10) -> bool` (executes `candidate + tests` with `python -I` in a temp dir; timeout/exception → `False`); `pass_at_1(generate: Callable[[str], str], problems: list[dict]) -> float` where a problem is `{"task_id", "prompt", "tests"}`; `build_benchmark(out_path, n=100)` — downloads MBPP (sanitized, test split; free HF dataset — a one-time download, not an API call) and writes the **first `n` rows** to `data/benchmarks/mbpp100.jsonl` so the subset is fixed forever; `BENCH_PROMPT` template. Note: `python -I` is isolation-lite, not a security sandbox — acceptable because candidate code comes from our own local model and MBPP's public tests.

- [ ] **Step 1: Write the failing tests** (append)

```python
from engine.eval import extract_code, pass_at_1, run_problem


def test_extract_code_strips_fence():
    fenced = "```python\ndef f():\n    return 1\n```"
    assert extract_code(fenced) == "def f():\n    return 1"
    assert extract_code("def g():\n    return 2") == "def g():\n    return 2"


def test_run_problem_pass_fail_and_timeout():
    good = "def add(a, b):\n    return a + b"
    tests = "assert add(1, 2) == 3"
    assert run_problem(good, tests)
    assert not run_problem("def add(a, b):\n    return a - b", tests)
    assert not run_problem("while True:\n    pass", "assert True", timeout=2)


def test_pass_at_1_scores_fraction():
    problems = [
        {"task_id": 1, "prompt": "add", "tests": "assert add(1, 2) == 3"},
        {"task_id": 2, "prompt": "sub", "tests": "assert sub(3, 1) == 2"},
    ]
    gen = lambda prompt: ("def add(a, b):\n    return a + b" if "add" in prompt
                          else "def sub(a, b):\n    return a + b")
    assert pass_at_1(gen, problems) == 0.5
```

- [ ] **Step 2: Run to verify failure**

Run: `& "modelEnvGpu\Scripts\python.exe" -m pytest tests\engine\test_eval.py -v`
Expected: FAIL — `ImportError: cannot import name 'extract_code'`.

- [ ] **Step 3: Implement** (append to `engine/eval.py`)

```python
BENCH_PROMPT = """Write a Python function for this task. Reply with only the code, no explanation.

{prompt}

The function must pass these tests:
{tests}"""


def extract_code(text):
    if "```" in text:
        block = text.split("```")[1]
        for lang in ("python", "py"):
            if block.startswith(lang):
                block = block[len(lang):]
        return block.strip()
    return text.strip()


def run_problem(candidate, tests, timeout=10):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "prog.py"
        path.write_text(candidate + "\n\n" + tests + "\n", encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(path)],
                capture_output=True, timeout=timeout,
            )
            return proc.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False


def pass_at_1(generate, problems):
    passed = 0
    for p in problems:
        candidate = extract_code(generate(BENCH_PROMPT.format(**{
            "prompt": p["prompt"], "tests": p["tests"]})))
        passed += run_problem(candidate, p["tests"])
    return passed / len(problems) if problems else 0.0


def build_benchmark(out_path, n=100):
    from datasets import load_dataset

    ds = load_dataset("mbpp", "sanitized", split="test")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in list(ds)[:n]:
            f.write(json.dumps({
                "task_id": row["task_id"],
                "prompt": row["prompt"],
                "tests": "\n".join(row["test_imports"] + row["test_list"]),
            }) + "\n")
    print(f"wrote {n} problems -> {out}")
```

- [ ] **Step 4: Run to verify pass**

Run: `& "modelEnvGpu\Scripts\python.exe" -m pytest tests\engine\test_eval.py -v`
Expected: 5 passed.

- [ ] **Step 5: Owner builds the fixed benchmark** (one-time MBPP download)

```powershell
& "modelEnvGpu\Scripts\python.exe" -c "from engine.eval import build_benchmark; build_benchmark('data/benchmarks/mbpp100.jsonl')"
```

Expected: `wrote 100 problems -> data\benchmarks\mbpp100.jsonl`.

- [ ] **Step 6: Commit**

```powershell
git add engine\eval.py tests\engine\test_eval.py; git commit -m "Eval: MBPP-subset regression benchmark with sandboxed pass@1"
```

---

### Task 13: Eval — style conformance + eval CLI

**Files:**
- Modify: `engine/eval.py` (append; add `import argparse` at top)
- Create: `configs/style_prompts.jsonl`
- Test: `tests/engine/test_eval.py` (append)

**Interfaces:**
- Consumes: everything above; `models/adapters/<tag>/` (Task 10); `data/benchmarks/mbpp100.jsonl` (Task 12).
- Produces: `style_score(py_texts: list[str]) -> float` — fraction of generated Python samples with zero `ruff check` diagnostics (ruff resolves the repo's config; refine rules later to encode more of the owner's style); CLI `python -m engine.eval --config <tune-yaml> --adapter <path|none> [--skip-bench]` printing one JSON line: `{"heldout_loss": ..., "pass_at_1": ..., "style_score": ...}`. `--adapter none` evaluates the raw base model (the baseline row). Generation for bench/style: greedy, `max_new_tokens=300`, chat template.

- [ ] **Step 1: Write the failing test** (append)

```python
from engine.eval import style_score


def test_style_score_fraction_of_clean_files():
    clean = "def add(a, b):\n    return a + b\n"
    dirty = "import os\ndef f( ):\n    x=1\n    return   x\n"   # unused import etc.
    score = style_score([clean, dirty])
    assert score == 0.5
```

- [ ] **Step 2: Run to verify failure**

Run: `& "modelEnvGpu\Scripts\python.exe" -m pytest tests\engine\test_eval.py -v`
Expected: FAIL — `ImportError: cannot import name 'style_score'`.

- [ ] **Step 3: Implement** (append to `engine/eval.py`)

```python
def style_score(py_texts):
    if not py_texts:
        return 0.0
    ok = 0
    with tempfile.TemporaryDirectory() as td:
        for i, text in enumerate(py_texts):
            (Path(td) / f"gen_{i}.py").write_text(text, encoding="utf-8")
        for i in range(len(py_texts)):
            proc = subprocess.run(
                [sys.executable, "-m", "ruff", "check", str(Path(td) / f"gen_{i}.py")],
                capture_output=True,
            )
            ok += proc.returncode == 0
    return ok / len(py_texts)


def main(argv=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from engine.config import load_config
    from engine.tune import PROMPT_TEMPLATE, load_rows

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", required=True, help="models/adapters/<tag> or 'none'")
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, required=("base_model", "data_dir"))
    device = cfg.get("device", "cuda")

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16
    ).to(device)
    if args.adapter != "none":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    def generate(prompt):
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        ids = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=300, do_sample=False)
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

    test_rows = load_rows(Path(cfg["data_dir"]) / "test.jsonl")
    metrics = {"heldout_loss": round(
        heldout_loss(model, tok, test_rows, device, cfg.get("max_len", 1024)), 4)}

    if not args.skip_bench:
        problems = load_rows("data/benchmarks/mbpp100.jsonl")
        metrics["pass_at_1"] = round(pass_at_1(generate, problems), 3)

    prompts = load_rows("configs/style_prompts.jsonl")
    samples = [extract_code(generate(PROMPT_TEMPLATE.format(**p))) for p in prompts]
    metrics["style_score"] = round(style_score(samples), 3)

    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `& "modelEnvGpu\Scripts\python.exe" -m pytest tests\engine\test_eval.py -v`
Expected: 6 passed.

- [ ] **Step 5: Create `configs/style_prompts.jsonl`** (fixed style-probe set; owner may extend, never edit existing rows once a baseline is recorded)

```json
{"instruction": "Write a Python function that reads a CSV file and returns the rows as a list of dictionaries."}
{"instruction": "Write a Python function that retries another function up to three times with a short delay between attempts."}
{"instruction": "Write a Python class that caches the results of an expensive lookup with a maximum size."}
{"instruction": "Write a Python function that walks a directory tree and returns all files matching an extension."}
{"instruction": "Write a Python function that parses an ISO-8601 date string and returns how many days ago it was."}
{"instruction": "Write a Python function that chunks a list into batches of a given size."}
{"instruction": "Write a Python function that validates a config dictionary against a list of required keys."}
{"instruction": "Write a Python function that downloads a file from a URL to a target path with a timeout."}
{"instruction": "Write a Python function that merges two sorted lists into one sorted list."}
{"instruction": "Write a Python function that formats a number of seconds as a human-readable duration string."}
```

- [ ] **Step 6: Commit**

```powershell
git add engine\eval.py tests\engine\test_eval.py configs\style_prompts.jsonl; git commit -m "Eval: ruff-based style conformance and full eval CLI"
```

---

### Task 14: Report — runs.csv appender + comparison

**Files:**
- Create: `engine/report.py`, `experiments/04_style_tune/README.md`
- Test: `tests/engine/test_report.py`

**Interfaces:**
- Produces: `FIELDNAMES = ["tag", "date", "base_model", "adapter", "train_pairs", "heldout_loss", "pass_at_1", "style_score", "wall_clock_s", "tok_s", "vram_gb", "notes"]`; `append_run(csv_path, row: dict)` (creates file + header if absent; unknown keys rejected with `ValueError`; missing keys written empty); `compare(csv_path, tag_a, tag_b) -> str` (aligned two-column diff of the two rows; raises `KeyError` for a missing tag); CLI `python -m engine.report --csv experiments/04_style_tune/runs.csv --compare base coder3b-lora-r16`.

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/test_report.py
import csv

import pytest

from engine.report import FIELDNAMES, append_run, compare


def test_append_creates_header_and_row(tmp_path):
    p = tmp_path / "runs.csv"
    append_run(p, {"tag": "base", "heldout_loss": 1.5, "pass_at_1": 0.4})
    append_run(p, {"tag": "tuned", "heldout_loss": 1.2, "pass_at_1": 0.39})
    rows = list(csv.DictReader(open(p)))
    assert [r["tag"] for r in rows] == ["base", "tuned"]
    assert rows[0]["heldout_loss"] == "1.5"
    assert rows[0]["notes"] == ""


def test_append_rejects_unknown_keys(tmp_path):
    with pytest.raises(ValueError):
        append_run(tmp_path / "runs.csv", {"tag": "x", "bogus": 1})


def test_compare_shows_both_tags(tmp_path):
    p = tmp_path / "runs.csv"
    append_run(p, {"tag": "base", "heldout_loss": 1.5})
    append_run(p, {"tag": "tuned", "heldout_loss": 1.2})
    text = compare(p, "base", "tuned")
    assert "base" in text and "tuned" in text and "heldout_loss" in text
    with pytest.raises(KeyError):
        compare(p, "base", "nope")
```

- [ ] **Step 2: Run to verify failure**

Run: `& "modelEnvGpu\Scripts\python.exe" -m pytest tests\engine\test_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'engine.report'`.

- [ ] **Step 3: Implement**

```python
# engine/report.py
import argparse
import csv
from pathlib import Path

FIELDNAMES = ["tag", "date", "base_model", "adapter", "train_pairs",
              "heldout_loss", "pass_at_1", "style_score",
              "wall_clock_s", "tok_s", "vram_gb", "notes"]


def append_run(csv_path, row):
    unknown = set(row) - set(FIELDNAMES)
    if unknown:
        raise ValueError(f"unknown fields: {sorted(unknown)}")
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDNAMES})


def compare(csv_path, tag_a, tag_b):
    rows = {r["tag"]: r for r in csv.DictReader(open(csv_path, encoding="utf-8"))}
    a, b = rows[tag_a], rows[tag_b]
    lines = [f"{'field':<14}{tag_a:>16}{tag_b:>16}"]
    for k in FIELDNAMES:
        if k != "notes" and (a.get(k) or b.get(k)):
            lines.append(f"{k:<14}{a.get(k, ''):>16}{b.get(k, ''):>16}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--compare", nargs=2, metavar=("TAG_A", "TAG_B"))
    args = ap.parse_args(argv)
    if args.compare:
        print(compare(args.csv, *args.compare))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `& "modelEnvGpu\Scripts\python.exe" -m pytest tests\engine\test_report.py -v` — expected 3 passed.
Then the full suite: `& "modelEnvGpu\Scripts\python.exe" -m pytest tests -v` — expected all green.

- [ ] **Step 5: Create `experiments/04_style_tune/README.md`**

```markdown
# Experiment 04 — Style tuning via LoRA (the engine's first run)

**Hypothesis:** LoRA-tuning Qwen2.5-Coder on instruction→code pairs harvested from the
owner's repos lowers held-out style loss versus the base model without losing more than
2 absolute points of MBPP-subset pass@1.

**Method:** engine pipeline (harvest → dataset → tune → eval → report), spec at
`Docs/superpowers/specs/2026-08-06-style-tuning-engine-design.md`. Configs:
`configs/coder3b-lora.yaml` (Stage 1), `configs/coder7b-qlora.yaml` (Stage 2, WSL2).

**Win condition:** `heldout_loss` improves over the base row AND
`pass_at_1 >= base - 0.02`. All runs logged in `runs.csv`.
```

- [ ] **Step 6: Commit**

```powershell
git add engine\report.py tests\engine\test_report.py experiments\04_style_tune\README.md; git commit -m "Report: runs.csv appender, comparison CLI, experiment 04 folder"
```

---

### Task 15: Stage 1 end-to-end run (owner-run)

**Files:** none created by hand — this task exercises everything and records the first real rows.

**Interfaces:**
- Consumes: the full engine + configs; harvest and dataset outputs from Tasks 6/8.

- [ ] **Step 1: Baseline row** — evaluate the raw 3B (downloads Qwen2.5-Coder-3B-Instruct on first run; slow — pass@1 alone is 100 generations)

```powershell
& "modelEnvGpu\Scripts\python.exe" -m engine.eval --config configs\coder3b-lora.yaml --adapter none
```

Expected: one JSON line of metrics. Record it (values from the JSON, `train_pairs` from Task 8's split output, date = today):

```powershell
& "modelEnvGpu\Scripts\python.exe" -c "from engine.report import append_run; append_run('experiments/04_style_tune/runs.csv', {'tag': 'coder3b-base', 'date': 'YYYY-MM-DD', 'base_model': 'Qwen/Qwen2.5-Coder-3B-Instruct', 'adapter': 'none', 'heldout_loss': FILL, 'pass_at_1': FILL, 'style_score': FILL, 'notes': 'baseline'})"
```

- [ ] **Step 2: Train the Stage 1 adapter**

```powershell
& "modelEnvGpu\Scripts\python.exe" -m engine.tune --config configs\coder3b-lora.yaml
```

Expected: falling train loss, epoch-end val loss, final `wall_clock_s / tok_s / vram_gb` line. If OOM: halve `batch_size`, double `grad_accum`, retry.

- [ ] **Step 3: Evaluate the adapter and record the row**

```powershell
& "modelEnvGpu\Scripts\python.exe" -m engine.eval --config configs\coder3b-lora.yaml --adapter models\adapters\coder3b-lora-r16
```

Then `append_run` as in Step 1 with `tag='coder3b-lora-r16'`, the adapter path, and the training run's wall-clock/tok_s/vram numbers.

- [ ] **Step 4: Compare and judge against the win condition**

```powershell
& "modelEnvGpu\Scripts\python.exe" -m engine.report --csv experiments\04_style_tune\runs.csv --compare coder3b-base coder3b-lora-r16
```

Win (spec §5): `heldout_loss` lower than base AND `pass_at_1 >= base - 0.02`. Either way, write the result into `experiments/04_style_tune/README.md` and update `research/ROADMAP.md` (LoRA checkbox) + `HANDOFF.md`.

- [ ] **Step 5: Commit results**

```powershell
git add experiments\04_style_tune HANDOFF.md research\ROADMAP.md; git commit -m "Experiment 04: Stage 1 LoRA style-tune results"
```

---

### Stage 2 track — WSL2 + 7B QLoRA (owner-run infra; parallel, never gates Stage 1)

- [ ] **Step 1: Install WSL2 + Ubuntu** (owner runs in an elevated PowerShell, then reboots)

```powershell
wsl --install -d Ubuntu-24.04
```

- [ ] **Step 2: Inside Ubuntu — ROCm-for-WSL + venv.** Follow AMD's current "ROCm on Radeon + WSL" guide for the RX 9060 XT (the hardware notes' Option B; verify RDNA4/gfx1200 is listed as supported for WSL in the current matrix before installing). Then create `~/modelEnvWsl` (python3.12 venv) and install: ROCm torch per the guide, plus `transformers peft datasets accelerate pyyaml bitsandbytes` (ROCm build — if `pip install bitsandbytes` yields no ROCm support, consult the bitsandbytes multi-backend/ROCm install docs; **this is the known-risk step** — if it fails, fall back per spec §7).

- [ ] **Step 3: Verify the risk gate**

```bash
python -c "import torch, bitsandbytes; print(torch.cuda.is_available(), bitsandbytes.__version__)"
```

Expected: `True <version>`. If this fails after honest effort → fallback: 7B bf16 LoRA with `grad_checkpointing: true` on native Windows, or stay on 3B.

- [ ] **Step 4: Create `configs/coder7b-qlora.yaml`** (repo is shared into WSL via `/mnt/c/...`; only two kinds of changes vs Stage 1 — model/tag and quantization)

```yaml
tag: coder7b-qlora-r16
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
data_dir: data/style_corpus
adapter: qlora            # tune.py Stage 2 edit: when adapter == 'qlora', load base with
                          # BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=bfloat16,
                          # bnb_4bit_quant_type='nf4') and call prepare_model_for_kbit_training
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
learning_rate: 2.0e-4
epochs: 2
batch_size: 1
grad_accum: 16
max_len: 1024
grad_checkpointing: true
device: cuda
```

The `tune.py` QLoRA branch (~10 lines: the `BitsAndBytesConfig` + `prepare_model_for_kbit_training` import and conditional) is written **in a code-by-hand session at Stage 2 start**, against whatever bitsandbytes version actually installed — designing it now against an unverified API would bake in guesses.

- [ ] **Step 5: Run Stage 2** = Task 15's steps with the 7B config (baseline eval → train → eval → compare → record), run from the WSL venv:

```bash
python -m engine.eval --config configs/coder7b-qlora.yaml --adapter none
python -m engine.tune --config configs/coder7b-qlora.yaml
python -m engine.eval --config configs/coder7b-qlora.yaml --adapter models/adapters/coder7b-qlora-r16
python -m engine.report --csv experiments/04_style_tune/runs.csv --compare coder7b-base coder7b-qlora-r16
```

---

## Deferred (explicitly out of this plan, per spec)

- OSS repo augmentation (harvest MIT/Apache OSS clones with `source: "oss"`, mechanically restyle outputs with the owner's ruff/prettier configs before pairing — the machinery already supports the rest via `backtranslate(units, gen, source="oss")` and `source_weights`; do it if Task 6's corpus stats come back thin).
- Stage 3 reasoning models; inference/serving; test-time compute scaffolds.
