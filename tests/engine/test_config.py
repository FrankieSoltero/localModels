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