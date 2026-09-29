from pathlib import Path

import pytest
import yaml

import kavach_config
from kavach_config import ConfigError, get_config, load_config

REAL = kavach_config.DEFAULT_CONFIG_PATH


def write_variant(tmp_path: Path, **section_updates) -> Path:
    data = yaml.safe_load(REAL.read_text(encoding="utf-8"))
    for section, updates in section_updates.items():
        data[section].update(updates)
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def test_default_config_loads_with_attribute_access():
    cfg = load_config(env={})
    assert cfg.runtime.provider == "dml"
    assert cfg.runtime.fallback_to_cpu is True
    assert cfg.runtime.qnn.backend_path == "QnnHtp.dll"
    assert cfg["runtime"]["qnn"]["enable_htp_fp16_precision"] == "1"
    assert cfg.privacy.write_raw_to_disk is False
    assert cfg.privacy.debug_show_text is False


def test_known_tools_present():
    cfg = load_config(env={})
    tools = {t.name: t.exe_names for t in cfg.processes.known_tools}
    assert "AnyDesk.exe" in tools["AnyDesk"]
    assert "TeamViewer_Service.exe" in tools["TeamViewer"]
    assert "AA_v3.exe" in tools["Ammyy"]
    assert len(tools) == 11
    assert cfg.processes.extra_exe_names == []


def test_fusion_values():
    s = load_config(env={}).fusion.signals
    assert (s.remote_tool.weight, s.remote_tool.decay_s, s.remote_tool.cap) == (30, None, 30)
    assert (s.money_screen.weight, s.money_screen.decay_s, s.money_screen.cap) == (25, 120, 25)
    assert (s.otp_card.weight, s.otp_card.decay_s, s.otp_card.cap) == (20, 120, 20)
    assert (s.screen_tactic.weight, s.screen_tactic.decay_s, s.screen_tactic.cap) == (15, 120, 30)
    assert (s.fake_alert_label.weight, s.fake_alert_label.decay_s, s.fake_alert_label.cap) == (20, 120, 20)
    assert (s.call_tactic.weight, s.call_tactic.decay_s, s.call_tactic.cap) == (13, 180, 65)
    assert (s.combo_bonus.weight, s.combo_bonus.cap) == (20, 20)
    f = load_config(env={}).fusion
    assert (f.bands.caution, f.bands.alert) == (50, 70)
    assert (f.fade_s, f.label_threshold, f.alert_clear_below, f.alert_clear_hold_s) == (20, 0.5, 60, 10)
    assert (f.caution_cooldown_s, f.override_minutes, f.no_tactic_max) == (60, 10, 69)


def test_provider_env_override():
    cfg = load_config(env={"KAVACH_PROVIDER": " QNN "})
    assert cfg.runtime.provider == "qnn"


def test_invalid_provider_rejected():
    with pytest.raises(ConfigError, match="runtime.provider"):
        load_config(env={"KAVACH_PROVIDER": "tensorrt"})


def test_write_raw_to_disk_true_is_rejected(tmp_path):
    p = write_variant(tmp_path, privacy={"write_raw_to_disk": True})
    with pytest.raises(ConfigError, match="write_raw_to_disk"):
        load_config(p, env={})


def test_alt_config_path_via_env(tmp_path):
    p = write_variant(tmp_path, runtime={"provider": "cpu"})
    cfg = load_config(env={"KAVACH_CONFIG": str(p)})
    assert cfg.runtime.provider == "cpu"
    assert cfg.config_path == str(p.resolve())


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml", env={})


def test_missing_key_raises_attribute_error():
    with pytest.raises(AttributeError):
        load_config(env={}).runtime.does_not_exist


def test_get_config_is_cached(monkeypatch):
    monkeypatch.delenv("KAVACH_CONFIG", raising=False)
    monkeypatch.delenv("KAVACH_PROVIDER", raising=False)
    get_config.cache_clear()
    try:
        assert get_config() is get_config()
    finally:
        get_config.cache_clear()


# ---------------------------------------------------------------- extends + the Snapdragon config

def test_extends_merges_over_the_base(tmp_path):
    over = tmp_path / "over.yaml"
    over.write_text(f"extends: {REAL.as_posix()}\nruntime: {{provider: cpu}}\nfusion: {{bands: {{caution: 55}}}}\n")
    cfg = load_config(over, env={})
    assert cfg.runtime.provider == "cpu" and cfg.runtime.fallback_to_cpu is True        # rest of runtime kept
    assert (cfg.fusion.bands.caution, cfg.fusion.bands.alert) == (55, 70)
    assert "extends" not in cfg and cfg.config_path == str(over.resolve())


def test_extends_loop_is_an_error(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ConfigError, match="loop"):
        load_config(tmp_path / "a.yaml", env={})


def test_snapdragon_config_is_strict_qnn_on_compiled_models():
    cfg = load_config(REAL.parent / "config.snapdragon.yaml", env={})
    assert cfg.runtime.provider == "qnn" and cfg.runtime.fallback_to_cpu is False
    assert cfg.runtime.qnn.backend_path == "QnnHtp.dll" and cfg.ocr.backend == "native"
    assert cfg.ocr.native_det.startswith("weights/qnn/") and "{w}" in cfg.ocr.native_rec
    assert cfg.asr.model_dir == "weights/qnn/whisper_base"
    assert cfg.fusion == load_config(env={}).fusion                                   # everything else inherited
