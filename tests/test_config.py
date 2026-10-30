"""Unit tests for configuration models and loader in src/core/config.py."""

import os
from pathlib import Path
import pytest
from pydantic import ValidationError

from src.core.config import (
    AppConfig,
    DecisionConfig,
    VisionConfig,
    load_config,
    save_config,
)


class TestConfigLoading:
    def test_load_default_yaml(self, repo_root):
        cfg_file = repo_root / "configs" / "default_config.yaml"
        config = load_config(cfg_file)

        assert isinstance(config, AppConfig)
        assert config.vision.target_board_size == 640
        assert config.temporal.settling_time_sec == 0.8
        assert config.decision.weight_visual == 0.50
        assert config.decision.weight_temporal == 0.25
        assert config.decision.weight_geometry == 0.15
        assert config.decision.weight_occlusion == 0.10
        assert config.camera.source_type == "synthetic"

    def test_overrides_applied_correctly(self, repo_root):
        cfg_file = repo_root / "configs" / "default_config.yaml"
        overrides = {
            "vision": {"target_board_size": 800},
            "temporal": {"settling_time_sec": 1.2},
        }
        config = load_config(cfg_file, overrides=overrides)
        assert config.vision.target_board_size == 800
        assert config.temporal.settling_time_sec == 1.2
        # Unchanged fields remain defaults
        assert config.decision.weight_visual == 0.50

    def test_environment_variable_override(self, monkeypatch):
        monkeypatch.setenv("STOCKFISH_PATH", "/custom/bin/stockfish")
        config = load_config()
        assert config.analysis.stockfish_path == "/custom/bin/stockfish"

    def test_nonexistent_config_file_raises_error(self, tmp_path):
        bad_path = tmp_path / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError):
            load_config(bad_path)


class TestConfigValidation:
    def test_decision_weights_sum_valid(self):
        # 0.40 + 0.30 + 0.20 + 0.10 = 1.00
        cfg = DecisionConfig(
            weight_visual=0.40,
            weight_temporal=0.30,
            weight_geometry=0.20,
            weight_occlusion=0.10,
        )
        assert cfg.weight_visual == 0.40

    def test_decision_weights_sum_invalid_raises_error(self):
        # 0.50 + 0.50 + 0.50 + 0.10 = 1.60 != 1.00
        with pytest.raises(ValidationError) as exc_info:
            DecisionConfig(
                weight_visual=0.50,
                weight_temporal=0.50,
                weight_geometry=0.50,
                weight_occlusion=0.10,
            )
        assert "Sum of decision scoring weights" in str(exc_info.value)

    def test_vision_invalid_angle(self):
        with pytest.raises(ValidationError):
            VisionConfig(min_camera_angle=20.0)  # Below min 30.0

    def test_save_and_reload(self, tmp_path):
        config = AppConfig()
        out_file = tmp_path / "test_out.yaml"
        save_config(config, out_file)
        assert out_file.is_file()

        reloaded = load_config(out_file)
        assert reloaded.vision.target_board_size == config.vision.target_board_size
