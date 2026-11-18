"""Unit tests for dataset parsing, grouped splitting and label decomposition.

Covers:
- STAGE_3.md Section 6: test_dataset_split.py specifications
- Multi-head label decomposition
- Grouped split isolation (unseen sets in test)
- Class weighting and inverse frequency calculation
- Manifest loading and ChessCropDataset tensor formatting
"""

import json
from pathlib import Path
import numpy as np
import pytest

from src.core.datatypes import PIECE_CLASSES
from src.ml.dataset import (
    ChessCropDataset,
    DatasetSample,
    compute_class_weights,
    decompose_piece_class,
    grouped_split_dataset,
    load_dataset_from_manifest,
)


def test_decompose_piece_class():
    """Verifies decomposition of 13 piece symbols into multi-head targets."""
    # Empty
    idx, occ, col, typ = decompose_piece_class("empty")
    assert idx == 0
    assert occ == 0
    assert col == -1
    assert typ == -1

    # White Pawn
    idx, occ, col, typ = decompose_piece_class("wP")
    assert idx == 1
    assert occ == 1
    assert col == 0
    assert typ == 0

    # Black King
    idx, occ, col, typ = decompose_piece_class("bK")
    assert idx == 12
    assert occ == 1
    assert col == 1
    assert typ == 5

    with pytest.raises(ValueError):
        decompose_piece_class("invalid_piece")


def test_grouped_split_isolation():
    """Verifies that grouped split strictly isolates test sets from train sets."""
    samples = []
    # 4 distinct chess sets, each with 20 samples
    for set_idx in [1, 2, 3, 4]:
        for i in range(20):
            sample = DatasetSample.create(
                sample_id=f"set{set_idx:02d}_{i:03d}",
                image_path=f"/dummy/set{set_idx:02d}_{i:03d}.png",
                piece_class="empty" if i < 10 else "wP",
                group_id=f"set_{set_idx:02d}",
            )
            samples.append(sample)

    assert len(samples) == 80

    train, val, test = grouped_split_dataset(samples, val_ratio=0.25, test_ratio=0.25, seed=42)

    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
    assert len(train) + len(val) + len(test) == 80

    train_groups = {s.group_id for s in train}
    val_groups = {s.group_id for s in val}
    test_groups = {s.group_id for s in test}

    # Strict isolation: no intersection between splits
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)


def test_grouped_split_single_group_fallback():
    """Verifies graceful fallback to sample shuffling when all samples belong to one group."""
    samples = [
        DatasetSample.create(
            sample_id=f"sample_{i}",
            image_path=f"/dummy/{i}.png",
            piece_class="empty",
            group_id="single_set",
        )
        for i in range(50)
    ]

    train, val, test = grouped_split_dataset(samples, val_ratio=0.20, test_ratio=0.20, seed=42)
    assert len(train) + len(val) + len(test) == 50
    assert len(test) == 10
    assert len(val) == 10
    assert len(train) == 30


def test_compute_class_weights():
    """Verifies inverse frequency class weights balance dominant empty squares vs rare pieces."""
    samples = []
    # 500 empty squares
    for i in range(500):
        samples.append(DatasetSample.create(f"e_{i}", f"/d/{i}.png", "empty"))
    # 50 white pawns
    for i in range(50):
        samples.append(DatasetSample.create(f"p_{i}", f"/d/{i}.png", "wP"))
    # 10 white kings
    for i in range(10):
        samples.append(DatasetSample.create(f"k_{i}", f"/d/{i}.png", "wK"))

    weights = compute_class_weights(samples)
    assert len(weights) == len(PIECE_CLASSES)
    # Empty squares should have lower weight than kings
    assert weights["empty"] < weights["wP"] < weights["wK"]
    assert pytest.approx(np.mean(list(weights.values())), rel=1e-3) == 1.0


def test_load_manifest_and_crop_dataset(tmp_path: Path):
    """Verifies loading from dataset_manifest.json into ChessCropDataset with tensor normalization."""
    # Create dummy image and manifest
    img_dir = tmp_path / "train" / "wP"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_file = img_dir / "test_sample.png"

    import cv2
    dummy_img = np.full((112, 112, 3), 150, dtype=np.uint8)
    cv2.imwrite(str(img_file), dummy_img)

    manifest_data = {
        "dataset_version": "1.0.0",
        "samples": [
            {
                "sample_id": "set01_0001_e4",
                "image_rel_path": "train/wP/test_sample.png",
                "piece_class": "wP",
                "square": "e4",
                "group_id": "set01",
            }
        ]
    }
    manifest_file = tmp_path / "dataset_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    loaded_samples = load_dataset_from_manifest(manifest_file)
    assert len(loaded_samples) == 1
    sample = loaded_samples[0]
    assert sample.piece_class == "wP"
    assert sample.occ_label == 1
    assert sample.color_label == 0
    assert sample.type_label == 0

    dataset = ChessCropDataset(loaded_samples, target_size=(112, 112), normalize=True)
    assert len(dataset) == 1

    item = dataset[0]
    assert "image" in item
    assert isinstance(item["image"], np.ndarray)
    assert item["image"].shape == (3, 112, 112)
    assert item["image"].dtype == np.float32
    assert item["occ_label"] == 1
    assert item["class_index"] == 1

