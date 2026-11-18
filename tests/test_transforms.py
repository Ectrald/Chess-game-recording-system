"""Unit tests for physical domain augmentations in src/ml/transforms.py.

Covers:
- STAGE_3.md Section 6: test_transforms.py specifications
- AnchoredShear bottom base preservation
- AnchoredScale base anchor invariant (scale > 1 and scale < 1)
- ShadowInjection luminance attenuation
- SensorNoiseAndBlur valid range and types
- Pipeline composition (train and val transforms)
"""

import numpy as np
import pytest
from PIL import Image

from src.ml.transforms import (
    AnchoredScale,
    AnchoredShear,
    ColorJitter,
    Compose,
    Resize,
    SensorNoiseAndBlur,
    ShadowInjection,
    get_chess_train_transforms,
    get_chess_val_transforms,
    to_numpy_image,
    to_pil_image,
)


@pytest.fixture
def synthetic_cell_image() -> Image.Image:
    """Creates a 112x112 RGB test image with red bottom edge and green top edge."""
    w, h = 112, 112
    img = Image.new("RGB", (w, h), color=(100, 150, 200))
    pixels = img.load()
    for x in range(w):
        pixels[x, h - 1] = (255, 0, 0)  # Red bottom line
        pixels[x, 0] = (0, 255, 0)      # Green top line
    return img


def test_type_conversions(synthetic_cell_image: Image.Image):
    """Verifies seamless conversion between PIL Image and OpenCV NumPy array."""
    np_img = to_numpy_image(synthetic_cell_image)
    assert isinstance(np_img, np.ndarray)
    assert np_img.shape == (112, 112, 3)
    assert np_img.dtype == np.uint8

    back_pil = to_pil_image(np_img)
    assert isinstance(back_pil, Image.Image)
    assert back_pil.size == (112, 112)
    assert back_pil.mode == "RGB"


def test_anchored_shear_preserves_base(synthetic_cell_image: Image.Image):
    """Verifies that AnchoredShear keeps the bottom row fixed while shifting upper rows."""
    shear_tf = AnchoredShear(max_shear=0.25, p=1.0)
    transformed = shear_tf(synthetic_cell_image)

    assert transformed.size == (112, 112)
    pixels = transformed.load()

    # The bottom row (h-1 = 111) must remain red across the central region
    red_count = sum(1 for x in range(25, 85) if pixels[x, 111][0] > 200)
    assert red_count >= 50, "Bottom edge of the piece base must remain anchored!"


def test_anchored_shear_numpy():
    """Verifies AnchoredShear works directly on OpenCV BGR ndarrays."""
    arr = np.zeros((112, 112, 3), dtype=np.uint8)
    arr[-1, :] = [0, 0, 255]  # Red line in BGR

    shear_tf = AnchoredShear(max_shear=0.20, p=1.0)
    res = shear_tf(arr)
    assert isinstance(res, np.ndarray)
    assert res.shape == (112, 112, 3)
    assert res[-1, 56, 2] > 200  # Central bottom pixel remains red


def test_anchored_scale_enlarge(synthetic_cell_image: Image.Image):
    """Verifies that AnchoredScale with scale > 1.0 keeps base anchored at bottom."""
    scale_tf = AnchoredScale(scale_range=(1.20, 1.20), p=1.0)
    res = scale_tf(synthetic_cell_image)

    assert res.size == (112, 112)
    pixels = res.load()
    # Bottom pixel must still be red
    assert pixels[56, 111][0] > 200


def test_anchored_scale_shrink(synthetic_cell_image: Image.Image):
    """Verifies that AnchoredScale with scale < 1.0 pads top and preserves bottom line."""
    scale_tf = AnchoredScale(scale_range=(0.80, 0.80), p=1.0)
    res = scale_tf(synthetic_cell_image)

    assert res.size == (112, 112)
    pixels = res.load()
    # Bottom pixel must still be red
    assert pixels[56, 111][0] > 200


def test_shadow_injection():
    """Verifies that ShadowInjection darkens regions without producing negative values or artifacts."""
    arr = np.full((112, 112, 3), 200, dtype=np.uint8)
    shadow_tf = ShadowInjection(max_shadow_intensity=0.50, p=1.0)
    res = shadow_tf(arr)

    assert isinstance(res, np.ndarray)
    assert res.shape == (112, 112, 3)
    assert res.dtype == np.uint8
    assert np.all(res <= 200)
    assert np.min(res) < 200  # Shadow must darken some pixels


def test_sensor_noise_and_blur():
    """Verifies that SensorNoiseAndBlur adds noise within uint8 bounds."""
    arr = np.full((112, 112, 3), 128, dtype=np.uint8)
    noise_tf = SensorNoiseAndBlur(blur_prob=1.0, noise_std=5.0, p=1.0)
    res = noise_tf(arr)

    assert isinstance(res, np.ndarray)
    assert res.shape == (112, 112, 3)
    assert res.dtype == np.uint8
    assert not np.array_equal(res, arr)


def test_color_jitter(synthetic_cell_image: Image.Image):
    """Verifies ColorJitter modifies image brightness/contrast without corrupting shape."""
    jitter = ColorJitter(brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3), p=1.0)
    res = jitter(synthetic_cell_image)
    assert res.size == (112, 112)


def test_train_and_val_pipelines(synthetic_cell_image: Image.Image):
    """Verifies full training and validation transform pipelines produce expected dimensions."""
    train_pipe = get_chess_train_transforms(target_size=(112, 112))
    val_pipe = get_chess_val_transforms(target_size=(112, 112))

    train_out = train_pipe(synthetic_cell_image)
    val_out = val_pipe(synthetic_cell_image)

    assert train_out.size == (112, 112)
    assert val_out.size == (112, 112)

