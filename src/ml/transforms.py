"""Specialized physical domain augmentations for chess cell crops.

Implements anchored shear, scale, shadow injection and sensor perturbations according to:
- STAGE_3.md, Section 4.1
- 06_DATA_AND_MLOPS.md, Section 5.4
- Explanatory Note, Section 2.4.2
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps


def to_pil_image(img: Union[Image.Image, np.ndarray]) -> Image.Image:
    """Converts a PIL Image or OpenCV NumPy array to PIL Image in RGB format."""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            return Image.fromarray(img).convert("RGB")
        if img.ndim == 3:
            if img.shape[2] == 3:
                # Assume BGR if coming from OpenCV
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb)
            if img.shape[2] == 4:
                return Image.fromarray(img).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img)}")


def to_numpy_image(img: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """Converts a PIL Image or NumPy array to OpenCV BGR format (H, W, 3) uint8."""
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img.copy()
    if isinstance(img, Image.Image):
        rgb = np.array(img.convert("RGB"), dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    raise TypeError(f"Unsupported image type: {type(img)}")


class AnchoredShear:
    """Affine horizontal shear with fixed bottom edge.

    Preserves the physical boundary where the piece base contacts the chessboard cell,
    while shifting upper regions to simulate viewing angles between 50° and 85°.
    """

    def __init__(self, max_shear: float = 0.25, p: float = 0.7) -> None:
        self.max_shear = max_shear
        self.p = p

    def __call__(self, img: Union[Image.Image, np.ndarray]) -> Union[Image.Image, np.ndarray]:
        if random.random() > self.p:
            return img

        is_numpy = isinstance(img, np.ndarray)
        pil_img = to_pil_image(img)
        w, h = pil_img.size

        shear = random.uniform(-self.max_shear, self.max_shear)
        if abs(shear) < 1e-4:
            return img

        # Target (xt, yt) to source (xs, ys):
        # At yt = h (bottom edge): xs = xt
        # At yt = 0 (top edge): xs = xt + shear * h
        # Affine matrix: xs = xt - shear*yt + shear*h
        affine_matrix = (1.0, -shear, float(shear * h), 0.0, 1.0, 0.0)
        transformed = pil_img.transform(
            (w, h),
            Image.AFFINE,
            affine_matrix,
            resample=Image.BILINEAR,
            fillcolor=(128, 128, 128),
        )

        return to_numpy_image(transformed) if is_numpy else transformed


class AnchoredScale:
    """Vertical scale variation anchored at the square base.

    Varies visible height of piece heads without detaching base contact from the bottom edge.
    """

    def __init__(self, scale_range: Tuple[float, float] = (0.85, 1.25), p: float = 0.7) -> None:
        self.scale_range = scale_range
        self.p = p

    def __call__(self, img: Union[Image.Image, np.ndarray]) -> Union[Image.Image, np.ndarray]:
        if random.random() > self.p:
            return img

        is_numpy = isinstance(img, np.ndarray)
        pil_img = to_pil_image(img)
        w, h = pil_img.size

        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        if abs(scale - 1.0) < 1e-3:
            return img

        new_h = max(4, int(round(h * scale)))
        resized = pil_img.resize((w, new_h), resample=Image.BILINEAR)

        if new_h >= h:
            # Crop bottom h pixels to keep base anchored at y=h
            result = resized.crop((0, new_h - h, w, new_h))
        else:
            # Pad at top to keep base anchored at bottom
            pad_h = h - new_h
            # Pick background color from top edge of resized image
            edge_color = resized.getpixel((w // 2, 0))
            result = Image.new("RGB", (w, h), color=edge_color)
            result.paste(resized, (0, pad_h))

        return to_numpy_image(result) if is_numpy else result


class ShadowInjection:
    """Synthesizes semi-transparent directional shadows cast across cell crops."""

    def __init__(self, max_shadow_intensity: float = 0.45, p: float = 0.5) -> None:
        self.max_shadow_intensity = max_shadow_intensity
        self.p = p

    def __call__(self, img: Union[Image.Image, np.ndarray]) -> Union[Image.Image, np.ndarray]:
        if random.random() > self.p:
            return img

        is_numpy = isinstance(img, np.ndarray)
        arr = to_numpy_image(img).astype(np.float32)
        h, w = arr.shape[:2]

        # Generate directional shadow mask using blurred ellipse or gradient
        mask = np.zeros((h, w), dtype=np.float32)
        center_x = random.randint(0, w)
        center_y = random.randint(0, h)
        axis_a = random.randint(w // 3, w)
        axis_b = random.randint(h // 4, h // 2)
        angle = random.randint(0, 180)

        cv2.ellipse(mask, (center_x, center_y), (axis_a, axis_b), angle, 0, 360, 1.0, -1)
        # Soften shadow boundary
        ksize = max(5, (w // 5) | 1)
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

        alpha = random.uniform(0.15, self.max_shadow_intensity)
        # Apply multiplicative darkening: I' = I * (1 - alpha * mask)
        attenuation = 1.0 - (alpha * mask[:, :, np.newaxis])
        shadowed = np.clip(arr * attenuation, 0.0, 255.0).astype(np.uint8)

        if not is_numpy:
            return to_pil_image(shadowed)
        return shadowed


class SensorNoiseAndBlur:
    """Models camera sensor noise, mild defocus blur, and optical artifacts."""

    def __init__(self, blur_prob: float = 0.3, noise_std: float = 4.0, p: float = 0.5) -> None:
        self.blur_prob = blur_prob
        self.noise_std = noise_std
        self.p = p

    def __call__(self, img: Union[Image.Image, np.ndarray]) -> Union[Image.Image, np.ndarray]:
        if random.random() > self.p:
            return img

        is_numpy = isinstance(img, np.ndarray)
        arr = to_numpy_image(img).astype(np.float32)

        # 1. Defocus blur
        if random.random() < self.blur_prob:
            k = random.choice([3, 5])
            arr = cv2.GaussianBlur(arr, (k, k), 0)

        # 2. Additive zero-mean Gaussian sensor noise
        if self.noise_std > 0:
            std = random.uniform(1.0, self.noise_std)
            noise = np.random.normal(0.0, std, arr.shape).astype(np.float32)
            arr = np.clip(arr + noise, 0.0, 255.0)

        res = arr.astype(np.uint8)
        return res if is_numpy else to_pil_image(res)


class ColorJitter:
    """Randomly perturbs brightness, contrast, and color balance (cool/warm lighting)."""

    def __init__(
        self,
        brightness_range: Tuple[float, float] = (0.80, 1.20),
        contrast_range: Tuple[float, float] = (0.80, 1.20),
        p: float = 0.5,
    ) -> None:
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.p = p

    def __call__(self, img: Union[Image.Image, np.ndarray]) -> Union[Image.Image, np.ndarray]:
        if random.random() > self.p:
            return img

        is_numpy = isinstance(img, np.ndarray)
        pil_img = to_pil_image(img)

        # Brightness
        b_factor = random.uniform(self.brightness_range[0], self.brightness_range[1])
        pil_img = ImageEnhance.Brightness(pil_img).enhance(b_factor)

        # Contrast
        c_factor = random.uniform(self.contrast_range[0], self.contrast_range[1])
        pil_img = ImageEnhance.Contrast(pil_img).enhance(c_factor)

        return to_numpy_image(pil_img) if is_numpy else pil_img


class Resize:
    """Resizes image to fixed canonical resolution (W, H)."""

    def __init__(self, size: Tuple[int, int] = (112, 112)) -> None:
        self.size = size

    def __call__(self, img: Union[Image.Image, np.ndarray]) -> Union[Image.Image, np.ndarray]:
        if isinstance(img, Image.Image):
            if img.size == self.size:
                return img
            return img.resize(self.size, resample=Image.BILINEAR)
        if isinstance(img, np.ndarray):
            h, w = img.shape[:2]
            if (w, h) == self.size:
                return img
            return cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
        raise TypeError(f"Unsupported image type: {type(img)}")


class Compose:
    """Chains a sequence of transform operations."""

    def __init__(self, transforms: Sequence[Callable[[Any], Any]]) -> None:
        self.transforms = list(transforms)

    def __call__(self, img: Any) -> Any:
        res = img
        for t in self.transforms:
            res = t(res)
        return res


def get_chess_train_transforms(target_size: Tuple[int, int] = (112, 112)) -> Compose:
    """Returns standard training transform pipeline with physical domain preservation."""
    return Compose([
        Resize(size=target_size),
        AnchoredShear(max_shear=0.20, p=0.6),
        AnchoredScale(scale_range=(0.88, 1.18), p=0.6),
        ShadowInjection(max_shadow_intensity=0.40, p=0.5),
        ColorJitter(brightness_range=(0.85, 1.15), contrast_range=(0.85, 1.15), p=0.5),
        SensorNoiseAndBlur(blur_prob=0.25, noise_std=3.5, p=0.4),
        Resize(size=target_size),
    ])


def get_chess_val_transforms(target_size: Tuple[int, int] = (112, 112)) -> Compose:
    """Returns deterministic validation / inference preprocessing pipeline."""
    return Compose([
        Resize(size=target_size),
    ])

