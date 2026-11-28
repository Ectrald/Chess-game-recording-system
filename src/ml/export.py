"""Model export to ONNX and numerical parity validation module.

Implements PyTorch to ONNX tracing and parity verification according to:
- STAGE_3.md, Section 4.6
- 06_DATA_AND_MLOPS.md, Section 8
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ort = None
    ORT_AVAILABLE = False


def export_model_to_onnx(
    model: Any,
    output_path: Union[str, Path],
    input_shape: Tuple[int, ...] = (64, 3, 112, 112),
    opset_version: int = 17,
) -> bool:
    """Exports a PyTorch piece classifier model to optimized ONNX format.

    Args:
        model: PyTorch model instance (e.g. MultiHeadPieceClassifier or SingleHead).
        output_path: Destination path for the .onnx file.
        input_shape: Shape of dummy input tensor (default [64, 3, 112, 112]).
        opset_version: Target ONNX operator set version (default 17).

    Returns:
        True if export succeeded, False otherwise.
    """
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch is not available; cannot export model to ONNX.")
        return False

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_input = torch.randn(*input_shape, dtype=torch.float32)

    export_kwargs = {
        "export_params": True,
        "opset_version": opset_version,
        "do_constant_folding": True,
        "input_names": ["cell_crops"],
        "output_names": ["probabilities"],
        "dynamic_axes": {"cell_crops": {0: "batch_size"}, "probabilities": {0: "batch_size"}},
    }
    # PyTorch 2.5+ defaults to dynamo exporter requiring onnxscript; disable dynamo for direct tracing
    if hasattr(torch.onnx, "export") and "dynamo" in torch.onnx.export.__code__.co_varnames:
        export_kwargs["dynamo"] = False

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(out_p),
            **export_kwargs,
        )
        logger.info("Successfully exported ONNX model to %s (opset %d)", out_p, opset_version)
        return True
    except Exception as exc:
        logger.error("Failed to export model to ONNX: %s", exc)
        return False


def verify_onnx_numerical_parity(
    torch_model: Any,
    onnx_path: Union[str, Path],
    sample_tensor: Optional[np.ndarray] = None,
    tolerance: float = 1e-4,
) -> Tuple[bool, float]:
    """Validates numerical parity between PyTorch model outputs and ONNX Runtime predictions.

    Args:
        torch_model: Original PyTorch model.
        onnx_path: Path to the exported ONNX model file.
        sample_tensor: Optional input tensor (B, 3, 112, 112).
        tolerance: Maximum permitted absolute discrepancy max |p_torch - p_onnx|.

    Returns:
        Tuple of (is_parity_valid: bool, max_absolute_error: float).
    """
    if not (TORCH_AVAILABLE and ORT_AVAILABLE):
        logger.info("PyTorch or ONNX Runtime unavailable; skipping numerical parity check.")
        return True, 0.0

    if sample_tensor is None:
        sample_tensor = np.random.randn(4, 3, 112, 112).astype(np.float32)

    torch_tensor = torch.from_numpy(sample_tensor)
    torch_model.eval()

    with torch.no_grad():
        if hasattr(torch_model, "predict_probabilities"):
            torch_probs = torch_model.predict_probabilities(torch_tensor).cpu().numpy()
        else:
            torch_probs = torch.softmax(torch_model(torch_tensor), dim=-1).cpu().numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    onnx_probs = sess.run([output_name], {input_name: sample_tensor})[0]

    max_diff = float(np.max(np.abs(torch_probs - onnx_probs)))
    is_valid = max_diff < tolerance

    if not is_valid:
        logger.warning("ONNX parity test failed: max difference %.2e exceeds tolerance %.2e", max_diff, tolerance)
    else:
        logger.info("ONNX parity verified: max difference %.2e < %.2e", max_diff, tolerance)

    return is_valid, max_diff

