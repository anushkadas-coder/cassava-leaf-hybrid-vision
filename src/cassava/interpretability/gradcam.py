"""Grad-CAM (Selvaraju et al., 2017) for the library's CNNs.

CAM_c = ReLU( sum_k alpha_k^c * A^k ),  alpha_k^c = spatial mean of d(score_c)/dA^k
where A is the activation of ``target_layer``. The map is upsampled to the input size and scaled to [0, 1].

Interpretation caveat: a plausible-looking heatmap does not prove the model reasons in a biologically correct way;
Grad-CAM is low-resolution and class-discriminative only up to the chosen layer. Use it together with the
quantitative shortcut tests in ``shortcut.py``.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        self.model = model.eval()
        self.layer = target_layer if target_layer is not None else model.gradcam_layer
        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        self._h1 = self.layer.register_forward_hook(self._fwd)

    def _fwd(self, _m, _i, out):
        # Tensor hook (not module full-backward hook): works with in-place ReLU layers.
        self._acts = out
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "_grads", g))

    def remove(self) -> None:
        self._h1.remove()

    def __call__(self, x: torch.Tensor, f: torch.Tensor | None = None, class_idx: int | None = None) -> tuple[np.ndarray, int, np.ndarray]:
        """x: (1, C, H, W). Returns (cam in [0,1] of shape (H, W), class_idx used, softmax probabilities)."""
        if x.shape[0] != 1:
            raise ValueError("GradCAM handles one image at a time")
        x = x.clone().requires_grad_(True)  # guarantees backward hooks fire even if the first layer is frozen
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x, f)
            c = int(logits.argmax(1).item()) if class_idx is None else int(class_idx)
            logits[0, c].backward()
        a, g = self._acts.detach(), self._grads
        if a is None or g is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")
        alpha = g.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((alpha * a).sum(1, keepdim=True))
        cam = nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / cam.max() if cam.max() > 1e-12 else np.zeros_like(cam)
        return cam.astype(np.float32), c, torch.softmax(logits.detach(), 1)[0].cpu().numpy()


def overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a JET heatmap onto the uint8 RGB image."""
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    return np.clip((1 - alpha) * rgb + alpha * heat, 0, 255).astype(np.uint8)


def cam_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two maps (used to compare explanations across models/representations)."""
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / d) if d > 1e-12 else 0.0
