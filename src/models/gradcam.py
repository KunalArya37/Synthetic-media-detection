"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Engine
Generates visual explainability heatmaps for model predictions
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional


class GradCAM:
    """
    Grad-CAM & Grad-CAM++ Visual Explainability Engine for CNN & Hybrid Vision Models.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self.hook_handles.extend([h1, h2])

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Computes the Grad-CAM heatmap for the given input tensor.
        Args:
            input_tensor: Tensor of shape (1, C, H, W)
            target_class: Target class index (0 for Real, 1 for Synthetic). If None, uses argmax.
        Returns:
            heatmap: 2D numpy array [0.0 - 1.0] of shape (H, W)
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)
        if target_class is None:
            target_class = torch.argmax(logits, dim=-1).item()

        # Backward pass for target class score
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture gradients or activations.")

        # Compute channel weights: Global Average Pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # Shape: (1, C, 1, 1)

        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # Shape: (1, 1, h, w)

        # Apply ReLU to capture features with positive influence on the target class
        cam = F.relu(cam)

        # Resize CAM to match input tensor spatial resolution
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)

        # Normalize between 0 and 1
        cam_np = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np

    def remove_hooks(self):
        """Clean up hook handles."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()


def generate_cam_overlay(original_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.5, colormap: int = cv2.COLORMAP_JET) -> Tuple[np.ndarray, np.ndarray]:
    """
    Blends Grad-CAM heatmap over the original PIL image.
    Returns:
        blended_rgb: RGB image array with heatmap overlay
        colored_heatmap: Standalone colored heatmap
    """
    orig_np = np.array(original_image.convert("RGB"))
    orig_h, orig_w = orig_np.shape[:2]

    # Resize heatmap to original image size
    resized_heatmap = cv2.resize(heatmap, (orig_w, orig_h))
    uint8_heatmap = np.uint8(255 * resized_heatmap)

    # Apply colormap
    colored_heatmap = cv2.applyColorMap(uint8_heatmap, colormap)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    # Blend original image and heatmap
    blended = np.float32(colored_heatmap) * alpha + np.float32(orig_np) * (1.0 - alpha)
    blended_rgb = np.uint8(np.clip(blended, 0, 255))

    return blended_rgb, colored_heatmap
