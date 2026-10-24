"""
Deep Learning Detection Model Architectures with Transfer Learning
Supports EfficientNet-B4, ResNet-50, and ConvNeXt
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any, Tuple, Optional


class SyntheticMediaDetector(nn.Module):
    """
    Unified Deep Learning Detector for Synthetic & AI-Generated Media.
    Incorporates pre-trained backbone, custom regularization head, and uncertainty estimation.
    """
    def __init__(self, backbone_name: str = "efficientnet_b4", pretrained: bool = True, num_classes: int = 2, dropout_rate: float = 0.3):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        if "efficientnet_b4" in self.backbone_name:
            weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b4(weights=weights)
            in_features = backbone.classifier[1].in_features
            # Extract feature layers for Grad-CAM
            self.features = backbone.features
            self.avgpool = backbone.avgpool
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate, inplace=True),
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.SiLU(),
                nn.Dropout(p=dropout_rate / 2, inplace=True),
                nn.Linear(512, num_classes)
            )
            self.target_cam_layer = self.features[-1]

        elif "resnet50" in self.backbone_name:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
            in_features = backbone.fc.in_features
            self.features = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4
            )
            self.avgpool = backbone.avgpool
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate / 2),
                nn.Linear(512, num_classes)
            )
            self.target_cam_layer = self.features[-1][-1]

        elif "convnext" in self.backbone_name:
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            backbone = models.convnext_tiny(weights=weights)
            in_features = backbone.classifier[2].in_features
            self.features = backbone.features
            self.avgpool = backbone.avgpool
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(p=dropout_rate / 2),
                nn.Linear(512, num_classes)
            )
            self.target_cam_layer = self.features[-1][-1]

        else:
            # Fallback to standard ResNet-18 for ultra-lightweight setups
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            in_features = backbone.fc.in_features
            self.features = nn.Sequential(
                backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
                backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
            )
            self.avgpool = backbone.avgpool
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes)
            )
            self.target_cam_layer = self.features[-1][-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        pooled = self.avgpool(feat)
        flattened = torch.flatten(pooled, 1)
        logits = self.classifier(flattened)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns deep visual feature embeddings."""
        feat = self.features(x)
        pooled = self.avgpool(feat)
        return torch.flatten(pooled, 1)

    def predict_with_uncertainty(self, x: torch.Tensor, num_samples: int = 10) -> Tuple[float, float, float]:
        """
        Monte Carlo Dropout inference for epistemic uncertainty estimation.
        Returns:
            mean_prob_synthetic: Mean predicted probability of being AI-generated [0.0 - 1.0]
            uncertainty_std: Standard deviation of predictions (confidence margin)
            confidence_score: Calibrated certainty score (1.0 - 2 * std)
        """
        self.eval()  # Keep batchnorm and backbone in eval mode
        # Enable only dropout layers for Monte-Carlo sampling
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        probs = []
        with torch.no_grad():
            for _ in range(num_samples):
                logits = self.forward(x)
                prob = torch.softmax(logits, dim=-1)[:, 1]  # Index 1 = Synthetic
                probs.append(prob.cpu().item())
        
        self.eval()

        mean_prob = float(torch.tensor(probs).mean().item())
        uncertainty_std = float(torch.tensor(probs).std().item()) if len(probs) > 1 else 0.0
        confidence = max(0.0, min(1.0, 1.0 - (uncertainty_std * 2)))

        return mean_prob, uncertainty_std, confidence


def build_detector(backbone_name: str = "efficientnet_b4", pretrained: bool = True, checkpoint_path: Optional[str] = None, device: str = "cpu") -> SyntheticMediaDetector:
    """Build and initialize detector, loading checkpoint if available."""
    model = SyntheticMediaDetector(backbone_name=backbone_name, pretrained=pretrained)
    
    if checkpoint_path and torch.os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded trained detector weights from: {checkpoint_path}")
    
    model.to(device)
    model.eval()
    return model
