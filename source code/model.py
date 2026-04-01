"""Multi-task EfficientNet-B7 for COVID-19 detection + source identification."""

import torch
import torch.nn as nn
import timm


class MultiTaskEfficientNet(nn.Module):
    """Shared EfficientNet-B7 backbone with two classification heads.

    The COVID-19 detection head produces a single logit (binary).
    The source identification head produces ``num_sources`` logits
    and is discarded at inference.

    Args:
        pretrained: Load ImageNet-pretrained weights.
        num_sources: Number of hospital sources (default: 4).
    """

    def __init__(self, pretrained=True, num_sources=4):
        super().__init__()
        self.backbone = timm.create_model(
            'tf_efficientnet_b7', pretrained=pretrained, num_classes=0
        )
        feat_dim = self.backbone.num_features  # 2560

        self.covid_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 1),
        )
        self.source_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, num_sources),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, 8, C, H, W) — 8 slices per scan.

        Returns:
            covid_logits: (B,) — raw logits for COVID-19 detection.
            source_logits: (B, num_sources) — raw logits for source ID.
        """
        batch_size, num_slices, C, H, W = x.shape

        # Forward each slice independently through the backbone
        features = []
        for i in range(num_slices):
            features.append(self.backbone(x[:, i]))

        # Element-wise mean pooling over slices
        scan_feature = torch.stack(features, dim=1).mean(dim=1)  # (B, 2560)

        covid_logits = self.covid_head(scan_feature).squeeze(-1)
        source_logits = self.source_head(scan_feature)
        return covid_logits, source_logits
