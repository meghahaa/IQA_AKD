import torch
import torch.nn as nn
import timm


class SwinBackbone(nn.Module):
    """
    Swin Transformer Backbone Wrapper
    Outputs multi-level feature maps for IQA
    """

    def __init__(
        self,
        model_name="swin_tiny_patch4_window7_224",
        pretrained=True,
        verbose=False,
    ):
        super().__init__()

        self.verbose = verbose

        # Load Swin backbone from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,   # IMPORTANT
            out_indices=(0, 1, 2, 3)
        )

        # Feature dimensions for Swin-T
        self.out_channels = self.backbone.feature_info.channels()

        if self.verbose:
            print("[SwinBackbone] Initialized")
            print(f"[SwinBackbone] Feature channels: {self.out_channels}")

    def forward(self, x):
        """
        Args:
            x: Tensor [B*N, 3, 224, 224]

        Returns:
            features: list of 4 tensors
        """
        features = self.backbone(x)

        if self.verbose:
            for i, f in enumerate(features):
                print(f"[SwinBackbone] Level {i} shape: {f.shape}")

        return features
