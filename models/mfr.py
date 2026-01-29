import torch
import torch.nn as nn


class MultiScaleFeatureRepresentation(nn.Module):
    """
    Multi-scale Feature Representation (MFR)
    Projects backbone features from different scales into a unified embedding space.
    """

    def __init__(
        self,
        in_channels,
        embed_dim=256,
        verbose=False
    ):
        """
        Args:
            in_channels (list): channels from backbone feature levels
            embed_dim (int): unified feature dimension
            verbose (bool): print debug info
        """
        super().__init__()

        self.verbose = verbose
        self.embed_dim = embed_dim

        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, embed_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True)
            )
            for c in in_channels
        ])

        if self.verbose:
            print("[MFR] Initialized")
            print(f"[MFR] In channels: {in_channels}")
            print(f"[MFR] Embed dim: {embed_dim}")

    def forward(self, features):
        """
        Args:
            features (list): list of feature maps from backbone

        Returns:
            unified_features (list): list of projected feature maps
        """
        assert len(features) == len(self.proj_layers)

        unified_features = []

        for i, (f, proj) in enumerate(zip(features, self.proj_layers)):
            uf = proj(f)
            unified_features.append(uf)

            if self.verbose:
                print(f"[MFR] Level {i} output shape: {uf.shape}")

        return unified_features
