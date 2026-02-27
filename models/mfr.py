import torch
import torch.nn as nn


class MultiScaleFeatureRepresentation(nn.Module):
    """
    Multi-scale Feature Representation (MFR)
    Projects backbone features from different scales into a unified embedding space.
    Includes spatial uniformity through adaptive average pooling.
    """

    def __init__(
        self,
        in_channels,
        embed_dim=256,
        target_spatial_dim=7,
        verbose=False
    ):
        """
        Args:
            in_channels (list): channels from backbone feature levels
            embed_dim (int): unified feature dimension/channels after projection
            target_spatial_dim (int): target spatial dimensions (H=W) for all features
            verbose (bool): print debug info
        """
        super().__init__()

        self.verbose = verbose
        self.embed_dim = embed_dim
        self.target_spatial_dim = target_spatial_dim

        # 1x1 conv to project each level to embed_dim
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, embed_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True)
            )
            for c in in_channels
        ])

        # Adaptive average pooling to make spatial dimensions uniform
        self.pool_layers = nn.ModuleList([
            nn.AdaptiveAvgPool2d((target_spatial_dim, target_spatial_dim))
            for _ in in_channels
        ])

        if self.verbose:
            print("[MFR] Initialized")
            print(f"[MFR] In channels: {in_channels}")
            print(f"[MFR] Embed dim: {embed_dim}")
            print(f"[MFR] Target spatial dim: {target_spatial_dim}")

    def forward(self, features):
        """
        Args:
            features (list): list of feature maps from backbone stages, each with shape [B*N, C_i, H_i, W_i]

        Returns:
            unified_features (list): list of projected and spatially-pooled feature maps
                                    all with shape [B, tokens, embed_dim] where tokens = target_spatial_dim^2
        """
        assert len(features) == len(self.proj_layers)

        unified_features = []

        for i, (f, proj, pool) in enumerate(zip(features, self.proj_layers, self.pool_layers)):
            uf = proj(f)
            uf = pool(uf)
            
            # Flatten spatial dimensions into tokens: [B, C, H, W] -> [B, H*W, C]
            B, C, H, W = uf.shape
            uf = uf.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
            uf = uf.view(B, H * W, C)                  # [B, H*W, C]
            
            unified_features.append(uf)

            if self.verbose:
                print(f"[MFR] Level {i} output shape: {uf.shape}")

        return unified_features
