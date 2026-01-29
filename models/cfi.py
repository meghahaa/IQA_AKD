import torch
import torch.nn as nn


class ScaleMixerBlock(nn.Module):
    """
    Mixer block operating across scales
    """

    def __init__(self, num_scales, embed_dim, mlp_dim=512):
        super().__init__()

        self.scale_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(num_scales, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_scales),
        )

        self.channel_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )

    def forward(self, x):
        # x: [B, S, C]
        y = x.transpose(1, 2)           # [B, C, S]
        y = self.scale_mlp(y)
        y = y.transpose(1, 2)           # [B, S, C]
        x = x + y

        y = self.channel_mlp(x)
        x = x + y

        return x

class CFI(nn.Module):
    """
    Cross-Scale Feature Integrator (CFI)
    """

    def __init__(
        self,
        num_scales=4,
        embed_dim=256,
        depth=9,
        mlp_dim=512,
        verbose=False
    ):
        """
        Args:
            num_scales (int): number of feature scales
            embed_dim (int): feature dimension
            depth (int): number of mixer blocks
            verbose (bool): print debug info
        """
        super().__init__()

        self.verbose = verbose
        self.num_scales = num_scales

        self.blocks = nn.ModuleList([
            ScaleMixerBlock(
                num_scales=num_scales,
                embed_dim=embed_dim,
                mlp_dim=mlp_dim
            )
            for _ in range(depth)
        ])

        if self.verbose:
            print(f"[CFI] Initialized with depth={depth}")

    def forward(self, features):
        """
        Args:
            features (list): list of tensors [B*N, N_i, C]

        Returns:
            integrated_feature: [B*N, C]
        """
        # Pool spatial tokens per scale
        pooled = [f.mean(dim=1) for f in features]  # [B*N, C]
        x = torch.stack(pooled, dim=1)              # [B*N, S, C]

        if self.verbose:
            print(f"[CFI] Input shape: {x.shape}")

        for i, block in enumerate(self.blocks):
            x = block(x)
            if self.verbose:
                print(f"[CFI] Block {i} output shape: {x.shape}")

        # Final global aggregation
        integrated_feature = x.mean(dim=1)  # [B*N, C]

        return integrated_feature
