import torch
import torch.nn as nn


class MixerBlock(nn.Module):
    """
    Single MLP-Mixer block
    """

    def __init__(self, num_tokens, embed_dim, token_mlp_dim=512, channel_mlp_dim=2048):
        super().__init__()

        self.token_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(num_tokens, token_mlp_dim),
            nn.GELU(),
            nn.Linear(token_mlp_dim, num_tokens),
        )

        self.channel_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, channel_mlp_dim),
            nn.GELU(),
            nn.Linear(channel_mlp_dim, embed_dim),
        )

    def forward(self, x):
        # x: [B, N, C]
        y = x.transpose(1, 2)
        y = self.token_mlp(y)
        y = y.transpose(1, 2)
        x = x + y

        y = self.channel_mlp(x)
        x = x + y

        return x

class MFDE(nn.Module):
    """
    Multi-level Feature Discrepancy Extractor (MFDE)
    Outputs intermediate features for AKD
    """

    def __init__(
        self,
        embed_dim=256,
        depth=18,
        token_mlp_dim=512,
        channel_mlp_dim=2048,
        verbose=False
    ):
        """
        Args:
            embed_dim (int): feature dimension
            depth (int): number of mixer blocks
            verbose (bool): print debug info
        """
        super().__init__()

        self.depth = depth
        self.verbose = verbose

        self.mixer_blocks = nn.ModuleList([
            MixerBlock(
                num_tokens=None,  # will be set dynamically
                embed_dim=embed_dim,
                token_mlp_dim=token_mlp_dim,
                channel_mlp_dim=channel_mlp_dim
            )
            for _ in range(depth)
        ])

        if self.verbose:
            print(f"[MFDE] Initialized with depth={depth}")

    def forward(self, x):
        """
        Args:
            x: Tensor [B*N, C, H, W]

        Returns:
            final_feature: [B*N, N_tokens, C]
            intermediates: list of intermediate features
        """
        B, C, H, W = x.shape
        N = H * W

        # Flatten spatial dimensions
        x = x.view(B, C, N).permute(0, 2, 1)  # [B, N, C]

        # Dynamically set token MLP input size (first forward)
        for block in self.mixer_blocks:
            if block.token_mlp[1].in_features is None:
                block.token_mlp[1] = nn.Linear(N, block.token_mlp[1].out_features)
                block.token_mlp[3] = nn.Linear(block.token_mlp[3].in_features, N)

        intermediates = []

        for i, block in enumerate(self.mixer_blocks):
            x = block(x)
            intermediates.append(x)

            if self.verbose:
                print(f"[MFDE] Layer {i} output shape: {x.shape}")

        return x, intermediates
