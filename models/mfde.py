import torch
import torch.nn as nn
from .backbone.mixerMLP import MixerBlock

class MFDE(nn.Module):
    """
    Multi-level Feature Discrepancy Extractor (MFDE)
    Outputs intermediate features for AKD
    """

    def __init__(
        self,
        num_tokens=196,  
        embed_dim=256,
        depth=18,
        token_mlp_dim=512,
        channel_mlp_dim=2048,
        verbose=False
    ):
        """
        Args:
            num_tokens (int): number of spatial tokens after flattening the 4 scales as well so 4x7x7=196
            embed_dim (int): feature dimension
            depth (int): number of mixer blocks
            verbose (bool): print debug info
        """
        super().__init__()

        self.depth = depth
        self.verbose = verbose

        self.mixer_blocks = nn.ModuleList([
            MixerBlock(
                num_tokens=num_tokens,
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
            x: Tensor of shape either
               - [B*N, C, H, W] (spatial feature map), or
               - [B*N, N_tokens, C] (flattened tokens from MFR)

        Returns:
            final_feature: [B*N, N_tokens, C]
            intermediates: list of intermediate features depth*[B*N, N_tokens, C]
        """
        # accept both 4D maps and 3D token sequences
        if x.dim() == 4:
            B, C, H, W = x.shape
            N = H * W
            x = x.view(B, C, N).permute(0, 2, 1)  # [B, N, C]
        elif x.dim() == 3:
            B, N, C = x.shape
        else:
            raise ValueError(f"Unexpected input shape {x.shape}; expected 3D or 4D tensor")


        intermediates = []

        for i, block in enumerate(self.mixer_blocks):
            x = block(x)
            intermediates.append(x)

            if self.verbose:
                print(f"[MFDE] Layer {i} output shape: {x.shape}")

        return x, intermediates
