import torch
import torch.nn as nn
from .backbone.mixerMLP import MixerBlock

class CFI(nn.Module):
    """
    Cross-Scale Feature Integrator (CFI)
    """

    def __init__(
        self,
        num_tokens=196,
        embed_dim=256,
        depth=9,
        mlp_dim=512,
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

        self.verbose = verbose

        self.blocks = nn.ModuleList([
            MixerBlock(
                num_tokens=num_tokens,
                embed_dim=embed_dim,
                token_mlp_dim=mlp_dim,
                channel_mlp_dim=channel_mlp_dim
            )
            for _ in range(depth)
        ])

        if self.verbose:
            print(f"[CFI] Initialized with depth={depth}")

    def forward(self, x):
        """
        Args:
            x : Tensor of shape [B*N, N_tokens, C] where N_tokens= H*W*Scales

        Returns:
            integrated_feature: [B*N, N_tokens,C]
        """

        if self.verbose:
            print(f"[CFI] Input shape: {x.shape}")

        for i, block in enumerate(self.blocks):
            x = block(x)
        
        if self.verbose:
            print(f"[CFI] Block {i} output shape: {x.shape}")

        return x
