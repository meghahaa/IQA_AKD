import torch
import torch.nn as nn
import torch.nn.functional as F


class CAF(nn.Module):
    """
    Cross-Attention Feature Fusion (CAF)
    """

    def __init__(
        self,
        embed_dim=256,
        dropout=0.1,
        verbose=False
    ):
        """
        Args:
            embed_dim (int): feature dimension
            dropout (float): dropout rate
            verbose (bool): print debug info
        """
        super().__init__()

        self.verbose = verbose
        self.embed_dim = embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        if self.verbose:
            print("[CAF] Initialized")

    def forward(self, diff_feat, dist_feat):
        """
        Args:
            diff_feat: Tensor [B*N, C] (difference-guided features)
            dist_feat: Tensor [B*N, C] (distorted image features)

        Returns:
            fused_feat: Tensor [B*N, C]
        """
        # Add dummy sequence dimension
        q = self.q_proj(diff_feat).unsqueeze(1)   # [B*N, 1, C]
        k = self.k_proj(diff_feat).unsqueeze(1)   # [B*N, 1, C]
        v = self.v_proj(dist_feat).unsqueeze(1)   # [B*N, 1, C]

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_scores = attn_scores / (self.embed_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_out = torch.matmul(attn_weights, v).squeeze(1)  # [B*N, C]

        # Residual + normalization
        fused = self.out_proj(attn_out)
        fused = self.dropout(fused)
        fused = self.norm(fused + dist_feat)

        if self.verbose:
            print(f"[CAF] Output shape: {fused.shape}")

        return fused
