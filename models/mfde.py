import torch
import torch.nn as nn
from .backbone.mixerMLP import MixerBlock


class MFDE(nn.Module):
    """
    Multi-Level Feature Discrepancy Extractor (MFDE)
    Processes each of the 4 scale levels INDEPENDENTLY through
    a stack of MLP-Mixer blocks. 

    Each level: (B*N, 49, 256) → MixerBlocks → (B*N, 49, 256)

    Intermediates are stored per-level, giving shape:
        [num_levels][depth] → each (B*N, 49, 256)
    """

    def __init__(
        self,
        num_tokens_per_level=49,   # 7×7 per level
        num_levels=4,
        embed_dim=256,
        depth=18,
        token_mlp_dim=128,
        channel_mlp_dim=512,
        verbose=False,
    ):
        super().__init__()

        self.num_levels = num_levels
        self.num_tokens_per_level = num_tokens_per_level
        self.embed_dim = embed_dim
        self.depth = depth
        self.verbose = verbose

        # Single shared stack — called once per level in forward()
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(
                num_tokens=num_tokens_per_level,
                embed_dim=embed_dim,
                token_mlp_dim=token_mlp_dim,
                channel_mlp_dim=channel_mlp_dim,
            )
            for _ in range(depth)
        ])

        if self.verbose:
            print(
                f"[MFDE] Initialized | num_levels={num_levels} | "
                f"depth={depth} | tokens_per_level={num_tokens_per_level} | "
                f"embed_dim={embed_dim}"
            )

    def forward(self, diff_feats):
        """
        Args:
            diff_feats: list of 4 tensors, each (B*N, 49, 256)
                        — one per scale level, from MFR subtraction

        Returns:
            final_feats:   list of 4 tensors, each (B*N, 49, 256)
                           — fed into CAF after concatenation
            intermediates: list of 4 lists, each of length `depth`
                           intermediates[level][layer] → (B*N, 49, 256)
                           — used for per-level AKD loss (Eq. 11)
        """
        assert len(diff_feats) == self.num_levels, (
            f"[MFDE] Expected {self.num_levels} levels, got {len(diff_feats)}"
        )

        final_feats   = []
        intermediates = []   # [num_levels][depth]

        for level_idx, x in enumerate(diff_feats):
            # Validate shape
            if x.dim() != 3:
                raise ValueError(
                    f"[MFDE] Level {level_idx}: expected 3D tensor "
                    f"(B*N, {self.num_tokens_per_level}, {self.embed_dim}), "
                    f"got shape {tuple(x.shape)}"
                )
            _, N, C = x.shape
            if N != self.num_tokens_per_level or C != self.embed_dim:
                raise ValueError(
                    f"[MFDE] Level {level_idx}: shape mismatch. "
                    f"Got (*, {N}, {C}), "
                    f"expected (*, {self.num_tokens_per_level}, {self.embed_dim})"
                )

            level_intermediates = []

            for layer_idx, block in enumerate(self.mixer_blocks):
                x = block(x)                        # (B*N, 49, 256)
                level_intermediates.append(x)

                if self.verbose:
                    print(
                        f"[MFDE] Level {level_idx} | "
                        f"Layer {layer_idx:>2d}/{self.depth} | "
                        f"shape: {tuple(x.shape)}"
                    )

            final_feats.append(x)                   # (B*N, 49, 256)
            intermediates.append(level_intermediates)

        return final_feats, intermediates
        # final_feats:   4 × (B*N, 49, 256)
        # intermediates: 4 × depth × (B*N, 49, 256)