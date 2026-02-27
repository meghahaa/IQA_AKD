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

    def forward(self, diff_feats,store_selected_only=False):
        """
        Args:
            diff_feats: list of 4 tensors, each (B*N, 49, 256)
                        — one per scale level, from MFR subtractions

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

        BN   = diff_feats[0].shape[0]
        L    = self.num_levels       # 4
        N_tok = self.num_tokens_per_level  # 49
        C    = self.embed_dim               # 256

        x = torch.stack(diff_feats, dim=1).view(BN * L, N_tok, C)

        selected_indices = set(range(1, self.depth, 2))  # {1,3,5,...,35}
    
        # intermediates[level][paired_layer_k]
        intermediates = [[] for _ in range(L)]

        for layer_idx, block in enumerate(self.mixer_blocks):
            x = block(x)
            if store_selected_only and layer_idx in selected_indices:
                # Split back to per-level and store
                x_split = x.view(BN, L, N_tok, C)
                for level_idx in range(L):
                    intermediates[level_idx].append(
                        x_split[:, level_idx].detach()     # (B*N, 49, 256)
                    )

        # Unstack levels
        x = x.view(BN, L, N_tok, C)                       # (B*N, 4, 49, 256)
        final_feats = [x[:, i] for i in range(L)]         # 4 × (B*N, 49, 256)

        return final_feats, intermediates
        # final_feats:   4 × (B*N, 49, 256)
        # intermediates: 4 × depth × (B*N, 49, 256)