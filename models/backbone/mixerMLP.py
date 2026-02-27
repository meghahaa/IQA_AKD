import torch.nn as nn


class MixerBlock(nn.Module):
    """
    MLP-Mixer block operating on difference features
    """

    def __init__(self, num_tokens, embed_dim, token_mlp_dim=512, channel_mlp_dim=2048):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.token_mlp = nn.Sequential(
            nn.Linear(num_tokens, token_mlp_dim),
            nn.GELU(),
            nn.Linear(token_mlp_dim, num_tokens),
        )

        self.channel_mlp = nn.Sequential(
            nn.Linear(embed_dim, channel_mlp_dim),
            nn.GELU(),
            nn.Linear(channel_mlp_dim, embed_dim),
        )

    def forward(self, x):
        # x: [B, N, C]

        # Token mixing
        y=self.norm1(x)
        y = y.transpose(1, 2)
        y = self.token_mlp(y)
        y = y.transpose(1, 2)
        x = x + y

        # Channel mixing
        y=self.norm2(x)
        y = self.channel_mlp(y)
        x = x + y

        return x