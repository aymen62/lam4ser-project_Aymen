import torch
import torch.nn as nn

from models.fusion.cross_attention import CrossAttentionAdapter


class AudioLLMFusionBlock(nn.Module):
    """
    Full Person 3 fusion block.

    Input:
        text_hidden:  [B, T_text, d_text]
        audio_hidden: [B, T_audio, d_audio]

    Output:
        fused_hidden: [B, T_text, d_text]
        attn_weights: [B, T_text, T_audio]
    """

    def __init__(
        self,
        text_dim=768,
        audio_dim=768,
        num_heads=8,
        adapter_dim=256,
        dropout=0.1,
    ):
        super().__init__()

        self.audio_projection = nn.Identity()
        if audio_dim != text_dim:
            self.audio_projection = nn.Linear(audio_dim, text_dim)

        self.fusion = CrossAttentionAdapter(
            hidden_dim=text_dim,
            num_heads=num_heads,
            adapter_dim=adapter_dim,
            dropout=dropout,
        )

    def forward(self, text_hidden, audio_hidden, audio_mask=None):
        audio_hidden = self.audio_projection(audio_hidden)

        fused_hidden, attn_weights = self.fusion(
            text_hidden=text_hidden,
            audio_hidden=audio_hidden,
            audio_mask=audio_mask,
        )

        return fused_hidden, attn_weights


if __name__ == "__main__":
    B = 2
    T_text = 32
    T_audio = 50
    text_dim = 768
    audio_dim = 1024

    text_hidden = torch.randn(B, T_text, text_dim)
    audio_hidden = torch.randn(B, T_audio, audio_dim)

    block = AudioLLMFusionBlock(
        text_dim=text_dim,
        audio_dim=audio_dim,
        num_heads=8,
    )

    fused_hidden, attn_weights = block(text_hidden, audio_hidden)

    print("text_hidden:", text_hidden.shape)
    print("audio_hidden:", audio_hidden.shape)
    print("fused_hidden:", fused_hidden.shape)
    print("attn_weights:", attn_weights.shape)
