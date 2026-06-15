import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanPoolingCompressor(nn.Module):
    """Baseline: group T frames into target_len buckets and average."""
    def __init__(self, target_len=50, hidden_dim=1024):
        super().__init__()
        self.target_len = target_len

    def forward(self, x):
        B, T, D = x.shape
        trim = (T // self.target_len) * self.target_len
        x = x[:, :trim, :]
        group_size = trim // self.target_len
        return x.reshape(B, self.target_len, group_size, D).mean(dim=2)


class MaxPoolingCompressor(nn.Module):
    """Group T frames into buckets and take the max."""
    def __init__(self, target_len=50, hidden_dim=1024):
        super().__init__()
        self.target_len = target_len

    def forward(self, x):
        B, T, D = x.shape
        trim = (T // self.target_len) * self.target_len
        x = x[:, :trim, :]
        group_size = trim // self.target_len
        return x.reshape(B, self.target_len, group_size, D).max(dim=2).values


class AttentionPoolingCompressor(nn.Module):
    """Learnable queries attend over all T frames freely."""
    def __init__(self, target_len=50, hidden_dim=1024):
        super().__init__()
        self.target_len = target_len
        self.queries = nn.Parameter(torch.randn(target_len, hidden_dim) * 0.02)
        self.scale = hidden_dim ** -0.5

    def forward(self, x):
        scores = torch.einsum('btd,kd->bkt', x, self.queries) * self.scale
        weights = F.softmax(scores, dim=-1)
        return torch.einsum('bkt,btd->bkd', weights, x)


class Conv1dCompressor(nn.Module):
    """Adaptive pool then depthwise conv refines local temporal patterns."""
    def __init__(self, target_len=50, hidden_dim=1024):
        super().__init__()
        self.target_len = target_len
        self.refine = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=3, padding=1, groups=hidden_dim, bias=False,
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.adaptive_avg_pool1d(x, self.target_len)
        x = self.refine(x)
        return x.transpose(1, 2)


class GatedPoolingCompressor(nn.Module):
    """Attention + sigmoid gate: can zero out irrelevant frames entirely."""
    def __init__(self, target_len=50, hidden_dim=1024):
        super().__init__()
        self.target_len = target_len
        self.queries = nn.Parameter(torch.randn(target_len, hidden_dim) * 0.02)
        self.gate_fc = nn.Linear(hidden_dim, 1)
        self.scale = hidden_dim ** -0.5

    def forward(self, x):
        scores = torch.einsum('btd,kd->bkt', x, self.queries) * self.scale
        weights = F.softmax(scores, dim=-1)
        gates = torch.sigmoid(self.gate_fc(x))
        x_gated = x * gates
        return torch.einsum('bkt,btd->bkd', weights, x_gated)


class MultiScalePoolingCompressor(nn.Module):
    """Pools at 3 scales and fuses — captures both fine and coarse patterns."""
    def __init__(self, target_len=50, hidden_dim=1024):
        super().__init__()
        self.target_len = target_len
        self.proj = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        s1 = F.adaptive_avg_pool1d(x, self.target_len)
        s2 = F.adaptive_avg_pool1d(x, self.target_len // 2)
        s2 = F.interpolate(s2, self.target_len)
        s3 = F.adaptive_avg_pool1d(x, self.target_len // 4)
        s3 = F.interpolate(s3, self.target_len)
        fused = torch.cat([s1, s2, s3], dim=1).transpose(1, 2)
        return self.proj(fused)


AudioCompressor = MeanPoolingCompressor

COMPRESSORS = {
    "mean":        MeanPoolingCompressor,
    "max":         MaxPoolingCompressor,
    "attention":   AttentionPoolingCompressor,
    "conv1d":      Conv1dCompressor,
    "gated":       GatedPoolingCompressor,
    "multiscale":  MultiScalePoolingCompressor,
}


def build_compressor(name: str, target_len: int = 50, hidden_dim: int = 1024) -> nn.Module:
    if name not in COMPRESSORS:
        raise ValueError(f"Unknown compressor '{name}'. Choose from: {list(COMPRESSORS)}")
    return COMPRESSORS[name](target_len=target_len, hidden_dim=hidden_dim)
