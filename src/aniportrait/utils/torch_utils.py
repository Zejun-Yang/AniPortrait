from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["get_mask_from_lengths", "linear_interpolation"]


def get_mask_from_lengths(lengths, max_len: Optional[int] = None) -> Tensor:
    import torch

    lengths = lengths.to(torch.long)
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = (
        torch.arange(0, max_len)
        .unsqueeze(0)
        .expand(lengths.shape[0], -1)
        .to(lengths.device)
    )
    mask = ids < lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def linear_interpolation(features: Tensor, seq_len: int) -> Tensor:
    import torch.nn.functional as F

    features = features.transpose(1, 2)
    output_features = F.interpolate(
        features, size=seq_len, align_corners=True, mode="linear"
    )
    return output_features.transpose(1, 2)
