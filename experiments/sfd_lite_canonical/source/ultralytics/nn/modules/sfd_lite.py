"""Lightweight SFD prototype modules.

These classes intentionally use names that do not replace the existing
SA_SPD, SF_DCA, or other production modules.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = (
    "Stride2ConvBaseline",
    "SPDOriginalPrototype",
    "PixelUnshuffleConv1x1",
    "PixelUnshuffleConv1x1DW",
    "SA_SPD_Lite",
    "SF_DCA_Lite",
)


class _ConvBNAct(nn.Module):
    """Convolution followed by BatchNorm and SiLU."""

    def __init__(
        self,
        c1: int,
        c2: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            c1,
            c2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, normalization, and activation."""
        return self.act(self.bn(self.conv(x)))


def _require_even_spatial(x: torch.Tensor) -> None:
    """Reject shapes that cannot be split into aligned 2x2 subpositions."""
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW input, received shape={tuple(x.shape)}")
    if x.shape[-2] % 2 or x.shape[-1] % 2:
        raise ValueError(
            "SA-SPD-Lite requires even input height and width, "
            f"received HxW={x.shape[-2]}x{x.shape[-1]}"
        )


def _split_subpositions(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return X00, X01, X10, and X11 from each non-overlapping 2x2 cell."""
    _require_even_spatial(x)
    x00 = x[..., 0::2, 0::2]
    x01 = x[..., 0::2, 1::2]
    x10 = x[..., 1::2, 0::2]
    x11 = x[..., 1::2, 1::2]
    return x00, x01, x10, x11


class Stride2ConvBaseline(nn.Module):
    """Matched stride-2 3x3 convolution baseline."""

    def __init__(self, c1: int, c2: int, *args, **kwargs) -> None:
        super().__init__()
        self.downsample = _ConvBNAct(c1, c2, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample by two with a conventional convolution."""
        return self.downsample(x)


class SPDOriginalPrototype(nn.Module):
    """Original space-to-depth followed by a dense 3x3 projection."""

    def __init__(self, c1: int, c2: int, *args, **kwargs) -> None:
        super().__init__()
        self.project = _ConvBNAct(c1 * 4, c2, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rearrange four subpositions and project them with a dense 3x3 convolution."""
        return self.project(torch.cat(_split_subpositions(x), dim=1))


class PixelUnshuffleConv1x1(nn.Module):
    """PixelUnshuffle followed immediately by a 1x1 channel projection."""

    def __init__(self, c1: int, c2: int, *args, **kwargs) -> None:
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(2)
        self.project = _ConvBNAct(c1 * 4, c2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rearrange spatial samples and project 4Cin channels to Cout."""
        _require_even_spatial(x)
        return self.project(self.unshuffle(x))


class PixelUnshuffleConv1x1DW(nn.Module):
    """PixelUnshuffle, 1x1 compression, and local 3x3 depthwise interaction."""

    def __init__(self, c1: int, c2: int, *args, **kwargs) -> None:
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(2)
        self.project = _ConvBNAct(c1 * 4, c2, kernel_size=1)
        self.local = _ConvBNAct(c2, c2, kernel_size=3, padding=1, groups=c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the lightweight space-to-depth downsampling path."""
        _require_even_spatial(x)
        return self.local(self.project(self.unshuffle(x)))


class SA_SPD_Lite(nn.Module):
    """Subposition-aware lightweight space-to-depth downsampling.

    Interface:
        input:  [B, c1, H, W], with even H and W
        output: [B, c2, H/2, W/2]
    """

    def __init__(self, c1: int, c2: int, *args, **kwargs) -> None:
        super().__init__()
        self.c1 = int(c1)
        self.c2 = int(c2)
        self.phase_gate = nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=True)
        self.project = _ConvBNAct(self.c1 * 4, self.c2, kernel_size=1)
        self.local = _ConvBNAct(self.c2, self.c2, kernel_size=3, padding=1, groups=self.c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Weight four 2x2 subpositions, concatenate, compress, and refine."""
        phases = _split_subpositions(x)
        descriptors = torch.cat([phase.mean(dim=1, keepdim=True) for phase in phases], dim=1)
        weights = torch.softmax(self.phase_gate(descriptors), dim=1)
        weighted = [phase * weights[:, index : index + 1] for index, phase in enumerate(phases)]
        return self.local(self.project(torch.cat(weighted, dim=1)))


class SF_DCA_Lite(nn.Module):
    """Context-conditioned bidirectional frequency-response modulation.

    The current neck does not expose a reusable bottleneck tensor, so this
    first prototype uses explicit 1x1 reduction and expansion. Frequency
    decomposition and context modeling operate only in the reduced space.

    Interface:
        input:  [B, c1, H, W]
        output: [B, c2, H, W]
    """

    def __init__(self, c1: int, c2: int | None = None, reduction: int = 4, *args, **kwargs) -> None:
        super().__init__()
        self.c1 = int(c1)
        self.c2 = self.c1 if c2 is None else int(c2)
        self.reduction = int(reduction)
        if self.reduction not in {4, 8}:
            raise ValueError(f"SF_DCA_Lite reduction must be 4 or 8, received {self.reduction}")

        hidden = max(1, self.c1 // self.reduction)
        self.hidden = hidden
        self.reduce = _ConvBNAct(self.c1, hidden, kernel_size=1)

        kernel = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).div_(16.0)
        self.register_buffer(
            "lowpass_kernel",
            kernel.view(1, 1, 3, 3).repeat(hidden, 1, 1, 1),
            persistent=True,
        )

        self.context = _ConvBNAct(
            hidden,
            hidden,
            kernel_size=3,
            padding=2,
            groups=hidden,
            dilation=2,
        )
        self.alpha_gate = nn.Conv2d(hidden * 3, hidden, kernel_size=1, bias=True)
        self.expand = nn.Sequential(
            nn.Conv2d(hidden, self.c2, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.c2),
        )
        self.residual = (
            nn.Identity()
            if self.c1 == self.c2
            else nn.Sequential(
                nn.Conv2d(self.c1, self.c2, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.c2),
            )
        )
        self.out_act = nn.SiLU(inplace=True)

    def _lowpass(self, z: torch.Tensor) -> torch.Tensor:
        """Apply the fixed binomial kernel independently to each channel."""
        kernel = self.lowpass_kernel.to(device=z.device, dtype=z.dtype)
        return F.conv2d(z, kernel, stride=1, padding=1, groups=self.hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decompose, conditionally modulate, expand, and add a residual."""
        z = self.reduce(x)
        low = self._lowpass(z)
        high = z - low
        spatial = self.context(z)

        reduce_dims = (-2, -1)
        descriptors = torch.cat(
            (
                high.abs().mean(dim=reduce_dims, keepdim=True),
                spatial.mean(dim=reduce_dims, keepdim=True),
                (high - spatial).abs().mean(dim=reduce_dims, keepdim=True),
            ),
            dim=1,
        )
        alpha = torch.tanh(self.alpha_gate(descriptors))
        fused = spatial + alpha * high
        return self.out_act(self.residual(x) + self.expand(fused))
