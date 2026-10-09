# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import torch
from pydantic import Field
from torch import nn

from .base import BaseModelConfig
from .common import INVALID_POS_VALUE


class OnTheFlyPreprocessor(BaseModelConfig):
    """Module to apply common preprocessing steps on-the-fly, inside an nn.Module.

    Each window is processed alone, in the order of the ``neuralset`` neuro extractors:
    bad channels, notch, band-pass, resampling, scaler, clamp. The notch, band-pass and
    resampling act on the window's spectrum.

    Parameters
    ----------
    ptp_threshold :
        If provided, channels with a ptp higher are considered bad. If `update_ch_pos` is True,
        their positions will be set to `common.INVALID_POS_VALUE`, otherwise the amplitude
        of bad channels are replaced by 0.
    sfreq :
        Sampling rate of the input, in Hz. Required by `notch_filter`, `filter` and
        `frequency`.
    notch_filter :
        Frequency or frequencies (Hz) to remove, with their harmonics up to 300 Hz, as the
        extractors do. The spectrum within 1 Hz of each is removed.
    filter :
        Band-pass limits ``(l_freq, h_freq)`` in Hz; either can be None.
    frequency :
        Sampling rate of the output, in Hz. It changes the number of samples, so the model
        has to accept the resampled length.
    scaler :
        Scaling strategy to apply to the input. If provided as a float, the amplitude will be
        multiplied by this value. If "RobustScaler" or "StandardScaler", the input data will be
        scaled along the dimensions specified by `scale_dim`. If "QuantileAbsScaler", each
        channel is divided by the ``scaler_quantile``-th percentile of its absolute value;
        this is the robust per-channel scaling used by the BIOT model (Yang et al., NeurIPS 2023).
        If "MinMaxScaler", each channel is linearly rescaled to [-1, 1] across the time
        dimension (dim=-1); this is the per-window normalization used by BENDR
        (Kostas et al., Front. Hum. Neurosci. 2021).
    scale_dim :
        Dimensions along which to apply scaling (see `scaler`). If None, scale along all
        dimensions.
    scaler_quantile :
        Quantile for ``"QuantileAbsScaler"`` mode (default 0.95, i.e. 95th percentile).
        Ignored when ``scaler`` is not ``"QuantileAbsScaler"``.
    clamp :
        If provided, clamp input to the range [-clamp, clamp].
    update_ch_pos :
        If True, update channel positions of bad channels above `ptp_threshold`.
    """

    ptp_threshold: float | None = None
    sfreq: float | None = None
    notch_filter: float | list[float] | None = None
    filter: tuple[float | None, float | None] | None = None
    frequency: float | None = None
    scaler: (
        float
        | tp.Literal[
            "RobustScaler", "StandardScaler", "QuantileAbsScaler", "MinMaxScaler"
        ]
        | None
    ) = None
    scale_dim: int | list[int] | None = None
    scaler_quantile: float = Field(default=0.95, ge=0.0, le=1.0)
    clamp: float | None = None
    update_ch_pos: bool = True

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if self.scaler == "RobustScaler" and isinstance(self.scale_dim, list):
            raise NotImplementedError(
                "Robust scaling with multiple scale_dim is not supported."
            )
        if self.sfreq is None and any(
            step is not None for step in (self.notch_filter, self.filter, self.frequency)
        ):
            raise ValueError("notch_filter, filter and frequency need the input's sfreq.")

    def build(self) -> "OnTheFlyPreprocessorModel":
        return OnTheFlyPreprocessorModel(self)


class OnTheFlyPreprocessorModel(nn.Module):
    """``nn.Module`` implementation of :class:`OnTheFlyPreprocessor`."""

    def __init__(self, config: OnTheFlyPreprocessor) -> None:
        super().__init__()

        self.ptp_threshold = config.ptp_threshold
        self.sfreq = config.sfreq
        self.notch_filter = config.notch_filter
        self.filter = config.filter
        self.frequency = config.frequency
        self.scaler = config.scaler
        self.scale_dim = config.scale_dim
        self.scaler_quantile = config.scaler_quantile
        self.clamp = config.clamp
        self.update_ch_pos = config.update_ch_pos

    def forward(
        self,
        x: torch.Tensor,
        channel_positions: torch.Tensor | None = None,
        **kwargs: tp.Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Preprocess the input tensor in-place (clone first).

        Parameters
        ----------
        x : Tensor
            Input of shape ``(B, C, T)``.
        channel_positions : Tensor or None
            Electrode coordinates of shape ``(B, C, D)``.  Updated in-place
            when *update_ch_pos* is True and bad channels are detected.

        Returns
        -------
        tuple of (Tensor, Tensor or None)
            Preprocessed tensor and (possibly updated) channel positions.
        """
        x_out = x.clone()
        ch_pos_out = channel_positions.clone() if channel_positions is not None else None

        if self.ptp_threshold is not None:
            mask = (x_out.max(dim=-1)[0] - x_out.min(dim=-1)[0]) > self.ptp_threshold
            if self.update_ch_pos:
                if ch_pos_out is None:
                    raise ValueError(
                        "valid channel_positions must be provided if update_ch_pos is True."
                    )
                ch_pos_out[mask] = INVALID_POS_VALUE
            else:
                x_out[mask] = 0.0

        if any(
            step is not None for step in (self.notch_filter, self.filter, self.frequency)
        ):
            x_out = self._filter_and_resample(x_out)

        if isinstance(self.scaler, float):
            x_out = x_out * self.scaler
        elif self.scaler == "RobustScaler":
            stats = x_out.quantile(
                torch.tensor([0.25, 0.5, 0.75], device=x_out.device),
                dim=self.scale_dim,  # type: ignore
                keepdim=True,
            )
            loc, iqr = stats[1], stats[2] - stats[0]
            iqr[iqr == 0.0] = 1.0  # Avoid division by zero
            x_out = (x_out - loc) / iqr
        elif self.scaler == "StandardScaler":
            x_out = x_out - x_out.mean(dim=self.scale_dim, keepdim=True)
            std = x_out.std(dim=self.scale_dim, keepdim=True)
            std[std == 0.0] = 1.0  # Avoid division by zero
            x_out = x_out / std
        elif self.scaler == "QuantileAbsScaler":
            q = torch.quantile(
                x_out.abs(),
                self.scaler_quantile,
                dim=self.scale_dim,  # type: ignore
                keepdim=True,
            )
            q[q == 0.0] = 1.0
            x_out = x_out / q
        elif self.scaler == "MinMaxScaler":
            x_min, x_max = torch.aminmax(x_out, dim=-1, keepdim=True)
            denom = x_max - x_min
            denom[denom == 0.0] = 1.0
            x_out = 2.0 * (x_out - x_min) / denom - 1.0

        if self.clamp is not None:
            x_out = x_out.clamp(-self.clamp, self.clamp)

        return x_out, ch_pos_out

    def _filter_and_resample(self, x: torch.Tensor) -> torch.Tensor:
        """Notch, band-pass and resample each window through its spectrum."""
        sfreq = self.sfreq
        assert sfreq is not None  # checked by the config
        n_times = x.shape[-1]
        freqs = torch.fft.rfftfreq(n_times, d=1.0 / sfreq, device=x.device)
        # brick-wall masks: exact on the window's frequency bins, ringing between them
        keep = torch.ones_like(freqs, dtype=torch.bool)
        notch = self.notch_filter if self.notch_filter is not None else []
        for base in notch if isinstance(notch, list) else [notch]:
            for harmonic in torch.arange(base, min(sfreq / 2, 301), base).tolist():
                keep &= (freqs - harmonic).abs() > 1.0
        if self.filter is not None:
            l_freq, h_freq = self.filter
            if l_freq is not None:
                keep &= freqs >= l_freq
            if h_freq is not None:
                keep &= freqs <= h_freq
        n_out = (
            n_times if self.frequency is None else round(n_times * self.frequency / sfreq)
        )
        # irfft trims or zero-pads the spectrum: an ideal low-pass when downsampling
        spectrum = torch.fft.rfft(x, dim=-1) * keep
        return torch.fft.irfft(spectrum, n=n_out, dim=-1) * (n_out / n_times)
