# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import typing as tp

import numpy as np
import torch

import neuralset as ns


class SleepOnsetTargetExtractor(ns.extractors.BaseStatic):
    """Compute the time-to-N2-onset target dynamically from the segment's ``stop``.

    Reads the absolute ``n2_onset`` timestamp from a ``SleepOnsetMarker``
    event (emitted by :class:`~neuralbench.transforms.AddSleepOnsetTargets`)
    and returns ``clip(n2_onset - segment.stop, 0, cap_s)``.  Computing the
    target from the actual segment boundary -- rather than reading a value
    baked into the event at transform time -- guarantees the label always
    matches the EEG window being fed to the model, even if the segmenter's
    ``duration`` or ``stride`` changes.

    Parameters
    ----------
    event_types : str or tuple of str
        Type of event(s) to read the ``n2_onset`` field from.  Defaults to
        ``"SleepOnsetMarker"``.
    cap_s : float
        Maximum target value; values larger than this are clipped (matches
        the competition spec where targets are capped at 10 minutes
        pre-onset).
    """

    event_types: str | tuple[str, ...] = "SleepOnsetMarker"
    cap_s: float = 600.0

    def prepare(self, obj: tp.Any) -> None:
        pass

    def get_static(self, event: ns.events.Event) -> torch.Tensor:
        """Unused: ``_get_timed_arrays`` is overridden to depend on the segment bounds."""
        raise NotImplementedError

    def _get_timed_arrays(
        self, events: list[ns.events.Event], start: float, duration: float
    ) -> tp.Iterable[ns.base.TimedArray]:
        stop = start + duration
        for event in events:
            n2_onset = float(event._get_field_or_extra("n2_onset"))
            time_to_onset = np.clip(n2_onset - stop, 0.0, self.cap_s)
            embedding = torch.tensor([time_to_onset], dtype=torch.float32)
            yield ns.base.TimedArray(
                frequency=0,
                duration=event.duration,
                start=event.start,
                data=embedding.numpy(),
            )


class TimeMajorExtractor(ns.extractors.BaseExtractor):
    """Wrap an extractor and swap its channel and time axes: ``(C, T) -> (T, C)``.

    :meth:`neuralbench.pl_module.PlModule._run_step` flattens a 3-D
    prediction and a 3-D target to ``(batch, -1)`` before handing them to the
    loss, so the two have to agree on axis order.  Dense regression backbones
    emit ``(B, T, C)`` (braindecode's convention for per-frame outputs) while
    :class:`~neuralset.extractors.MneRaw` yields ``(C, T)``; without a
    transpose the loss would compare a time-major prediction against a
    channel-major target and silently optimise nonsense.

    Parameters
    ----------
    extractor : BaseExtractor
        The extractor whose output should be transposed.
    """

    event_types: str | tuple[str, ...] = "Event"
    extractor: ns.extractors.BaseExtractor

    def model_post_init(self, log__: tp.Any) -> None:
        self.event_types = self.extractor.event_types
        self.frequency = self.extractor.frequency
        return super().model_post_init(log__)

    def prepare(self, events: tp.Any) -> None:
        self.extractor.prepare(events)

    def __call__(
        self,
        events: tp.Any,
        start: float,
        duration: float,
        trigger: tp.Any = None,
    ) -> torch.Tensor:
        out = self.extractor(events, start, duration, trigger)
        return out.transpose(-1, -2)
