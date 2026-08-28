# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Regression coverage for :mod:`neuralbench.pl_module`."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from torchmetrics.regression import MeanAbsoluteError

from neuraltrain.metrics.metrics import GroupedMetric

from .pl_module import BrainModule


class _TrajectoryModel(nn.Module):
    def forward(self, neuro: torch.Tensor) -> torch.Tensor:
        return neuro


def test_trajectory_metrics_keep_the_output_axis() -> None:
    """A per-output metric accepts dense ``(batch, time, output)`` targets."""
    module = BrainModule(
        model=_TrajectoryModel(),
        loss=nn.MSELoss(),
        metrics={
            "mae": MeanAbsoluteError(num_outputs=2),
            "mae_by_subject": GroupedMetric(metric_name="MeanAbsoluteError"),
        },
        lightning_optimizer_config=SimpleNamespace(),
    )
    module._trainer = SimpleNamespace(world_size=1)
    module.log = lambda *args, **kwargs: None  # type: ignore[method-assign]
    batch = SimpleNamespace(
        data={
            "neuro": torch.zeros(2, 3, 2),
            "target": torch.ones(2, 3, 2),
            "subject_id": torch.tensor([0, 1]),
        }
    )

    module._run_step(batch, step_name="val", batch_idx=0)

    assert torch.equal(module.metrics["val/mae"].compute(), torch.tensor([1.0, 1.0]))
    assert module.metrics["val/mae_by_subject"].compute() == {
        "0": 1.0,
        "1": 1.0,
    }
