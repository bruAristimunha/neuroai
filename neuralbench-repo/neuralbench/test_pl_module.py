# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Regression coverage for :mod:`neuralbench.pl_module`."""

from __future__ import annotations

import lightning.pytorch as pl
import pytest
import torch
from torch import nn
from torchmetrics.regression import MeanAbsoluteError

from neuralset.dataloader import Batch
from neuralset.segments import Segment
from neuraltrain.optimizers import BaseOptimizer, LightningOptimizer

from .pl_module import BrainModule


class TrajectoryModel(nn.Module):
    def forward(self, neuro: torch.Tensor) -> torch.Tensor:
        return neuro


def test_validation_step_updates_multivariate_trajectory_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A per-joint metric sees time points as observations, not outputs."""
    module = BrainModule(
        model=TrajectoryModel(),
        loss=nn.MSELoss(),
        metrics={"mae": MeanAbsoluteError(num_outputs=2)},
        lightning_optimizer_config=LightningOptimizer(
            optimizer=BaseOptimizer.model_validate({"name": "Adam", "lr": 1e-3})
        ),
    )
    module._trainer = pl.Trainer(
        accelerator="cpu",
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=False,
    )
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: None)
    batch = Batch(
        data={
            "neuro": torch.zeros(2, 3, 2),
            "target": torch.ones(2, 3, 2),
            "subject_id": torch.tensor([0, 1]),
        },
        segments=[
            Segment(0.0, 1.0, "subject-0"),
            Segment(0.0, 1.0, "subject-1"),
        ],
    )

    module.validation_step(batch, batch_idx=0)

    assert torch.equal(module.metrics["val/mae"].compute(), torch.tensor([1.0, 1.0]))
