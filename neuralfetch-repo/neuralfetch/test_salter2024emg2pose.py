# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the EMG2Pose BIDS study source."""

from pathlib import Path
from types import SimpleNamespace

import mne
import pytest

from neuralfetch.studies.salter2024emg2pose import Salter2024Emg2pose


def test_emg2pose_bids_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    study = Salter2024Emg2pose(path=tmp_path)
    root = study.path / "download" / study.NEMAR_DATASET_ID
    emg_dir = root / "sub-01" / "ses-01" / "emg"
    emg_dir.mkdir(parents=True)
    filename = "sub-01_ses-01_task-emg2pose_recording-left_emg.bdf"
    bdf = emg_dir / filename
    bdf.write_bytes(b"BDF")
    bdf.with_suffix(".json").write_text('{"SourceFile": "run-01.hdf5"}')
    (root / "sourcedata").mkdir()
    (root / "sourcedata" / "emg2pose_metadata.csv").write_text(
        "filename,split,generalization\nrun-01,train,user\n"
    )
    (emg_dir.parent / "sub-01_ses-01_scans.tsv").write_text(
        f"filename\tduration\nemg/{filename}\t10.0\n"
    )
    monkeypatch.setattr(
        "neuralfetch.studies.salter2024emg2pose.mne.io.read_raw_bdf",
        lambda *_args, **_kwargs: SimpleNamespace(
            annotations=mne.Annotations([2.0], [3.0], [study.IK_ANNOTATION])
        ),
    )

    timeline = next(study.iter_timelines())
    events = study._load_timeline_events(timeline)

    assert timeline["path"] == str(bdf)
    assert timeline["split"] == "train"
    assert events.to_dict("records") == [
        {"type": "BidsEmg", "filepath": str(bdf), "start": 0.0, "duration": 2.0},
        {"type": "BidsEmg", "filepath": str(bdf), "start": 5.0, "duration": 5.0},
    ]
