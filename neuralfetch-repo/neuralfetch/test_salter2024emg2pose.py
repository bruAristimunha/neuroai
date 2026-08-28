# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the EMG2Pose BIDS study source."""

from __future__ import annotations

from pathlib import Path

import pytest
from exca.steps.patterns import BranchResult

from neuralfetch.studies.salter2024emg2pose import Salter2024Emg2pose


def _make_bids_tree(root: Path) -> tuple[Salter2024Emg2pose, Path]:
    study = Salter2024Emg2pose(path=root)
    bids_root = study.path / "download" / study.NEMAR_DATASET_ID
    emg_dir = bids_root / "sub-01" / "ses-01" / "emg"
    emg_dir.mkdir(parents=True)
    filename = "sub-01_ses-01_task-emg2pose_recording-left_emg.bdf"
    bdf = emg_dir / filename
    bdf.write_bytes(b"BDF")
    bdf.with_suffix(".json").write_text(
        '{"Stage": "free", "HandSide": "left", "SourceFile": "run-01.hdf5"}'
    )
    (bids_root / "participants.tsv").write_text(
        "participant_id\toriginal_user\nsub-01\tuser-01\n"
    )
    (emg_dir.parent / "sub-01_ses-01_scans.tsv").write_text(
        f"filename\tduration\nemg/{filename}\t10.0\n"
    )
    sourcedata = bids_root / "sourcedata"
    sourcedata.mkdir()
    (sourcedata / "emg2pose_metadata.csv").write_text(
        "filename,split,generalization,moving_hand,held_out_user,held_out_stage\n"
        "run-01,train,user,left,False,False\n"
    )
    return study, bdf


def test_timelines_keep_the_resolved_path_and_only_event_metadata(tmp_path: Path) -> None:
    """Loading a timeline reuses its path and never serializes missing BIDS fields."""
    study, bdf = _make_bids_tree(tmp_path)

    timeline = next(study.iter_timelines())

    assert timeline == {
        "subject": "01",
        "session": "01",
        "task": "emg2pose",
        "recording": "left",
        "path": str(bdf),
        "user": "user-01",
        "stage": "free",
        "side": "left",
        "user_stage": "user-01/free",
        "split": "train",
        "generalization": "user",
    }


def test_loading_reuses_the_timeline_path_without_a_second_bids_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 25k-recording study must not recursively discover each BDF twice."""
    study, bdf = _make_bids_tree(tmp_path)
    timeline = next(study.iter_timelines())
    monkeypatch.setattr(
        study, "_ik_clean_spans", lambda path, duration: [(0.0, duration)]
    )
    monkeypatch.setattr(
        "neuralfetch.studies.salter2024emg2pose.mne_bids.find_matching_paths",
        lambda **kwargs: pytest.fail("_load_timeline_events rescanned the BIDS tree"),
    )

    events = study._load_timeline_events(timeline)

    assert events.to_dict("records") == [
        {"type": "BidsEmg", "filepath": str(bdf), "start": 0.0, "duration": 10.0}
    ]


def test_missing_sourcedata_metadata_fails_before_predefined_split(
    tmp_path: Path,
) -> None:
    """The source-only contract rejects a BIDS tree without its split table."""
    study, _ = _make_bids_tree(tmp_path)
    (study.bids_root / "sourcedata" / "emg2pose_metadata.csv").unlink()

    with pytest.raises(FileNotFoundError, match="emg2pose_metadata.csv"):
        next(study.iter_timelines())


def test_unmatched_source_file_fails_before_predefined_split(tmp_path: Path) -> None:
    """Every BIDS recording must resolve to a paper split assignment."""
    study, bdf = _make_bids_tree(tmp_path)
    bdf.with_suffix(".json").write_text('{"SourceFile": "missing.hdf5"}')

    with pytest.raises(ValueError, match="No EMG2Pose split assignment"):
        next(study.iter_timelines())


def test_a_fully_failed_recording_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An all-BAD_IK recording contributes no invalid target event."""
    study, _ = _make_bids_tree(tmp_path)
    timeline = next(study.iter_timelines())
    monkeypatch.setattr(study, "_ik_clean_spans", lambda path, duration: [])

    events = study._load_one(timeline)

    assert events.empty
    with pytest.raises(RuntimeError, match="Salter2024Emg2pose produced no events"):
        study.gather([BranchResult(branch=timeline, result=events)])
