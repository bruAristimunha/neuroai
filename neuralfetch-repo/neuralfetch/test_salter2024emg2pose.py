# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the EMG2Pose BIDS study source."""

from pathlib import Path

from neuralfetch.studies.salter2024emg2pose import Salter2024Emg2pose


def test_emg2pose_bids_event(tmp_path: Path) -> None:
    study = Salter2024Emg2pose(path=tmp_path)
    root = study.path / "download" / study.NEMAR_DATASET_ID
    emg_dir = root / "sub-01" / "ses-01" / "emg"
    emg_dir.mkdir(parents=True)
    filename = "sub-01_ses-01_task-emg2pose_recording-left_emg.bdf"
    bdf = emg_dir / filename
    bdf.write_bytes(b"BDF")
    (root / "participants.tsv").write_text(
        "participant_id\toriginal_user\nsub-01\tuser-01\n"
    )
    (root / "sub-01" / "ses-01" / "sub-01_ses-01_scans.tsv").write_text(
        "filename\tsplit\tgeneralization\n"
        f"emg/{filename}\ttrain\tuser\n"
    )
    bdf.with_name(filename.replace("_emg.bdf", "_events.tsv")).write_text(
        "onset\tduration\ttrial_type\n0.0\t10.0\tstage\n2.0\t3.0\tBAD_IK\n"
    )

    timeline = next(study.iter_timelines())
    events = study._load_timeline_events(timeline)

    assert timeline["path"] == str(bdf)
    assert timeline["split"] == "train"
    assert events.to_dict("records") == [
        {"type": "BidsEmg", "filepath": str(bdf), "start": 0.0, "duration": 2.0},
        {"type": "BidsEmg", "filepath": str(bdf), "start": 5.0, "duration": 5.0},
    ]
