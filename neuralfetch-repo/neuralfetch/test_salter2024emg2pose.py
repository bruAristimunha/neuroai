# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the EMG2Pose BIDS study source."""

import json
from pathlib import Path

import mne
import mne_bids
import numpy as np
import pytest

from neuralfetch import download
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
        f"filename\tsplit\tgeneralization\nemg/{filename}\ttrain\tuser\n"
    )
    bdf.with_name(filename.replace("_emg.bdf", "_events.tsv")).write_text(
        "onset\tduration\ttrial_type\n0.0\t10.0\tstage\n2.0\t3.0\tBAD_IK\n"
    )

    timeline = next(study.iter_timelines())
    events = study._load_timeline_events(timeline)

    assert timeline["path"] == str(bdf)
    assert timeline["split"] == "train"
    assert events[["type", "start"]].to_dict("records") == [
        {"type": "BidsEmg", "start": 0.0}
    ]
    loader = json.loads(events.iloc[0]["filepath"])
    assert loader["method"] == "_load_raw"
    assert loader["timeline"] == timeline


def test_emg2pose_marks_bad_ik_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = mne.create_info(["emg", "target"], sfreq=10.0, ch_types=["emg", "misc"])
    raw = mne.io.RawArray(np.ones((2, 10)), info)
    raw.set_annotations(mne.Annotations([0.0, 0.2], [0.8, 0.3], ["stage", "BAD_IK"]))
    monkeypatch.setattr(mne_bids, "read_raw_bids", lambda *args, **kwargs: raw)

    loaded = Salter2024Emg2pose(path=tmp_path)._load_raw(
        {"path": str(tmp_path / "recording_emg.bdf")}
    )

    np.testing.assert_array_equal(
        loaded.get_data(picks="misc")[0],
        [1.0, 1.0, -1000.0, -1000.0, -1000.0, 1.0, 1.0, 1.0],
    )


def test_emg2pose_debug_downloads_one_subject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, str | None] = {}

    class Eegdash:
        def __init__(self, **kwargs: str | Path | None) -> None:
            subject = kwargs.get("subject")
            captured["subject"] = subject if isinstance(subject, str) else None

        def download(self, overwrite: bool = False) -> None:
            pass

    monkeypatch.setattr(download, "Eegdash", Eegdash)
    Salter2024Emg2pose(
        path=tmp_path, query="subject == 'Salter2024Emg2pose/13'"
    )._download()
    assert captured["subject"] == "13"
