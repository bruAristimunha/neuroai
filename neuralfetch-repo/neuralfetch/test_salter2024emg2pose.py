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


def _make_release(study: Salter2024Emg2pose) -> Path:
    """Write the release layout the study reads: BIDS tree + upstream metadata.

    The NEMAR ``scans.tsv`` carries no split, only the ``source_file`` that
    joins each recording to ``emg2pose_metadata.csv``.
    """
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
        f"filename\tsource_file\nemg/{filename}\trec-1_left.hdf5\n"
    )
    study.metadata_path.write_text(
        "filename,split,generalization,stage,side\n"
        "rec-1_left,train,none,HandClawGraspFlicks,left\n"
    )
    bdf.with_name(filename.replace("_emg.bdf", "_events.tsv")).write_text(
        "onset\tduration\ttrial_type\n0.0\t10.0\tstage\n2.0\t3.0\tBAD_IK\n"
    )
    return bdf


def test_emg2pose_bids_event(tmp_path: Path) -> None:
    study = Salter2024Emg2pose(path=tmp_path)
    bdf = _make_release(study)

    timeline = next(study.iter_timelines())
    events = study._load_timeline_events(timeline)

    assert timeline["path"] == str(bdf)
    assert timeline["split"] == "train"
    assert timeline["generalization"] == "none"
    assert timeline["stage"] == "HandClawGraspFlicks"
    assert timeline["user_stage"] == "user-01/HandClawGraspFlicks"
    assert events[["type", "start"]].to_dict("records") == [
        {"type": "BidsEmg", "start": 0.0}
    ]
    loader = json.loads(events.iloc[0]["filepath"])
    assert loader["method"] == "_load_raw"
    assert loader["timeline"] == timeline


def test_emg2pose_missing_metadata_fails_clearly(tmp_path: Path) -> None:
    """A release without the upstream table must not yield an unlabelled split."""
    study = Salter2024Emg2pose(path=tmp_path)
    _make_release(study)
    study.metadata_path.unlink()

    with pytest.raises(FileNotFoundError, match="emg2pose_metadata.csv"):
        next(study.iter_timelines())


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


@pytest.mark.parametrize(
    "query,expected",
    [
        ("subject == 'Salter2024Emg2pose/13'", ["13"]),
        ("subject in ['Salter2024Emg2pose/60', 'Salter2024Emg2pose/166']", ["60", "166"]),
        # Not a subject selector: the full release is the only safe scope.
        ("timeline_index < 8", None),
        (None, None),
    ],
)
def test_emg2pose_download_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    query: str | None,
    expected: list[str] | None,
) -> None:
    captured: dict[str, object] = {}

    class Eegdash:
        def __init__(self, **kwargs: object) -> None:
            captured["subject"] = kwargs.get("subject")

        def download(self, overwrite: bool = False) -> None:
            pass

    monkeypatch.setattr(download, "Eegdash", Eegdash)
    monkeypatch.setattr(download, "download_file", lambda *a, **k: None)
    Salter2024Emg2pose(path=tmp_path, query=query)._download()
    assert captured["subject"] == expected
