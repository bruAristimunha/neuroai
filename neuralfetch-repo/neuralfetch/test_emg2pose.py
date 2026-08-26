"""Tests for the public ``Emg2pose`` BIDS study source."""

from __future__ import annotations

import json

import mne
import numpy as np
import pytest

from neuralfetch.studies.emg2pose import Emg2pose, Emg2poseRecording


def _make_bids_tree(root, recordings=1):
    """Build a small BIDS EMG2Pose tree in the Eegdash cache layout."""
    bids_root = root / "download" / Emg2pose.NEMAR_DATASET_ID
    (bids_root / "participants.tsv").parent.mkdir(parents=True, exist_ok=True)
    (bids_root / "participants.tsv").write_text(
        "participant_id\toriginal_user\nsub-01\tuser-123\n"
    )
    for run in range(1, recordings + 1):
        emg_dir = bids_root / "sub-01" / "ses-01" / "emg"
        emg_dir.mkdir(parents=True, exist_ok=True)
        stem = f"sub-01_ses-01_task-emg2pose_run-{run:02d}_recording-left_emg"
        (emg_dir / f"{stem}.bdf").write_bytes(b"\x00" * 16)
        (emg_dir / f"{stem}.json").write_text(
            json.dumps(
                {
                    "Stage": "unconstrained",
                    "HandSide": "left",
                    "SourceFile": f"recording-{run:02d}.hdf5",
                    "ValidSamples": 12_345,
                }
            )
        )
    return bids_root


def test_emg2pose_study_source(tmp_path):
    """BIDS discovery yields one raw EMG event for each BDF recording."""
    study = Emg2pose(path=tmp_path)
    bids_root = _make_bids_tree(study.path)

    assert study.bids_root == bids_root
    timelines = list(study.iter_timelines())
    assert timelines == [
        {
            "subject": "01",
            "session": "01",
            "task": "emg2pose",
            "run": "01",
            "recording": "left",
            "user": "user-123",
            "stage": "unconstrained",
            "side": "left",
            "source_file": "recording-01.hdf5",
            "valid_samples": 12_345,
            "user_stage": "user-123/unconstrained",
        }
    ]

    events = study._load_timeline_events(timelines[0])
    assert events["type"].tolist() == ["Emg2poseRecording"]
    assert events["subject"].tolist() == ["01"]
    assert events["filepath"].iloc[0].endswith("_run-01_recording-left_emg.bdf")
    assert events["user_stage"].tolist() == ["user-123/unconstrained"]
    assert events["valid_samples"].tolist() == [12_345]


def test_bids_reader_restores_emg_and_joint_channel_semantics(monkeypatch, tmp_path):
    """The BIDS reader keeps sidecar channel types and restores EMG scale."""
    channel_names = [
        *Emg2pose.EMG_CHANNEL_NAMES,
        *[f"joint{index}" for index in range(20)],
    ]
    raw = mne.io.RawArray(
        np.concatenate(
            (
                np.full((Emg2pose.EMG_CHANNEL_COUNT, 1), 5e-6),
                np.full((20, 1), 0.5),
            )
        ),
        mne.create_info(
            channel_names,
            sfreq=2000.0,
            ch_types=[
                *["emg"] * Emg2pose.EMG_CHANNEL_COUNT,
                *["misc"] * 20,
            ],
        ),
    )
    monkeypatch.setattr(
        "neuralfetch.studies.emg2pose.mne_bids.get_bids_path_from_fname",
        lambda _: object(),
    )
    monkeypatch.setattr(
        "neuralfetch.studies.emg2pose.mne_bids.read_raw_bids",
        lambda *_args, **_kwargs: raw.copy(),
    )

    path = tmp_path / "recording.bdf"
    path.touch()
    restored = Emg2poseRecording(
        filepath=str(path), start=0.0, timeline="recording", subject="01"
    )._read()

    assert restored.get_channel_types(picks=["emg0", "joint0"]) == ["emg", "misc"]
    assert restored.get_data(picks=["emg0", "joint0"]).ravel().tolist() == [5.0, 0.5]


@pytest.mark.parametrize("recordings", [1, 2, 3, 4])
def test_emg2pose_discovers_each_bids_run(tmp_path, recordings):
    """Runs are BIDS entities rather than an inferred paper manifest."""
    study = Emg2pose(path=tmp_path)
    _make_bids_tree(study.path, recordings=recordings)

    assert len(list(study.iter_timelines())) == recordings
