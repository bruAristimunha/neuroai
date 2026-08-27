"""Tests for the public ``Emg2pose`` BIDS study source."""

from __future__ import annotations

import json

import mne
import numpy as np
import pytest

from neuralfetch.studies.emg2pose import (
    Emg2pose,
    Emg2poseRecording,
    _complement,
    contiguous_spans,
    ik_failure_mask,
)


def _make_split_metadata(bids_root, recordings=1, splits=("train", "test")):
    """Write an upstream-shaped metadata table keyed on the HDF5 stem."""
    path = bids_root / "sourcedata" / "emg2pose_metadata.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "session,user,stage,start,end,side,filename,moving_hand,"
        "held_out_user,held_out_stage,split,generalization\n"
    )
    rows = "".join(
        f"ses-01,user-123,unconstrained,0.0,1.0,left,recording-{run:02d},both,"
        f"True,False,{splits[(run - 1) % len(splits)]},user\n"
        for run in range(1, recordings + 1)
    )
    path.write_text(header + rows)
    return path


def _make_bids_tree(root, recordings=1):
    """Build a small BIDS EMG2Pose tree in the Eegdash cache layout."""
    bids_root = root / "download" / Emg2pose.NEMAR_DATASET_ID
    (bids_root / "participants.tsv").parent.mkdir(parents=True, exist_ok=True)
    (bids_root / "participants.tsv").write_text(
        "participant_id\toriginal_user\nsub-01\tuser-123\n"
    )
    scans = ["filename\tacq_time\tduration\tstage\tside\tsource_file"]
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
                    "SamplingFrequency": 2000.0,
                }
            )
        )
        scans.append(
            f"emg/{stem}.bdf\tn/a\t6.1725\tunconstrained\tleft\trecording-{run:02d}.hdf5"
        )
    session_dir = bids_root / "sub-01" / "ses-01"
    (session_dir / "sub-01_ses-01_scans.tsv").write_text("\n".join(scans) + "\n")
    return bids_root


def test_emg2pose_study_source(tmp_path):
    """BIDS discovery yields one raw EMG event for each BDF recording."""
    study = Emg2pose(path=tmp_path, skip_ik_failures=False)
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
            "sampling_frequency": 2000.0,
            "scan_duration": 6.1725,
            "user_stage": "user-123/unconstrained",
        }
    ]

    events = study._load_timeline_events(timelines[0])
    assert events["type"].tolist() == ["Emg2poseRecording"]
    assert events["subject"].tolist() == ["01"]
    assert events["filepath"].iloc[0].endswith("_run-01_recording-left_emg.bdf")
    assert events["user_stage"].tolist() == ["user-123/unconstrained"]
    assert events["valid_samples"].tolist() == [12_345]


def test_bids_reader_uses_bids_channel_metadata(monkeypatch, tmp_path):
    """The BIDS reader preserves the BIDS channel types and SI values."""
    channel_names = [
        *[f"emg{index}" for index in range(16)],
        *[f"joint{index}" for index in range(20)],
    ]
    raw = mne.io.RawArray(
        np.concatenate(
            (
                np.full((16, 1), 5e-6),
                np.full((20, 1), 0.5),
            )
        ),
        mne.create_info(
            channel_names,
            sfreq=2000.0,
            ch_types=[
                *["emg"] * 16,
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
    assert restored.get_data(picks=["emg0", "joint0"]).ravel().tolist() == [5e-6, 0.5]


@pytest.mark.parametrize("recordings", [1, 2, 3, 4])
def test_emg2pose_discovers_each_bids_run(tmp_path, recordings):
    """Runs are BIDS entities rather than an inferred paper manifest."""
    study = Emg2pose(path=tmp_path, skip_ik_failures=False)
    _make_bids_tree(study.path, recordings=recordings)

    assert len(list(study.iter_timelines())) == recordings


def test_event_duration_stops_at_valid_samples(tmp_path):
    """Windows are bounded by the un-padded region, not the padded BDF."""
    study = Emg2pose(path=tmp_path, skip_ik_failures=False)
    _make_bids_tree(study.path)

    timeline = next(iter(study.iter_timelines()))
    events = study._load_timeline_events(timeline)

    # 12_345 samples at 2 kHz, not the whole-second padded file length.
    assert events["duration"].tolist() == [12_345 / 2000.0]


def test_event_duration_falls_back_to_scans_tsv(tmp_path):
    """ValidSamples is absent from some sidecars; scans.tsv covers them."""
    study = Emg2pose(path=tmp_path, skip_ik_failures=False)
    _make_bids_tree(study.path)

    timeline = next(iter(study.iter_timelines()))
    timeline["valid_samples"] = None
    events = study._load_timeline_events(timeline)

    # 6.1725 s == 12_345 / 2000: scans.tsv states the same un-padded span.
    assert events["duration"].tolist() == [6.1725]


def test_event_duration_absent_without_any_source(tmp_path):
    """With neither ValidSamples nor scans.tsv, duration is left to auto-fill."""
    study = Emg2pose(path=tmp_path, skip_ik_failures=False)
    _make_bids_tree(study.path)

    timeline = next(iter(study.iter_timelines()))
    timeline["valid_samples"] = None
    timeline["scan_duration"] = None
    events = study._load_timeline_events(timeline)

    assert "duration" not in events.columns


def test_paper_splits_joined_from_metadata(tmp_path):
    """metadata.csv joins onto recordings through the sidecar's SourceFile."""
    study = Emg2pose(path=tmp_path, skip_ik_failures=False)
    bids_root = _make_bids_tree(study.path, recordings=2)
    _make_split_metadata(bids_root, recordings=2)

    timelines = list(study.iter_timelines())
    assert [t["split"] for t in timelines] == ["train", "test"]
    assert {t["generalization"] for t in timelines} == {"user"}

    events = study._load_timeline_events(timelines[0])
    assert events["split"].tolist() == ["train"]
    assert events["held_out_user"].tolist() == [True]


def test_split_columns_absent_without_metadata(tmp_path):
    """A BIDS-only download stays usable, just without the paper's splits."""
    study = Emg2pose(path=tmp_path, skip_ik_failures=False)
    _make_bids_tree(study.path)

    timeline = next(iter(study.iter_timelines()))
    assert "split" not in timeline
    assert "generalization" not in timeline
    assert study.recording_splits == {}

    events = study._load_timeline_events(timeline)
    assert "split" not in events.columns


def test_malformed_metadata_table_is_rejected(tmp_path):
    """A CSV without the expected columns fails loudly rather than silently."""
    study = Emg2pose(path=tmp_path, skip_ik_failures=False)
    bids_root = _make_bids_tree(study.path)
    path = bids_root / "sourcedata" / "emg2pose_metadata.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("filename,split\nrecording-01,train\n")

    with pytest.raises(ValueError, match="missing expected column"):
        _ = study.recording_splits


def test_ik_failure_mask_marks_all_zero_frames():
    """An IK failure is an all-zero joint vector, as upstream defines it."""
    joints = np.zeros((20, 6))
    joints[:, 2:4] = 0.5
    joints[3, 5] = 1e-3  # a single non-zero joint is enough to count as resolved

    assert ik_failure_mask(joints).tolist() == [True, True, False, False, True, False]


def test_ik_failure_mask_keeps_small_angles():
    """Small but real angles in radians are not mistaken for the zero marker."""
    joints = np.full((20, 3), 1e-4)

    assert not ik_failure_mask(joints).any()


@pytest.mark.parametrize(
    "mask, expected",
    [
        ([0, 0, 0], []),
        ([1, 1, 1], [(0, 3)]),
        ([0, 1, 1, 0, 1], [(1, 3), (4, 5)]),
        ([1, 0, 0, 1], [(0, 1), (3, 4)]),
    ],
)
def test_contiguous_spans(mask, expected):
    """Spans are half-open and cover every run of True."""
    assert contiguous_spans(np.array(mask, dtype=bool)) == expected


def test_events_split_on_ik_failures(monkeypatch, tmp_path):
    """With skip_ik_failures, each resolved run becomes its own event."""
    study = Emg2pose(path=tmp_path)
    _make_bids_tree(study.path)
    monkeypatch.setattr(
        Emg2pose,
        "_ik_clean_spans",
        lambda self, _path, _valid: [(0.0, 2.5), (4.0, 1.5)],
    )

    timeline = next(iter(study.iter_timelines()))
    events = study._load_timeline_events(timeline)

    assert events["start"].tolist() == [0.0, 4.0]
    assert events["duration"].tolist() == [2.5, 1.5]
    # Recording-level metadata is copied onto every span.
    assert events["user"].tolist() == ["user-123", "user-123"]
    assert events["type"].tolist() == ["Emg2poseRecording"] * 2


@pytest.mark.parametrize(
    "bad, limit, expected",
    [
        ([], 10.0, [(0.0, 10.0)]),
        ([(2.0, 3.0)], 10.0, [(0.0, 2.0), (3.0, 7.0)]),
        ([(0.0, 10.0)], 10.0, []),
        ([(0.0, 2.0)], 10.0, [(2.0, 8.0)]),
        # A BAD_IK span running into the padded tail is clipped at the limit.
        ([(2.0, 4.0)], 3.0, [(0.0, 2.0)]),
    ],
)
def test_complement_of_bad_spans(bad, limit, expected):
    """Clean spans are the complement of BAD_IK, clipped to the valid region."""
    assert _complement(bad, limit) == expected


def test_ik_spans_prefer_bad_ik_annotations(monkeypatch, tmp_path):
    """NM000281 marks IK failures with BAD_IK annotations; use them, not the
    all-zero recomputation, and never read the joint signal to find them."""
    study = Emg2pose(path=tmp_path)
    raw = mne.io.RawArray(
        np.ones((36, 20_000)),
        mne.create_info(
            [f"emg{i}" for i in range(16)] + [f"joint{i}" for i in range(20)],
            sfreq=2000.0,
            ch_types=["emg"] * 16 + ["misc"] * 20,
        ),
        verbose="ERROR",
    )
    raw.set_annotations(
        mne.Annotations(onset=[2.0], duration=[3.0], description=["BAD_IK"])
    )
    monkeypatch.setattr(
        "neuralfetch.studies.emg2pose.mne.io.read_raw_bdf",
        lambda *_a, **_k: raw,
    )

    def _fail(*_args, **_kwargs):
        raise AssertionError("must not fall back to the signal when annotated")

    monkeypatch.setattr(Emg2pose, "_ik_clean_spans_from_signal", _fail)

    assert study._ik_clean_spans("x.bdf", 10.0) == [(0.0, 2.0), (5.0, 5.0)]


def test_ik_spans_fall_back_to_signal_without_annotations(monkeypatch, tmp_path):
    """Without annotations, recompute the all-zero test, still clipped."""
    study = Emg2pose(path=tmp_path)
    joints = np.ones((20, 20_000)) * 1e-6  # 1.0 rad after JOINT_SCALE
    joints[:, 4_000:8_000] = 0.0
    raw = mne.io.RawArray(
        np.concatenate([np.ones((16, 20_000)), joints]),
        mne.create_info(
            [f"emg{i}" for i in range(16)] + [f"joint{i}" for i in range(20)],
            sfreq=2000.0,
            ch_types=["emg"] * 16 + ["misc"] * 20,
        ),
        verbose="ERROR",
    )
    monkeypatch.setattr(
        "neuralfetch.studies.emg2pose.mne.io.read_raw_bdf",
        lambda *_a, **_k: raw,
    )

    # Clipped at 9 s, so the trailing clean span stops short of the padding.
    assert study._ik_clean_spans("x.bdf", 9.0) == [(0.0, 2.0), (4.0, 5.0)]
