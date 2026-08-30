# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""NM000281 (Meta emg2pose) -- surface-EMG hand-pose recordings."""

from __future__ import annotations

import json
import logging
import typing as tp
from pathlib import Path

import mne
import mne_bids
import pandas as pd
import pydantic

from neuralfetch import download
from neuralfetch.bids import BidsEmg  # noqa: F401
from neuralset.events import study

LOGGER = logging.getLogger(__name__)


class Salter2024Emg2pose(study.Study):
    """emg2pose (Meta Reality Labs, NeurIPS 2024) -- surface-EMG hand pose.

    Notes
    -----
    The paper's train/val/test and generalization assignments are read from
    ``sourcedata/emg2pose_metadata.csv``, which must be materialized from the
    release before running the task.

    Events are emitted one per contiguous span of frames whose
    inverse-kinematics labels resolved, matching upstream's
    ``skip_ik_failures`` datamodule default. This is not configurable:
    neuralset rejects any study carrying non-default constructor fields
    ("Class parameters are not yet supported"), so such a knob would raise
    the moment it was set.
    """

    bibtex: tp.ClassVar[str] = """
    @inproceedings{salter2024emg2pose,
        author = {Salter, Sasha and Warren, Richard and Schlager, Collin and
                  Spurr, Adrian and Han, Shangchen and Bhasin, Rohin and
                  Cai, Yujun and Walkington, Peter and Bolarinwa, Anuoluwapo and
                  Wang, Robert and Danielson, Nathan and Merel, Josh and
                  Pnevmatikakis, Eftychios and Marshall, Jesse},
        title = {emg2pose: A Large and Diverse Benchmark for Surface
                 Electromyographic Hand Pose Estimation},
        booktitle = {Advances in Neural Information Processing Systems},
        volume = {37},
        year = {2024},
        url = {https://arxiv.org/abs/2412.02725},
    }
    """
    url: tp.ClassVar[str] = "https://nemar.org/dataexplorer/detail?dataset_id=NM000281"
    licence: tp.ClassVar[str] = "CC-BY-NC-SA-4.0"
    description: tp.ClassVar[str] = (
        "193 subjects performing staged hand movements with an EMG wristband, "
        "paired with tracked hand-joint angles."
    )

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=25253,
        num_subjects=193,
        num_events_in_query=16,
        event_types_in_query={"BidsEmg"},
        data_shape=(16, 3267),
        frequency=2000,
    )

    NEMAR_DATASET_ID: tp.ClassVar[str] = "nm000281"
    #: Paper split fields not represented by BIDS entities or sidecars.
    SPLIT_COLUMNS: tp.ClassVar[tuple[str, ...]] = (
        "split",
        "generalization",
    )
    aliases: tp.ClassVar[tuple[str, ...]] = ("emg2pose", "nm000281")
    _bids_root_cache: Path | None = pydantic.PrivateAttr(default=None)
    _participant_users_cache: dict[str, str] | None = pydantic.PrivateAttr(default=None)
    _split_metadata_cache: dict[str, dict[str, tp.Any]] | None = pydantic.PrivateAttr(
        default=None
    )
    _scans_cache: dict[Path, dict[str, float]] = pydantic.PrivateAttr(
        default_factory=dict
    )

    def _download(self, overwrite: bool = False) -> None:
        download.Eegdash(study=self.NEMAR_DATASET_ID, dset_dir=self.path).download(
            overwrite=overwrite
        )

    @property
    def bids_root(self) -> Path:
        """Return the BIDS root created by Eegdash or supplied by the user."""
        if self._bids_root_cache is not None:
            return self._bids_root_cache
        candidate = self.path / "download" / self.NEMAR_DATASET_ID
        if not (candidate.is_dir() and any(candidate.glob("sub-*"))):
            raise FileNotFoundError(
                f"No BIDS tree found under {candidate}. Run Study.download() or "
                f"symlink an existing {self.NEMAR_DATASET_ID} BIDS copy there."
            )
        self._bids_root_cache = candidate
        return candidate

    @property
    def participant_users(self) -> dict[str, str]:
        """Map BIDS subject labels to the release's anonymized user labels."""
        if self._participant_users_cache is not None:
            return self._participant_users_cache
        path = self.bids_root / "participants.tsv"
        if not path.is_file():
            self._participant_users_cache = {}
            return self._participant_users_cache
        participants = pd.read_csv(path, sep="\t", dtype="string")
        required = {"participant_id", "original_user"}
        if not required.issubset(participants.columns):
            self._participant_users_cache = {}
            return self._participant_users_cache
        self._participant_users_cache = {
            str(participant).removeprefix("sub-"): str(user)
            for participant, user in zip(
                participants["participant_id"],
                participants["original_user"],
                strict=True,
            )
            if pd.notna(participant) and pd.notna(user)
        }
        return self._participant_users_cache

    @property
    def recording_splits(self) -> dict[str, dict[str, tp.Any]]:
        """Map upstream HDF5 stems to the paper's split assignment.

        ``split`` and ``generalization`` are not BIDS entities, so they are
        joined from the release's required ``sourcedata`` table.
        """
        if self._split_metadata_cache is not None:
            return self._split_metadata_cache
        path = self.bids_root / "sourcedata" / "emg2pose_metadata.csv"
        table = pd.read_csv(path)
        required = {"filename", *self.SPLIT_COLUMNS}
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(
                f"{path} is not the materialized EMG2Pose metadata table; "
                f"missing {sorted(missing)}. Materialize the release's "
                "sourcedata/ contents before running the paper split."
            )
        splits = {
            str(row["filename"]): {column: row[column] for column in self.SPLIT_COLUMNS}
            for _, row in table.iterrows()
        }
        if not splits:
            raise ValueError(f"{path} contains no EMG2Pose split assignments")
        self._split_metadata_cache = splits
        return splits

    def _splits_for(self, source_file: tp.Any) -> dict[str, tp.Any]:
        """Look up the split row for a recording's upstream HDF5 name."""
        splits = self.recording_splits
        if source_file is None or pd.isna(source_file):
            raise ValueError("BIDS sidecar has no SourceFile for paper split lookup")
        # ``metadata.csv`` keys on the stem; the sidecar keeps the ``.hdf5``.
        source_stem = Path(str(source_file)).stem
        try:
            return splits[source_stem]
        except KeyError as error:
            raise ValueError(
                f"No EMG2Pose split assignment for BIDS SourceFile {source_file!r}"
            ) from error

    def _scan_durations(self, session_dir: Path) -> dict[str, float]:
        """Un-padded recording durations from a session's ``scans.tsv``."""
        if session_dir in self._scans_cache:
            return self._scans_cache[session_dir]
        durations: dict[str, float] = {}
        for path in session_dir.glob("*_scans.tsv"):
            table = pd.read_csv(path, sep="\t")
            missing = {"filename", "duration"}.difference(table.columns)
            if missing:
                LOGGER.warning(
                    "%s has no %s column; recordings in this session have no "
                    "un-padded length and will be rejected.",
                    path,
                    sorted(missing),
                )
                continue
            for name, duration in zip(table["filename"], table["duration"]):
                if pd.notna(duration):
                    durations[Path(str(name)).name] = float(duration)
        self._scans_cache[session_dir] = durations
        return durations

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Yield recordings from BIDS entities, never parsed file names."""
        for bids_path in mne_bids.find_matching_paths(
            root=self.bids_root, datatypes="emg", extensions=".bdf"
        ):
            sidecar_path = bids_path.fpath.with_suffix(".json")
            sidecar = (
                json.loads(sidecar_path.read_text()) if sidecar_path.is_file() else {}
            )
            subject = bids_path.subject
            user = self.participant_users.get(subject, subject)
            stage = sidecar.get("Stage")
            source_file = sidecar.get("SourceFile")
            values = {
                "subject": bids_path.subject,
                "session": bids_path.session,
                "task": bids_path.task,
                "run": bids_path.run,
                "recording": bids_path.recording,
                "path": str(bids_path.fpath),
                "user": user,
                "stage": stage,
                "side": sidecar.get("HandSide") or bids_path.recording,
            }
            timeline = {key: value for key, value in values.items() if value is not None}
            timeline["user_stage"] = f"{user}/{stage}" if stage else user
            timeline.update(self._splits_for(source_file))
            yield timeline

    #: NM000281 marks IK failures with this BDF annotation.
    IK_ANNOTATION: tp.ClassVar[str] = "BAD_IK"

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        yield_rows: list[dict[str, tp.Any]] = []
        filepath = timeline["path"]
        base = {
            "type": "BidsEmg",
            "filepath": filepath,
        }

        for start, duration in self._event_spans(filepath):
            yield_rows.append(dict(base, start=start, duration=duration))
        return pd.DataFrame(yield_rows)

    def _event_spans(self, filepath: str) -> list[tuple[float, float]]:
        """Return valid, non-IK-failure spans without BDF padding."""
        path = Path(filepath)
        valid = self._scan_durations(path.parent.parent).get(path.name)
        if valid is None:
            raise ValueError(
                f"No un-padded duration for {filepath}; refusing to emit an event "
                "that would extend into the BDF's edge-value padding."
            )
        annotations = mne.io.read_raw_bdf(
            filepath, preload=False, verbose="ERROR"
        ).annotations
        bad = sorted(
            (max(0.0, float(onset)), min(valid, float(onset) + float(duration)))
            for onset, duration, description in zip(
                annotations.onset,
                annotations.duration,
                annotations.description,
                strict=True,
            )
            if str(description) == self.IK_ANNOTATION
            and onset < valid
            and onset + duration > 0
        )
        spans, cursor = [], 0.0
        for start, stop in bad:
            if cursor < start:
                spans.append((cursor, start - cursor))
            cursor = max(cursor, stop)
        if cursor < valid:
            spans.append((cursor, valid - cursor))
        return spans
