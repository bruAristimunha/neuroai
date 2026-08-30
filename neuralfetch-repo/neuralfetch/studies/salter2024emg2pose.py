# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""NM000281 (Meta emg2pose) -- surface-EMG hand-pose recordings."""

from __future__ import annotations

import json
import typing as tp
from pathlib import Path

import mne_bids
import pandas as pd
import pydantic

from neuralfetch import download
from neuralfetch.bids import BidsEmg  # noqa: F401
from neuralset.events import study


class Salter2024Emg2pose(study.Study):
    """emg2pose (Meta Reality Labs, NeurIPS 2024) -- surface-EMG hand pose.

    Notes
    -----
    The paper's train/val/test and generalization assignments are read from
    ``sourcedata/emg2pose_metadata.csv``, which must be materialized from the
    release before running the task.

    Events cover the valid inverse-kinematics spans recorded in the BIDS
    ``events.tsv`` files.
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
    IK_ANNOTATION: tp.ClassVar[str] = "BAD_IK"
    #: Paper split fields not represented by BIDS entities or sidecars.
    SPLIT_COLUMNS: tp.ClassVar[tuple[str, ...]] = (
        "split",
        "generalization",
    )
    aliases: tp.ClassVar[tuple[str, ...]] = ("emg2pose", "nm000281")
    _participant_users_cache: dict[str, str] | None = pydantic.PrivateAttr(default=None)
    _split_metadata_cache: dict[str, dict[str, tp.Any]] | None = pydantic.PrivateAttr(
        default=None
    )

    def _download(self, overwrite: bool = False) -> None:
        download.Eegdash(study=self.NEMAR_DATASET_ID, dset_dir=self.path).download(
            overwrite=overwrite
        )

    @property
    def bids_root(self) -> Path:
        """Return the BIDS root created by Eegdash or supplied by the user."""
        candidate = self.path / "download" / self.NEMAR_DATASET_ID
        if not (candidate.is_dir() and any(candidate.glob("sub-*"))):
            raise FileNotFoundError(
                f"No BIDS tree found under {candidate}. Run Study.download() or "
                f"symlink an existing {self.NEMAR_DATASET_ID} BIDS copy there."
            )
        return candidate

    @property
    def participant_users(self) -> dict[str, str]:
        """Map BIDS subject labels to the release's anonymized user labels."""
        if self._participant_users_cache is not None:
            return self._participant_users_cache
        self._participant_users_cache = {
            participant.removeprefix("sub-"): user
            for participant, user in pd.read_csv(
                self.bids_root / "participants.tsv",
                sep="\t",
                usecols=["participant_id", "original_user"],
            )
            .dropna()
            .itertuples(index=False, name=None)
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
        splits = tp.cast(
            dict[str, dict[str, tp.Any]],
            pd.read_csv(self.bids_root / "sourcedata" / "emg2pose_metadata.csv")
            .set_index("filename")[list(self.SPLIT_COLUMNS)]
            .to_dict("index"),
        )
        self._split_metadata_cache = splits
        return splits

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
            source_file = sidecar["SourceFile"]
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
            timeline.update(self.recording_splits[Path(source_file).stem])
            yield timeline

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        filepath = timeline["path"]
        return pd.DataFrame(
            [
                dict(type="BidsEmg", filepath=filepath, start=start, duration=duration)
                for start, duration in self._event_spans(filepath)
            ]
        )

    def _event_spans(self, filepath: str) -> list[tuple[float, float]]:
        """Return valid, non-IK-failure spans from the BIDS events."""
        path = Path(filepath)
        events = pd.read_csv(
            path.with_name(path.name.replace("_emg.bdf", "_events.tsv")), sep="\t"
        )
        valid = (
            (events["onset"] + events["duration"])
            .where(events["trial_type"] != self.IK_ANNOTATION)
            .max()
        )
        bad = sorted(
            (max(0.0, onset), min(valid, onset + duration))
            for onset, duration in events.query("trial_type == @self.IK_ANNOTATION")[
                ["onset", "duration"]
            ].itertuples(index=False, name=None)
            if onset < valid and onset + duration > 0
        )
        spans, cursor = [], 0.0
        for start, stop in bad:
            if cursor < start:
                spans.append((cursor, start - cursor))
            cursor = max(cursor, stop)
        if cursor < valid:
            spans.append((cursor, valid - cursor))
        return spans
