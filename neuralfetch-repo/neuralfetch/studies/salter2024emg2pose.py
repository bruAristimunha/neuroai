# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""NM000281 (Meta emg2pose) -- surface-EMG hand-pose recordings."""

from __future__ import annotations

import logging
import re
import typing as tp
from pathlib import Path

import mne
import mne_bids
import numpy as np
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
    The NEMAR release carries no split: the paper's ``split`` and
    ``generalization`` labels come from the upstream ``emg2pose_metadata.csv``,
    joined per recording on the ``source_file`` of the BIDS ``scans.tsv``.

    Invalid inverse-kinematics samples are marked in the target channels with
    ``-1000`` so downstream losses and predictions can mask them.
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
        num_events_in_query=1,
        event_types_in_query={"BidsEmg"},
        data_shape=(16, 3267),
        frequency=2000,
    )

    NEMAR_DATASET_ID: tp.ClassVar[str] = "nm000281"
    IK_ANNOTATION: tp.ClassVar[str] = "BAD_IK"
    INVALID_TARGET: tp.ClassVar[float] = -1000.0
    METADATA_URL: tp.ClassVar[str] = (
        "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_metadata.csv"
    )
    METADATA_FIELDS: tp.ClassVar[tuple[str, ...]] = (
        "split",
        "generalization",
        "stage",
        "side",
    )
    aliases: tp.ClassVar[tuple[str, ...]] = ("emg2pose", "nm000281")
    _participant_users_cache: dict[str, str] | None = pydantic.PrivateAttr(default=None)
    _recording_metadata_cache: dict[str, dict[str, str]] | None = pydantic.PrivateAttr(
        default=None
    )
    _source_file_cache: dict[Path, dict[str, str]] = pydantic.PrivateAttr(
        default_factory=dict
    )

    @staticmethod
    def _query_subjects(query: str | None) -> list[str] | None:
        """Return the subject labels a subject-only query selects, else ``None``."""
        # ``None`` means "no safe scope": the caller then fetches everything.
        if query is None or not re.fullmatch(
            r"subject\s*(?:==\s*'[^']+'|in\s*\[[^\]]+\])", query.strip()
        ):
            return None
        return [
            label.rsplit("/", maxsplit=1)[-1] for label in re.findall(r"'([^']+)'", query)
        ]

    def _download(self, overwrite: bool = False) -> None:
        subjects = self._query_subjects(self.query)
        if subjects is None and self.query is not None:
            LOGGER.warning(
                "Query %r does not name its subjects, so the whole %s release "
                "(193 subjects, ~340 GB) will be downloaded.",
                self.query,
                self.NEMAR_DATASET_ID,
            )
        download.Eegdash(
            study=self.NEMAR_DATASET_ID,
            dset_dir=self.path,
            subject=subjects,
        ).download(overwrite=overwrite)
        if overwrite or not self.metadata_path.is_file():
            download.download_file(self.METADATA_URL, self.metadata_path)

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
    def metadata_path(self) -> Path:
        """Return the upstream metadata table holding the paper's splits."""
        return self.path / "emg2pose_metadata.csv"

    @property
    def recording_metadata(self) -> dict[str, dict[str, str]]:
        """Map upstream recording names to their paper split and description."""
        if self._recording_metadata_cache is None:
            if not self.metadata_path.is_file():
                raise FileNotFoundError(
                    f"{self.metadata_path} is missing. The NEMAR release does not "
                    "carry the paper's train/val/test assignment; run "
                    f"Study.download() to fetch it from {self.METADATA_URL}."
                )
            table = pd.read_csv(
                self.metadata_path, usecols=["filename", *self.METADATA_FIELDS]
            ).set_index("filename")
            self._recording_metadata_cache = tp.cast(
                dict[str, dict[str, str]], table.to_dict("index")
            )
        return self._recording_metadata_cache

    def _source_file(self, bids_path: mne_bids.BIDSPath) -> str:
        """Return the upstream recording name one BIDS file was converted from."""
        scans = mne_bids.BIDSPath(
            root=bids_path.root,
            subject=bids_path.subject,
            session=bids_path.session,
            suffix="scans",
            extension=".tsv",
        ).fpath
        if scans not in self._source_file_cache:
            table = pd.read_csv(scans, sep="\t").set_index("filename")
            self._source_file_cache[scans] = tp.cast(
                dict[str, str], table["source_file"].to_dict()
            )
        name = self._source_file_cache[scans][
            f"{bids_path.datatype}/{bids_path.fpath.name}"
        ]
        return name.removesuffix(".hdf5")

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Yield recordings from BIDS entities, never parsed file names."""
        for bids_path in mne_bids.find_matching_paths(
            root=self.bids_root, datatypes="emg", extensions=".bdf"
        ):
            user = self.participant_users.get(bids_path.subject, bids_path.subject)
            metadata = self.recording_metadata[self._source_file(bids_path)]
            values = {
                "subject": bids_path.subject,
                "session": bids_path.session,
                "task": bids_path.task,
                "run": bids_path.run,
                "recording": bids_path.recording,
                "path": str(bids_path.fpath),
                "user": user,
                **{field: metadata[field] for field in self.METADATA_FIELDS},
            }
            timeline = {key: value for key, value in values.items() if value is not None}
            timeline["user_stage"] = f"{user}/{metadata['stage']}"
            yield timeline

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        filepath = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        return pd.DataFrame([dict(type="BidsEmg", filepath=filepath, start=0.0)])

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.BaseRaw:
        """Load one recording and mark targets without valid IK."""
        raw = mne_bids.read_raw_bids(
            mne_bids.get_bids_path_from_fname(timeline["path"]), verbose=False
        ).load_data()
        valid_end = max(
            (
                annotation["onset"] + annotation["duration"]
                for annotation in raw.annotations
                if annotation["description"] != self.IK_ANNOTATION
            ),
            default=raw.times[-1] + 1 / raw.info["sfreq"],
        )
        stop = max(
            1, min(raw.n_times, raw.time_as_index([valid_end], use_rounding=True)[0])
        )
        if stop < raw.n_times:
            raw.crop(tmax=raw.times[stop - 1])
        invalid = np.zeros(raw.n_times, dtype=bool)
        for annotation in raw.annotations:
            if annotation["description"] != self.IK_ANNOTATION:
                continue
            start, stop = raw.time_as_index(
                [annotation["onset"], annotation["onset"] + annotation["duration"]],
                use_rounding=True,
            )
            invalid[max(0, start) : min(raw.n_times, stop)] = True
        if invalid.any():
            raw.apply_function(
                lambda data: np.where(invalid, self.INVALID_TARGET, data), picks="misc"
            )
        return raw
