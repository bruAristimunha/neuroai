"""EMG2Pose -- BIDS-native EMG hand-pose recordings."""

from __future__ import annotations

import json
import typing as tp
from pathlib import Path

import mne_bids
import pandas as pd
import pydantic

from neuralfetch import download
from neuralset.events import etypes, study


class Emg2poseRecording(etypes.Emg):
    """One EMG2Pose BIDS recording."""

    def _read(self) -> tp.Any:
        bids_path = mne_bids.get_bids_path_from_fname(self.filepath)
        return mne_bids.read_raw_bids(bids_path, verbose=False)


class Emg2pose(study.Study):
    """EMG2Pose hand-pose regression data published on NEMAR.

    BDF files and BIDS sidecars are the source of truth. No paper-specific
    checkpoint, recording manifest, or split is embedded in this study.
    """

    NEMAR_DATASET_ID: tp.ClassVar[str] = "nm000281"
    aliases: tp.ClassVar[tuple[str, ...]] = ("emg2pose", "nm000281")
    description: tp.ClassVar[str] = "16-channel EMG and hand-pose recordings."
    _bids_root_cache: Path | None = pydantic.PrivateAttr(default=None)
    _participant_users_cache: dict[str, str] | None = pydantic.PrivateAttr(default=None)

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
            timeline = {
                "subject": bids_path.subject,
                "session": bids_path.session,
                "task": bids_path.task,
                "run": bids_path.run,
                "recording": bids_path.recording,
                "user": user,
                "stage": stage,
                "side": sidecar.get("HandSide", bids_path.recording),
                "source_file": sidecar.get("SourceFile"),
                "valid_samples": sidecar.get("ValidSamples"),
            }
            timeline["user_stage"] = f"{user}/{stage}" if stage else user
            yield timeline

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        matches = mne_bids.find_matching_paths(
            root=self.bids_root,
            subjects=timeline["subject"],
            sessions=timeline["session"],
            tasks=timeline["task"],
            runs=timeline["run"],
            recordings=timeline["recording"],
            datatypes="emg",
            extensions=".bdf",
        )
        if len(matches) != 1:
            raise ValueError(
                f"Expected one BDF for BIDS timeline {timeline}, got {len(matches)}"
            )
        return pd.DataFrame(
            [
                {
                    "type": "Emg2poseRecording",
                    "filepath": str(matches[0].fpath),
                    "start": 0.0,
                    "subject": timeline["subject"],
                    "user": timeline["user"],
                    "stage": timeline["stage"],
                    "side": timeline["side"],
                    "source_file": timeline["source_file"],
                    "valid_samples": timeline["valid_samples"],
                    "user_stage": timeline["user_stage"],
                }
            ]
        )
