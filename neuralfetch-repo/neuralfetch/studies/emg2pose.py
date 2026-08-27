"""EMG2Pose -- BIDS-native EMG hand-pose recordings."""

from __future__ import annotations

import json
import logging
import typing as tp
import urllib.error
import urllib.request
from pathlib import Path

import mne
import mne_bids
import numpy as np
import pandas as pd
import pydantic

from neuralfetch import download
from neuralset.events import etypes, study

logger = logging.getLogger(__name__)

#: Columns of the upstream ``emg2pose_metadata.csv`` that are not recoverable
#: from the BIDS entities and sidecars alone.
_SPLIT_COLUMNS = (
    "split",
    "generalization",
    "moving_hand",
    "held_out_user",
    "held_out_stage",
)

#: Public standalone copy of the metadata table, for users whose BIDS download
#: skipped ``sourcedata/``.
METADATA_URL = "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_metadata.csv"


def ik_failure_mask(joint_angles: np.ndarray) -> np.ndarray:
    """``True`` where the inverse-kinematics solver failed.

    Reproduces ``emg2pose.utils.get_ik_failures_mask``: the upstream release
    writes an all-zero joint vector wherever the offline IK solver could not
    resolve a frame, which the paper reports for 12.7% of frames. There is no
    separate annotation -- the zeros *are* the marker.

    Parameters
    ----------
    joint_angles : np.ndarray
        Joint angles shaped ``(joint, time)``.
    """
    return np.all(np.isclose(joint_angles, 0.0), axis=0)


def contiguous_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``(start, stop)`` index spans where *mask* is ``True``."""
    if mask.ndim != 1:
        raise ValueError(f"mask must be 1-D, got shape {mask.shape}")
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def _clip(spans: list[tuple[float, float]], limit: float) -> list[tuple[float, float]]:
    """Clip ``(start, stop)`` spans to ``[0, limit]``, as ``(start, duration)``."""
    out = []
    for start, stop in spans:
        start, stop = max(0.0, start), min(stop, limit)
        if stop > start:
            out.append((start, stop - start))
    return out


def _complement(
    bad: list[tuple[float, float]], limit: float
) -> list[tuple[float, float]]:
    """``(start, duration)`` spans of ``[0, limit]`` not covered by *bad*."""
    good: list[tuple[float, float]] = []
    cursor = 0.0
    for start, stop in sorted(bad):
        if start > cursor:
            good.append((cursor, min(start, limit)))
        cursor = max(cursor, stop)
        if cursor >= limit:
            break
    if cursor < limit:
        good.append((cursor, limit))
    return _clip(good, limit)


class Emg2poseRecording(etypes.Emg):
    """One EMG2Pose BIDS recording."""

    def _read(self) -> tp.Any:
        bids_path = mne_bids.get_bids_path_from_fname(self.filepath)
        return mne_bids.read_raw_bids(bids_path, verbose=False)


class Emg2pose(study.Study):
    """EMG2Pose hand-pose regression data published on NEMAR.

    BDF files and BIDS sidecars are the source of truth. No paper-specific
    checkpoint or recording manifest is embedded in this study.

    The paper's ``split`` / ``generalization`` assignments are not encoded in
    BIDS entities, but the release keeps each recording's upstream HDF5 name
    (``SourceFile`` in the sidecar), which is the key of the published
    ``emg2pose_metadata.csv``. When that table is available the split columns
    are joined onto every recording; when it is not, they are simply absent
    and the study stays usable BIDS-only.

    Parameters
    ----------
    split_metadata : Path, optional
        Explicit path to ``emg2pose_metadata.csv``. Defaults to the copy
        shipped inside the release at ``sourcedata/emg2pose_metadata.csv``;
        a standalone copy (~5 MiB) is published at :data:`METADATA_URL`.
    skip_ik_failures : bool
        Emit one event per contiguous span of frames whose inverse-kinematics
        labels resolved, instead of one event per recording. Matches the
        upstream ``skip_ik_failures`` datamodule default, which is ``True``
        for train and validation/test alike. Costs one pass over each
        recording's joint channels when the events are first built.
    """

    NEMAR_DATASET_ID: tp.ClassVar[str] = "nm000281"
    aliases: tp.ClassVar[tuple[str, ...]] = ("emg2pose", "nm000281")
    description: tp.ClassVar[str] = "16-channel EMG and hand-pose recordings."
    split_metadata: Path | None = None
    skip_ik_failures: bool = True
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
        self._download_split_metadata(overwrite=overwrite)

    def _download_split_metadata(self, overwrite: bool = False) -> None:
        """Fetch the standalone metadata table unless the release shipped one.

        Best-effort: a failure here leaves the study fully usable, just without
        the paper's split columns.
        """
        if self._existing_metadata_path() is not None and not overwrite:
            return
        target = self.path / "emg2pose_metadata.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(METADATA_URL, target)  # noqa: S310
        except (urllib.error.URLError, OSError) as exc:
            logger.warning("Could not fetch %s: %s", METADATA_URL, exc)

    def _existing_metadata_path(self) -> Path | None:
        """First metadata table found among the supported locations."""
        candidates = []
        if self.split_metadata is not None:
            candidates.append(Path(self.split_metadata))
        else:
            candidates.append(self.bids_root / "sourcedata" / "emg2pose_metadata.csv")
            candidates.append(self.path / "emg2pose_metadata.csv")
        return next((c for c in candidates if c.is_file()), None)

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

        Returns an empty mapping (and warns once) when the metadata table is
        not present, so BIDS-only downloads keep working.
        """
        if self._split_metadata_cache is not None:
            return self._split_metadata_cache
        path = self._existing_metadata_path()
        if path is None:
            logger.warning(
                "No emg2pose metadata table found under %s; the paper's split and "
                "generalization columns will be absent. Fetch it from %s or pass "
                "split_metadata=.",
                self.path,
                METADATA_URL,
            )
            self._split_metadata_cache = {}
            return self._split_metadata_cache
        table = pd.read_csv(path)
        missing = {"filename", *_SPLIT_COLUMNS}.difference(table.columns)
        if missing:
            raise ValueError(
                f"{path} is missing expected column(s): {sorted(missing)}. "
                f"Expected the upstream emg2pose metadata table ({METADATA_URL})."
            )
        self._split_metadata_cache = {
            str(row["filename"]): {col: row[col] for col in _SPLIT_COLUMNS}
            for _, row in table.iterrows()
        }
        return self._split_metadata_cache

    def _splits_for(self, source_file: tp.Any) -> dict[str, tp.Any]:
        """Look up the split row for a recording's upstream HDF5 name."""
        splits = self.recording_splits
        if not splits or source_file is None or pd.isna(source_file):
            return {}
        # ``metadata.csv`` keys on the stem; the sidecar keeps the ``.hdf5``.
        return splits.get(Path(str(source_file)).stem, {})

    def _scan_durations(self, session_dir: Path) -> dict[str, float]:
        """Un-padded recording durations from a session's ``scans.tsv``.

        ``ValidSamples`` is absent from a minority of sidecars (2,785 of
        25,253 in NM000281), but ``scans.tsv`` covers every recording and
        agrees exactly with ``ValidSamples / SamplingFrequency`` wherever both
        are present, so it is the fallback for bounding an event.
        """
        if session_dir in self._scans_cache:
            return self._scans_cache[session_dir]
        durations: dict[str, float] = {}
        for path in session_dir.glob("*_scans.tsv"):
            table = pd.read_csv(path, sep="\t")
            if not {"filename", "duration"}.issubset(table.columns):
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
            timeline = {
                "subject": bids_path.subject,
                "session": bids_path.session,
                "task": bids_path.task,
                "run": bids_path.run,
                "recording": bids_path.recording,
                "user": user,
                "stage": stage,
                "side": sidecar.get("HandSide", bids_path.recording),
                "source_file": source_file,
                "valid_samples": sidecar.get("ValidSamples"),
                "sampling_frequency": sidecar.get("SamplingFrequency"),
                "scan_duration": self._scan_durations(bids_path.fpath.parent.parent).get(
                    bids_path.fpath.name
                ),
            }
            timeline["user_stage"] = f"{user}/{stage}" if stage else user
            timeline.update(self._splits_for(source_file))
            yield timeline

    JOINT_CHANNEL_PREFIX: tp.ClassVar[str] = "joint"
    #: NM000281 marks IK failures with this BDF annotation.
    IK_ANNOTATION: tp.ClassVar[str] = "BAD_IK"
    #: The BDF physical header declares ``uV`` for every channel, so MNE
    #: returns joint angles 1e6 times too small. IK failures are detected in
    #: radians, as upstream does, so a genuinely small angle is not mistaken
    #: for the all-zero failure marker.
    JOINT_SCALE: tp.ClassVar[float] = 1e6

    def _ik_clean_spans(
        self, filepath: str, valid_duration: float | None
    ) -> list[tuple[float, float]]:
        """``(start, duration)`` seconds for each span with resolved IK.

        NM000281 records IK failures as ``BAD_IK`` BDF annotations -- its
        README: "BAD_IK annotations mark samples where inverse-kinematics
        labels are all zero". Those annotations are the authoritative marker
        and are cheap to read, so they are preferred over recomputing the
        all-zero test on quantized joint samples. The recomputation stays as a
        fallback for a tree without annotations.

        Spans are clipped to *valid_duration*: the BDF writer pads the last
        data record with **edge values**, not zeros, so neither test excludes
        the padded tail on its own.
        """
        raw = mne.io.read_raw_bdf(filepath, preload=False, verbose="ERROR")
        limit = valid_duration if valid_duration is not None else raw.times[-1]

        annotations = raw.annotations
        descriptions = [str(d) for d in annotations.description]
        if self.IK_ANNOTATION in descriptions:
            bad = [
                (float(onset), float(onset) + float(duration))
                for onset, duration, description in zip(
                    annotations.onset,
                    annotations.duration,
                    descriptions,
                    strict=True,
                )
                if description == self.IK_ANNOTATION
            ]
            return _complement(bad, limit)

        return self._ik_clean_spans_from_signal(raw, limit)

    def _ik_clean_spans_from_signal(
        self, raw: tp.Any, limit: float
    ) -> list[tuple[float, float]]:
        """Fallback: recompute the all-zero test from the joint channels."""
        picks = [
            name for name in raw.ch_names if name.startswith(self.JOINT_CHANNEL_PREFIX)
        ]
        if not picks:
            raise ValueError(
                f"{raw.filenames[0]} has neither {self.IK_ANNOTATION} annotations nor "
                f"{self.JOINT_CHANNEL_PREFIX}* channels; cannot detect IK failures."
            )
        joints = raw.get_data(picks=picks) * self.JOINT_SCALE
        frequency = float(raw.info["sfreq"])
        spans = contiguous_spans(~ik_failure_mask(joints))
        seconds = [(start / frequency, stop / frequency) for start, stop in spans]
        return _clip([(a, b) for a, b in seconds], limit)

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
        yield_rows: list[dict[str, tp.Any]] = []
        filepath = str(matches[0].fpath)
        base = {
            "type": "Emg2poseRecording",
            "filepath": filepath,
            "subject": timeline["subject"],
            "user": timeline["user"],
            "stage": timeline["stage"],
            "side": timeline["side"],
            "source_file": timeline["source_file"],
            "valid_samples": timeline["valid_samples"],
            "user_stage": timeline["user_stage"],
        }
        for column in _SPLIT_COLUMNS:
            if column in timeline:
                base[column] = timeline[column]

        for start, duration in self._event_spans(timeline, filepath):
            span = dict(base, start=start)
            if duration is not None:
                span["duration"] = duration
            yield_rows.append(span)
        return pd.DataFrame(yield_rows)

    def _event_spans(
        self, timeline: dict[str, tp.Any], filepath: str
    ) -> list[tuple[float, float | None]]:
        """Spans of a recording to emit as events.

        With ``skip_ik_failures`` this is one span per contiguous run of
        resolved IK frames; otherwise it is the single un-padded region.
        Spans shorter than a window are not filtered here -- the study does
        not know the window length, and ``stride_drop_incomplete`` drops them
        at segmentation time.
        """
        valid = self._valid_duration(timeline)
        if self.skip_ik_failures:
            return list(self._ik_clean_spans(filepath, valid))
        return [(0.0, valid)]

    @staticmethod
    def _valid_duration(timeline: dict[str, tp.Any]) -> float | None:
        """Duration of the un-padded region of a recording, in seconds.

        Every BDF in the release is zero-padded up to a whole number of
        one-second records, so the file is longer than the data. The sidecar
        records how much of it is real (``ValidSamples``); bounding the event
        there keeps sliding windows off the padded tail. Where that field is
        missing, the session's ``scans.tsv`` duration says the same thing.
        Returns ``None`` when neither is available, leaving the duration to be
        auto-filled from the file as before.
        """
        samples = timeline.get("valid_samples")
        frequency = timeline.get("sampling_frequency")
        if (
            samples is not None
            and frequency is not None
            and not pd.isna(samples)
            and not pd.isna(frequency)
            and float(frequency) > 0
        ):
            return float(samples) / float(frequency)
        scanned = timeline.get("scan_duration")
        if scanned is None or pd.isna(scanned) or float(scanned) <= 0:
            return None
        return float(scanned)
