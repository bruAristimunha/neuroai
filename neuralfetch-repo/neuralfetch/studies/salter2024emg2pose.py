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
import pandas as pd
import pydantic

from neuralfetch import download

# Imported for its side effect: registers the ``BidsEmg`` event type this
# study emits. Same reader, same BIDS conventions as emg2qwerty.
from neuralfetch.studies.sivakumar2024emg2qwerty import BidsEmg  # noqa: F401
from neuralset.events import study

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


class Salter2024Emg2pose(study.Study):
    """emg2pose (Meta Reality Labs, NeurIPS 2024) -- surface-EMG hand pose.

    Parameters
    ----------
    split_metadata : Path, optional
        Explicit path to ``emg2pose_metadata.csv``, which carries the paper's
        train/val/test and generalization assignments. Defaults to the copy
        shipped in the release at ``sourcedata/emg2pose_metadata.csv``; a
        standalone copy is published at :data:`METADATA_URL`. Without it the
        split columns are simply absent.
    skip_ik_failures : bool
        Emit one event per contiguous span of frames whose inverse-kinematics
        labels resolved, rather than one per recording. Matches the upstream
        datamodule default.
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
    description: tp.ClassVar[str] = (
        "193 subjects performing 29 hand-movement stages with a 16-channel EMG "
        "wristband, paired with motion-capture joint angles."
    )

    NEMAR_DATASET_ID: tp.ClassVar[str] = "nm000281"
    aliases: tp.ClassVar[tuple[str, ...]] = ("emg2pose", "nm000281")
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
        # Download to a sibling temp file and rename: a truncation that lands
        # after the CSV header would otherwise pass the column check and build
        # the split map from a prefix, silently dropping every recording past
        # the cut from train/val/test.
        partial = target.with_suffix(".csv.partial")
        try:
            urllib.request.urlretrieve(METADATA_URL, partial)  # noqa: S310
            partial.replace(target)
        except (urllib.error.URLError, OSError) as exc:
            partial.unlink(missing_ok=True)
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

        The BDF writer pads the final data record with edge values, so the
        file runs past the data and the event has to be bounded explicitly.
        ``scans.tsv`` covers every recording and agrees exactly with the
        sidecar's ``ValidSamples / SamplingFrequency`` wherever both are
        present (22,468 of 25,253; the rest have no ``ValidSamples``).
        """
        if session_dir in self._scans_cache:
            return self._scans_cache[session_dir]
        durations: dict[str, float] = {}
        for path in session_dir.glob("*_scans.tsv"):
            table = pd.read_csv(path, sep="\t")
            missing = {"filename", "duration"}.difference(table.columns)
            if missing:
                logger.warning(
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
                "duration": self._scan_durations(bids_path.fpath.parent.parent).get(
                    bids_path.fpath.name
                ),
            }
            timeline["user_stage"] = f"{user}/{stage}" if stage else user
            timeline.update(self._splits_for(source_file))
            yield timeline

    #: NM000281 marks IK failures with this BDF annotation.
    IK_ANNOTATION: tp.ClassVar[str] = "BAD_IK"

    def _ik_clean_spans(
        self, filepath: str, valid_duration: float | None
    ) -> list[tuple[float, float]]:
        """``(start, duration)`` seconds for each span with resolved IK.

        NM000281 records IK failures as ``BAD_IK`` annotations -- its README:
        "BAD_IK annotations mark samples where inverse-kinematics labels are
        all zero". The annotations are the only usable source here: measured
        over 60 recordings, upstream's all-zero test on the BDF joint samples
        agrees with them only 87% of the time on average (as low as 32%), and
        typically reports no failures at all where the annotations mark 15% of
        the recording. Quantization does not preserve the exact zeros the
        HDF5-side test relies on.

        A recording with no annotation has no IK failures, so its whole valid
        region is one clean span.

        Spans are clipped to *valid_duration*: the BDF writer pads the last
        data record with edge values, so the file runs past the data.
        """
        if valid_duration is None:
            # Falling back to the file length would hand the BDF's edge-value
            # padding to the loss as ground truth: constant joint angles, for
            # up to a second per recording. Refuse instead.
            raise ValueError(
                f"No un-padded duration for {filepath}: its session's scans.tsv "
                "has no usable 'duration' entry, and the BDF is padded with edge "
                "values, so the recording cannot be safely bounded."
            )
        raw = mne.io.read_raw_bdf(filepath, preload=False, verbose="ERROR")
        limit = valid_duration
        annotations = raw.annotations
        bad = [
            (float(onset), float(onset) + float(duration))
            for onset, duration, description in zip(
                annotations.onset,
                annotations.duration,
                annotations.description,
                strict=True,
            )
            if str(description) == self.IK_ANNOTATION
        ]
        return _complement(bad, limit)

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
            "type": "BidsEmg",
            "filepath": filepath,
            "subject": timeline["subject"],
            "user": timeline["user"],
            "stage": timeline["stage"],
            "side": timeline["side"],
            "source_file": timeline["source_file"],
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
        valid = timeline.get("duration")
        if valid is None:
            raise ValueError(
                f"No un-padded duration for {filepath}; refusing to emit an event "
                "that would extend into the BDF's edge-value padding."
            )
        if self.skip_ik_failures:
            return list(self._ik_clean_spans(filepath, valid))
        return [(0.0, valid)]
