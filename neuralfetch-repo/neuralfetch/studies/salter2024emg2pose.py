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

# Imported for its side effect: registers the ``BidsEmg`` event type this
# study emits. Same reader, same BIDS conventions as emg2qwerty.
from neuralfetch.studies.sivakumar2024emg2qwerty import BidsEmg  # noqa: F401
from neuralset.events import study

LOGGER = logging.getLogger(__name__)


class Salter2024Emg2pose(study.Study):
    """emg2pose (Meta Reality Labs, NeurIPS 2024) -- surface-EMG hand pose.

    Notes
    -----
    The paper's train/val/test and generalization assignments are read from
    ``sourcedata/emg2pose_metadata.csv``, which the release ships. Without it
    the split columns are absent.

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
    #: Columns of ``emg2pose_metadata.csv`` that the BIDS conversion drops.
    SPLIT_COLUMNS: tp.ClassVar[tuple[str, ...]] = (
        "split",
        "generalization",
        "moving_hand",
        "held_out_user",
        "held_out_stage",
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

        Read from ``sourcedata/emg2pose_metadata.csv``, which the release
        ships: ``split`` and ``generalization`` are not BIDS entities and so
        do not survive the conversion. Returns an empty mapping (and warns
        once) when the table is absent, so a BIDS-only tree still loads and
        ``PredefinedSplit`` fails at split time instead.
        """
        if self._split_metadata_cache is not None:
            return self._split_metadata_cache
        path = self.bids_root / "sourcedata" / "emg2pose_metadata.csv"
        try:
            table = pd.read_csv(path)
            splits = {
                str(row["filename"]): {
                    column: row[column] for column in self.SPLIT_COLUMNS
                }
                for _, row in table.iterrows()
            }
        except (OSError, ValueError, KeyError):
            LOGGER.warning(
                "No usable %s; the paper's split and generalization columns "
                "will be absent.",
                path,
            )
            splits = {}
        self._split_metadata_cache = splits
        return self._split_metadata_cache

    def _splits_for(self, source_file: tp.Any) -> dict[str, tp.Any]:
        """Look up the split row for a recording's upstream HDF5 name."""
        splits = self.recording_splits
        if not splits or source_file is None or pd.isna(source_file):
            return {}
        # ``metadata.csv`` keys on the stem; the sidecar keeps the ``.hdf5``.
        return splits.get(Path(str(source_file)).stem, {})

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
                # Not "duration": neuralset merges timeline keys into the
                # events frame, where "duration" is the event's own span and
                # would collide. This is the recording's un-padded length,
                # used only to bound those spans.
                "valid_duration": self._scan_durations(bids_path.fpath.parent.parent).get(
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

        NM000281 marks IK failures with ``BAD_IK`` annotations -- its README:
        "BAD_IK annotations mark samples where inverse-kinematics labels are
        all zero". Those annotations are the only usable source: measured over
        60 recordings, upstream's all-zero test on the BDF joint samples agrees
        with them only 87% of the time on average and as little as 32%, because
        BDF quantization does not preserve the exact zeros it relies on.

        Spans are clipped to *valid_duration*: the BDF writer pads the final
        data record with edge values, so the file runs past the data, and a
        window over that tail would train on constant joint angles.
        """
        if valid_duration is None:
            raise ValueError(
                f"No un-padded duration for {filepath}: its session's scans.tsv "
                "has no usable 'duration' entry, and the BDF is padded with edge "
                "values, so the recording cannot be safely bounded."
            )
        annotations = mne.io.read_raw_bdf(
            filepath, preload=False, verbose="ERROR"
        ).annotations
        bad = sorted(
            (float(onset), float(onset) + float(duration))
            for onset, duration, description in zip(
                annotations.onset,
                annotations.duration,
                annotations.description,
                strict=True,
            )
            if str(description) == self.IK_ANNOTATION
        )

        spans: list[tuple[float, float]] = []
        cursor = 0.0
        for start, stop in [*bad, (valid_duration, valid_duration)]:
            if start > cursor:
                spans.append((cursor, min(start, valid_duration) - cursor))
            cursor = max(cursor, stop)
            if cursor >= valid_duration:
                break
        return [(start, duration) for start, duration in spans if duration > 0]

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
            # Only per-event fields belong here. neuralset copies every other
            # timeline key onto the events frame itself, and a column set in
            # both places warns today and raises tomorrow.
            "subject": timeline["subject"],
        }

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

        One span per contiguous run of resolved IK frames.
        Spans shorter than a window are not filtered here -- the study does
        not know the window length, and ``stride_drop_incomplete`` drops them
        at segmentation time.
        """
        valid = timeline.get("valid_duration")
        if valid is None:
            raise ValueError(
                f"No un-padded duration for {filepath}; refusing to emit an event "
                "that would extend into the BDF's edge-value padding."
            )
        return list(self._ik_clean_spans(filepath, valid))
