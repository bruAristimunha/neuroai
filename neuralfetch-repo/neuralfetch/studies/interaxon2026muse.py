"""Muse sleep-onset EEG, NEMAR nm000287."""

import typing as tp
from pathlib import Path

import pandas as pd
from mne_bids import BIDSPath, read_raw_bids

from neuralfetch import download
from neuralset.events import study


class Interaxon2026Muse(study.Study):
    """At-home Muse S family EEG with first-N2 point annotations.

    Version 1.0.0 contains 540 recordings from 203 participants: 500 training
    sessions and 40 test sessions, all from participants also in training.
    Session-table splits are preserved, not inferred from participant IDs.

    Signals have four channels (TP9, AF7, AF8, TP10) at 128 Hz. MNE-BIDS
    applies BrainVision unit scaling; the loader does not filter or resample.
    Full hypnograms, demographic information, acquisition dates, the N2 scoring
    method and prior filtering history are not provided.

    Every recording ends 300 seconds after N2. Duration, annotations and
    session quality summaries must remain outside model inputs. These files
    alone do not enforce causal evaluation or define the sealed test cohort.

    Downloads use the existing NEMAR backend, pinned to version 1.0.0. A manually
    supplied BIDS tree directly under the study directory remains supported.
    """

    licence: tp.ClassVar[str] = "CC-BY-NC-SA-4.0"
    url: tp.ClassVar[str] = "https://doi.org/10.82901/nemar.nm000287"
    aliases: tp.ClassVar[tuple[str, ...]] = ("muse", "nm000287")
    bibtex: tp.ClassVar[str] = """
    @misc{muse2026sleeponset,
        author = {{Muse Team}},
        title = {Muse Sleep-Onset EEG — EEG/EMG Foundation Challenge 2026, Track 03},
        year = {2026},
        publisher = {NEMAR},
        version = {1.0.0},
        doi = {10.82901/nemar.nm000287},
        url = {https://doi.org/10.82901/nemar.nm000287},
    }
    """
    description: tp.ClassVar[str] = (
        "Muse Team: 540 at-home recordings from 203 participants; Muse S family "
        "EEG, four channels at 128 Hz, with first-N2 point annotations. "
        "500 train / 40 seen-participant test sessions; no full hypnograms."
    )
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=540,
        num_subjects=203,
        num_events_in_query=2,
        event_types_in_query={"Eeg", "SleepStage"},
        data_shape=(4, 122880),
        frequency=128,
    )

    def _download(self, overwrite: bool = False) -> None:
        download.Nemar(
            study="nm000287",
            dset_dir=self.path,
            version="1.0.0",
        ).download(overwrite=overwrite)

    @property
    def bids_root(self) -> Path:
        if any(self.path.glob("sub-*/sub-*_sessions.tsv")):
            return self.path
        return self.path / "download" / "nm000287"

    def iter_timelines(self):
        files = sorted(self.bids_root.glob("sub-*/sub-*_sessions.tsv"))
        if not files:
            raise FileNotFoundError(
                f"No BIDS session tables in {self.bids_root}; run study.download() first"
            )
        for path in files:
            for row in pd.read_csv(path, sep="\t").itertuples():
                if row.split not in {"train", "test"}:
                    raise ValueError(f"Unknown split in {path}: {row.split}")
                yield dict(subject=path.parent.name, session=row.session_id)

    def _bids_path(self, timeline):
        return BIDSPath(
            root=self.bids_root,
            subject=timeline["subject"].removeprefix("sub-"),
            session=timeline["session"].removeprefix("ses-"),
            task="sleeponset",
            datatype="eeg",
            suffix="eeg",
            extension=".vhdr",
        )

    def _load_raw(self, timeline):
        raw = read_raw_bids(self._bids_path(timeline), verbose="ERROR")
        if raw.ch_names != ["TP9", "AF7", "AF8", "TP10"] or raw.info["sfreq"] != 128:
            raise ValueError("Unexpected channel order or sampling frequency")
        return raw

    def _load_timeline_events(self, timeline):
        header = self._bids_path(timeline).fpath
        annotations = pd.read_csv(
            header.with_name(header.name.replace("_eeg.vhdr", "_events.tsv")),
            sep="\t",
        )
        onset = annotations.loc[annotations.trial_type == "n2_onset", "onset"]
        if len(onset) != 1:
            raise ValueError(f"Expected exactly one N2 onset: {header}")
        sessions = pd.read_csv(
            self.bids_root / timeline["subject"] / f"{timeline['subject']}_sessions.tsv",
            sep="\t",
        ).set_index("session_id")
        split = sessions.loc[timeline["session"], "split"]
        raw = self._load_raw(timeline)
        if not 0 <= onset.iloc[0] <= raw.n_times / 128:
            raise ValueError(f"N2 onset outside recording: {header}")
        return pd.DataFrame(
            [
                dict(
                    type="Eeg",
                    start=0.0,
                    duration=raw.n_times / 128,
                    filepath=study.SpecialLoader(
                        method=self._load_raw, timeline=timeline
                    ).to_json(),
                    split=split,
                ),
                dict(
                    type="SleepStage",
                    start=float(onset.iloc[0]),
                    duration=0.0,
                    stage="N2",
                    split=split,
                ),
            ]
        )
