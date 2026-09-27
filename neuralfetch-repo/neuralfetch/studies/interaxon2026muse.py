# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Muse sleep-onset EEG, NEMAR nm000287."""

import typing as tp
from functools import cached_property
from pathlib import Path

import pandas as pd
from mne_bids import BIDSPath, read_raw_bids

from neuralfetch import download
from neuralset.events import study


class Interaxon2026Muse(study.Study):
    """At-home Muse S family EEG with first-N2 point annotations.

    Version 1.0.0 contains 540 recordings from 203 participants, totaling
    approximately 157.52 hours. Four channels (TP9, AF7, AF8, TP10) are stored
    at 128 Hz in EEG-BIDS/BrainVision format.

    Notes
    -----
    - Session tables supply 500 train and 40 test recordings. Every test
      participant also appears in training; this is not the sealed competition
      cohort. The loader preserves these labels; benchmark split transforms
      may replace them in memory.
    - Each recording has one zero-duration first-N2 annotation, not stable N2
      or a full hypnogram. Onset is relative to recording start, not necessarily
      lights out. The scoring method is undocumented.
    - Every recording ends 300 seconds after N2. Total length, annotations,
      future EEG and whole-recording quality summaries must stay outside
      model inputs. Sequential batches alone do not make preprocessing causal.
    - MNE-BIDS applies header unit scaling. This loader does not filter,
      resample, clean artifacts or exclude recordings using quality flags.
    - Exact Muse S generation, firmware, reference and prior filters are
      unconfirmed. Downsampling from 256 Hz is a curator assumption, not a
      verified acquisition fact. Demographics and acquisition dates are absent.
    - Shared electrode coordinates have unknown provenance and should not be
      treated as participant-specific measurements. Quality flags describe
      signal screening, not independently confirmed artifacts.
    - The deposit records consent and sharing authorization, an internal Muse
      exemption determination, and destruction of the re-identification key.
      Credit Muse Team under CC-BY-NC-SA-4.0.
    - The NEMAR download is pinned to 1.0.0. An existing BIDS tree directly
      under the study directory is also supported.
    """

    NEMAR_DATASET_ID: tp.ClassVar[str] = "nm000287"
    licence: tp.ClassVar[str] = "CC-BY-NC-SA-4.0"
    url: tp.ClassVar[str] = "https://doi.org/10.82901/nemar.nm000287"
    aliases: tp.ClassVar[tuple[str, ...]] = ("muse", NEMAR_DATASET_ID)
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
            study=self.NEMAR_DATASET_ID,
            dset_dir=self.path,
            version="1.0.0",
        ).download(overwrite=overwrite)
        self.__dict__.pop("_sessions", None)

    @property
    def bids_root(self) -> Path:
        nested = self.path / "download" / self.NEMAR_DATASET_ID
        if any(self.path.glob("sub-*/sub-*_sessions.tsv")):
            if any(nested.glob("sub-*/sub-*_sessions.tsv")):
                raise ValueError(f"Ambiguous BIDS roots: {self.path} and {nested}")
            return self.path
        return nested

    @cached_property
    def _sessions(self) -> dict[str, pd.DataFrame]:
        files = sorted(self.bids_root.glob("sub-*/sub-*_sessions.tsv"))
        if not files:
            raise FileNotFoundError(
                f"No BIDS session tables in {self.bids_root}; run study.download() first"
            )
        sessions = {}
        for path in files:
            table = pd.read_csv(path, sep="\t").set_index("session_id")
            if not table.index.is_unique:
                raise ValueError(f"Duplicate session IDs in {path}")
            if not table["split"].isin(["train", "test"]).all():
                raise ValueError(f"Unknown split in {path}")
            sessions[path.parent.name] = table
        return sessions

    def iter_timelines(self):
        for subject, sessions in self._sessions.items():
            for session in sessions.index:
                yield dict(subject=subject, session=session)

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
        return read_raw_bids(self._bids_path(timeline), verbose="ERROR")

    def _load_timeline_events(self, timeline):
        raw = self._load_raw(timeline)
        onset = raw.annotations.onset[raw.annotations.description == "n2_onset"].item()
        duration = raw.n_times / raw.info["sfreq"]
        if not 0 <= onset <= duration:
            raise ValueError(f"N2 onset outside recording: {timeline}")
        split = self._sessions[timeline["subject"]].loc[timeline["session"], "split"]
        return pd.DataFrame(
            [
                dict(
                    type="Eeg",
                    start=0.0,
                    duration=duration,
                    filepath=study.SpecialLoader(
                        method=self._load_raw, timeline=timeline
                    ).to_json(),
                ),
                dict(
                    type="SleepStage",
                    start=onset,
                    duration=0.0,
                    stage="N2",
                ),
            ]
        ).assign(split=split)
