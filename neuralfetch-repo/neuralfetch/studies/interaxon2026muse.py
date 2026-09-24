"""Local Muse sleep-onset BIDS release; no download or signal modification."""

import typing as tp

import mne
import pandas as pd

from neuralset.events import study


class Interaxon2026Muse(study.Study):
    licence: tp.ClassVar[str] = "CC-BY-NC-SA-4.0"
    url: tp.ClassVar[str] = "https://neural-interfaces26.github.io/tracks.html"
    description: tp.ClassVar[str] = (
        "Muse Team: collective at-home Muse S family EEG, four channels at "
        "128 Hz, with first-N2 onset annotations (not full sleep staging)."
    )

    def iter_timelines(self):
        files = sorted(self.path.glob("sub-*/sub-*_sessions.tsv"))
        if not files:
            raise FileNotFoundError(f"No BIDS session tables in {self.path}")
        for path in files:
            for row in pd.read_csv(path, sep="\t").itertuples():
                if row.split not in {"train", "test"}:
                    raise ValueError(f"Unknown split in {path}: {row.split}")
                yield dict(subject=path.parent.name, session=row.session_id)

    def _header(self, timeline):
        subject, session = timeline["subject"], timeline["session"]
        return (
            self.path
            / subject
            / session
            / "eeg"
            / (f"{subject}_{session}_task-sleeponset_eeg.vhdr")
        )

    def _load_raw(self, timeline):
        raw = mne.io.read_raw_brainvision(self._header(timeline), verbose="ERROR")
        if raw.ch_names != ["TP9", "AF7", "AF8", "TP10"] or raw.info["sfreq"] != 128:
            raise ValueError("Unexpected channel order or sampling frequency")
        return raw

    def _load_timeline_events(self, timeline):
        header = self._header(timeline)
        annotations = pd.read_csv(
            header.with_name(header.name.replace("_eeg.vhdr", "_events.tsv")),
            sep="\t",
        )
        onset = annotations.loc[annotations.trial_type == "n2_onset", "onset"]
        if len(onset) != 1:
            raise ValueError(f"Expected exactly one N2 onset: {header}")
        sessions = pd.read_csv(
            self.path / timeline["subject"] / f"{timeline['subject']}_sessions.tsv",
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
