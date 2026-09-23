# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from itertools import product
from pathlib import Path

import mne
import pandas as pd

from neuralfetch import download
from neuralset.events import study


class Xu2024Alljoined(study.Study):
    url: tp.ClassVar[str] = "https://osf.io/kqgs8"
    """Alljoined: EEG responses to static images for EEG-to-Image decoding.

    8 participants viewing static images from the NSD stimulus set, recorded
    with 64-channel EEG at 512 Hz. Predecessor to the larger Alljoined-1.6M
    dataset (Xu2025).

    Experimental Design:
        - EEG recordings (64-channel, standard 1020 montage, BDF format)
        - 8 participants, 2 sessions each
        - Image presentation duration: 300 ms
        - Paradigm: passive viewing of NSD natural images

    Notes:
        - Known broken/missing files: subj02 ses2, subj07 ses2, subj08 ses2.
    """

    aliases: tp.ClassVar[tuple[str, ...]] = ("Alljoined1",)

    bibtex: tp.ClassVar[str] = """
    @article{xu2024alljoined,
        title={Alljoined -- A dataset for {EEG}-to-Image decoding},
        author={Xu, Jonathan and Aristimunha, Bruno and Feucht, Max Emanuel and Qian, Emma
        and Liu, Charles and Shahjahan, Tazik and Spyra, Martyna and Zhang, Steven Zifan
        and Short, Nicholas and Kim, Jioh and others},
        journal={arXiv preprint arXiv:2404.05553},
        year={2024},
        doi={10.48550/arXiv.2404.05553},
        archiveprefix={arXiv},
        eprint={2404.05553}
    }

    @misc{xu2025alljoined1,
        title={Alljoined1},
        url={osf.io/kqgs8},
        publisher={OSF},
        author={Xu, Jonathan},
        year={2025},
        month={Sep}
    }
    """
    licence: tp.ClassVar[str] = "UNKNOWN"
    description: tp.ClassVar[str] = (
        "8 participants viewing static NSD images in 64-channel EEG at 512 Hz."
    )

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=13,
        num_subjects=8,
        num_events_in_query=3840,
        event_types_in_query={"Eeg", "Image"},
        data_shape=(64, 1778688),
        frequency=512.0,
    )

    def _download(self, overwrite: bool = False) -> None:
        download.Nemar(
            study="nm000133",
            dset_dir=self.path,
            version="1.0.4",
            # events name the NSD crops in stimuli/nsd/; the COCO originals are unused
            exclude=["code/**", "derivatives/**", "stimuli/train2017/**"],
        ).download(overwrite=overwrite)

    @staticmethod
    def _get_fname(
        path: str | Path,
        subject: str,
        session: int,
        kind: tp.Literal["raw", "events"] = "raw",
    ):
        sub, ses = f"sub-{int(subject):02}", f"ses-{session:02}"
        suffix = "eeg.bdf" if kind == "raw" else "events.tsv"
        folder = Path(path) / "download" / "nm000133" / sub / ses / "eeg"
        return folder / f"{sub}_{ses}_task-images_{suffix}"

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Returns a generator of all recordings.

        A timeline is only yielded when both the raw EEG (``.fif``) and the
        image-event (``.h5``) files exist. Some recordings ship the raw EEG
        without the accompanying event file (e.g. subj03 session1), and those
        would otherwise yield empty event frames downstream.
        """
        for subject, session in product(range(1, 9), range(1, 3)):
            raw_fname = self._get_fname(self.path, str(subject), session, kind="raw")
            h5_fname = self._get_fname(self.path, str(subject), session, "events")
            if raw_fname.exists() and h5_fname.exists():
                yield dict(subject=str(subject), session=session)

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.RawArray:
        tl = timeline
        # Necessary to ensure montage information is available in the Raw object
        filepath = str(
            self._get_fname(self.path, tl["subject"], tl["session"], kind="raw")
        )
        raw = mne.io.read_raw(filepath)
        raw.set_montage("standard_1020")
        return raw

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        """
        Broken/missing files:
        - subj02, session 2
        - subj07, session 2
        - subj08, session 2
        """
        tl = timeline
        # Load image event information
        h5_fname = self._get_fname(self.path, tl["subject"], tl["session"], "events")
        events = pd.read_csv(h5_fname, sep="\t")
        stimuli = Path(self.path).resolve() / "download" / "nm000133" / "stimuli"
        events["filepath"] = events["stim_file"].apply(lambda x: str(stimuli / x))
        events["start"] = events.onset
        events["duration"] = 0.3
        events["type"] = "Image"

        events = events.drop(columns=["onset"])

        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        eeg = {
            "type": "Eeg",
            "start": 0.0,
            "filepath": info,
        }
        events = pd.concat([pd.DataFrame([eeg]), events])  # type: ignore
        return events
