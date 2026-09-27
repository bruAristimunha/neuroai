# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Exercise the Muse loader without downloading participant data."""

import mne
import numpy as np
import pandas as pd
import pytest
from mne_bids import BIDSPath, write_raw_bids

from neuralfetch import download
from neuralfetch.studies.interaxon2026muse import Interaxon2026Muse
from neuralset.events.study import SpecialLoader


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "case",
    ["valid", "missing", "duplicate_onset", "duplicate_session", "split", "ambiguous"],
)
def test_muse_bids_loader(tmp_path, nested, case):
    study = Interaxon2026Muse(path=tmp_path)
    root = study.path / "download" / study.NEMAR_DATASET_ID if nested else study.path
    bids = BIDSPath(
        root=root, subject="001", session="001", task="sleeponset", datatype="eeg"
    )
    signal = (np.arange(4)[:, None] + 1) * 20e-6 * np.ones((1, 256))
    raw = mne.io.RawArray(
        signal, mne.create_info(["TP9", "AF7", "AF8", "TP10"], 128, "eeg")
    )
    onsets = (
        [] if case == "missing" else [0.5, 1.0] if case == "duplicate_onset" else [1.0]
    )
    raw.set_annotations(
        mne.Annotations(onsets, [0.0] * len(onsets), ["n2_onset"] * len(onsets))
    )
    write_raw_bids(raw, bids, format="BrainVision", allow_preload=True, verbose="ERROR")
    sessions = root / "sub-001" / "sub-001_sessions.tsv"
    pd.DataFrame(
        {"session_id": ["ses-001"], "split": ["unknown" if case == "split" else "test"]}
    ).to_csv(sessions, sep="\t", index=False)
    if case == "duplicate_session":
        table = pd.read_csv(sessions, sep="\t")
        pd.concat([table, table]).to_csv(sessions, sep="\t", index=False)
    if case == "ambiguous":
        other = study.path if nested else study.path / "download" / study.NEMAR_DATASET_ID
        (other / "sub-001").mkdir(parents=True, exist_ok=True)
        pd.read_csv(sessions, sep="\t").to_csv(
            other / "sub-001" / sessions.name, sep="\t", index=False
        )
    if case in {"split", "ambiguous", "duplicate_session"}:
        with pytest.raises(
            ValueError, match="Unknown split|Ambiguous BIDS|Duplicate session"
        ):
            list(study.iter_timelines())
        return
    (timeline,) = study.iter_timelines()
    if case != "valid":
        with pytest.raises(ValueError):
            study._load_timeline_events(timeline)
        return
    events = study._load_timeline_events(timeline)
    assert events["split"].tolist() == ["test", "test"]
    assert events["start"].tolist() == [0.0, 1.0]
    assert events["duration"].tolist() == [2.0, 0.0]
    assert events.iloc[1]["stage"] == "N2"
    loaded = SpecialLoader.from_json(events.iloc[0]["filepath"]).load()
    assert loaded.ch_names == raw.ch_names
    np.testing.assert_allclose(loaded.get_data(), signal, rtol=1e-6)


def test_muse_download_refreshes_sessions(tmp_path, monkeypatch):
    study = Interaxon2026Muse(path=tmp_path)
    sessions = study.bids_root / "sub-001" / "sub-001_sessions.tsv"
    sessions.parent.mkdir(parents=True)
    table = pd.DataFrame({"session_id": ["ses-001"], "split": ["train"]})
    table.to_csv(sessions, sep="\t", index=False)
    assert study._sessions["sub-001"].loc["ses-001", "split"] == "train"

    def download_corrected_sessions(self, overwrite=False):
        table.assign(split="test").to_csv(sessions, sep="\t", index=False)

    monkeypatch.setattr(download.Nemar, "download", download_corrected_sessions)
    study.download(overwrite=True)
    assert study._sessions["sub-001"].loc["ses-001", "split"] == "test"
