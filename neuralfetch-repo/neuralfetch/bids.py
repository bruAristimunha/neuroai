# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""BIDS-backed event types shared by NeuralFetch studies."""

import typing as tp

import mne_bids

from neuralset.events import etypes


class BidsEmg(etypes.Emg):
    """EMG event read from a BIDS recording."""

    def _read(self) -> tp.Any:
        return mne_bids.read_raw_bids(
            mne_bids.get_bids_path_from_fname(self.filepath), verbose=False
        )
