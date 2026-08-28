"""Tests for the span helpers in the ``salter2024emg2pose`` study.

Study loading itself is covered by the automated ``StudyInfo`` check in
``test_studies.py``.
"""

from __future__ import annotations

import pytest

from neuralfetch.studies.salter2024emg2pose import _complement, _has_split_columns


@pytest.mark.parametrize(
    "bad, limit, expected",
    [
        ([], 10.0, [(0.0, 10.0)]),
        ([(2.0, 3.0)], 10.0, [(0.0, 2.0), (3.0, 7.0)]),
        ([(0.0, 10.0)], 10.0, []),
        ([(0.0, 2.0)], 10.0, [(2.0, 8.0)]),
        ([(1.0, 2.0), (4.0, 5.0)], 10.0, [(0.0, 1.0), (2.0, 2.0), (5.0, 5.0)]),
        # Overlapping annotations merge rather than producing a negative span.
        ([(1.0, 5.0), (2.0, 3.0)], 10.0, [(0.0, 1.0), (5.0, 5.0)]),
        # A BAD_IK span running into the padded tail is clipped at the limit.
        ([(2.0, 4.0)], 3.0, [(0.0, 2.0)]),
    ],
)
def test_complement_of_bad_spans(bad, limit, expected):
    """Clean spans are the complement of BAD_IK, clipped to the valid region."""
    assert _complement(bad, limit) == expected


def test_annex_pointer_is_not_mistaken_for_the_metadata_table(tmp_path):
    """An unfetched git-annex pointer must not pass as the split table.

    NEMAR ships NM000281 through git-annex; a tree cloned without its annexed
    content leaves ``sourcedata/emg2pose_metadata.csv`` holding a single
    ``/annex/objects/...`` line.
    """
    pointer = tmp_path / "emg2pose_metadata.csv"
    pointer.write_text(
        "/annex/objects/SHA256E-s5674462--0e7ad04a79a0f162385d22ec8d6f43af.csv\n"
    )
    assert not _has_split_columns(pointer)

    real = tmp_path / "real.csv"
    real.write_text(
        "filename,split,generalization,moving_hand,held_out_user,held_out_stage\n"
        "2022-01-01-abc,train,user,left,False,False\n"
    )
    assert _has_split_columns(real)
