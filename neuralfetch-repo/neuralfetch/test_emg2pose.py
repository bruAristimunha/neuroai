"""Tests for the utility functions defined in the ``emg2pose`` study.

Study loading itself is covered by the automated ``StudyInfo`` check in
``test_studies.py``; only the IK-span helpers are tested here.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuralfetch.studies.emg2pose import (
    _complement,
    contiguous_spans,
    ik_failure_mask,
)


def test_ik_failure_mask_marks_all_zero_frames():
    """An IK failure is an all-zero joint vector, as upstream defines it."""
    joints = np.zeros((20, 6))
    joints[:, 2:4] = 0.5
    joints[3, 5] = 1e-3  # one non-zero joint is enough to count as resolved

    assert ik_failure_mask(joints).tolist() == [True, True, False, False, True, False]


def test_ik_failure_mask_keeps_small_angles():
    """Small but real angles in radians are not mistaken for the zero marker."""
    joints = np.full((20, 3), 1e-4)

    assert not ik_failure_mask(joints).any()


@pytest.mark.parametrize(
    "mask, expected",
    [
        ([0, 0, 0], []),
        ([1, 1, 1], [(0, 3)]),
        ([0, 1, 1, 0, 1], [(1, 3), (4, 5)]),
        ([1, 0, 0, 1], [(0, 1), (3, 4)]),
    ],
)
def test_contiguous_spans(mask, expected):
    """Spans are half-open and cover every run of True."""
    assert contiguous_spans(np.array(mask, dtype=bool)) == expected


@pytest.mark.parametrize(
    "bad, limit, expected",
    [
        ([], 10.0, [(0.0, 10.0)]),
        ([(2.0, 3.0)], 10.0, [(0.0, 2.0), (3.0, 7.0)]),
        ([(0.0, 10.0)], 10.0, []),
        ([(0.0, 2.0)], 10.0, [(2.0, 8.0)]),
        # A BAD_IK span running into the padded tail is clipped at the limit.
        ([(2.0, 4.0)], 3.0, [(0.0, 2.0)]),
    ],
)
def test_complement_of_bad_spans(bad, limit, expected):
    """Clean spans are the complement of BAD_IK, clipped to the valid region."""
    assert _complement(bad, limit) == expected
