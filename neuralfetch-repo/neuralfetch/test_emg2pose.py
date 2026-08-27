"""Tests for the span helpers in the ``emg2pose`` study.

Study loading itself is covered by the automated ``StudyInfo`` check in
``test_studies.py``.
"""

from __future__ import annotations

import pytest

from neuralfetch.studies.emg2pose import _complement


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
