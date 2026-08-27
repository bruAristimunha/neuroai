Hand pose decoding
==================

| **Name**: pose
| **Category**: motor / hand-pose decoding
| **Dataset**: :py:class:`~neuralset.studies.Emg2pose` (NM000281, emg2pose)
| **Objective**: :bdg-dark:`20-joint angle trajectory regression`
| **Split**: The paper's ``train`` / ``val`` / ``test`` assignment
| **Upstream**: `paper <https://arxiv.org/abs/2412.02725>`_, `code <https://github.com/facebookresearch/emg2pose>`_, `blog <https://ai.meta.com/blog/open-sourcing-surface-electromyography-datasets-neurips-2024/>`_

.. image:: https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_overview.png
   :alt: emg2pose overview: sEMG wristband recordings paired with motion-capture hand pose
   :width: 75%
   :align: center

Usage
~~~~~

.. code-block:: bash

   # Auto-fetch NM000281 via eegdash
   neuralbench emg pose -m neuropose --download

   # Local 2-epoch sanity check
   neuralbench emg pose -m neuropose --debug

   # Full benchmark run
   neuralbench emg pose -m neuropose

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/emg/pose/config.yaml
      :language: yaml

Description
~~~~~~~~~~~

Hand-pose regression from 16-channel surface EMG against the 20 joint angles
of the UmeTrack hand skeleton, following [Salter2024]_.  The upstream release
pairs 2 kHz sEMG from a wrist-worn band with time-aligned motion-capture
joint angles: 25,253 recordings over 193 participants, 370 hours, and 29
movement stages, one hand and one stage per recording, each roughly a minute
long.  Joint angles were solved offline by inverse kinematics from 19 markers
per hand tracked on a 26-camera OptiTrack rig, low-pass filtered at 15 Hz and
upsampled to 2 kHz; the sEMG was high-pass filtered at 40 Hz and rescaled so
its noise floor has unit standard deviation.

Each 5-s window (10,000 samples at 2 kHz, the paper's evaluation length) is
mapped to the joint-angle *trajectory* over that window -- 20 angles per
frame, not a single pose.  Targets are in **degrees**, the unit the paper
reports angular error in, so ``val/mae`` reads directly against its tables.
The default metrics are RMSE, MAE, Pearson *r*, R², and normalized RMSE.

This is the paper's **regression** setting.  Upstream
``PoseModule._predict_pose`` is ``pred = self.network(emg)`` -- no state
conditioning and no initial pose (``regression_neuropose.yaml`` sets
``provide_initial_pos: False``, ``predict_vel: False``) -- so a plain
sequence-to-sequence backbone is the right shape.  The default is
``NeuroPoseNet``, the paper's NeuroPose baseline.

The paper's **tracking** setting is not implemented: it feeds the initial
pose in and conditions on the previous state at each step
(``StatePoseModule``), which is a model-side change rather than a
configuration one.  Published tracking numbers are therefore not the
comparison point for this task; the regression baselines are.

The upstream repository ships three reference experiments --
``tracking_vemg2pose``, ``regression_vemg2pose``, and
``regression_neuropose`` -- along with pre-trained checkpoints for vemg2pose
(tracking and regression) and NeuroPose (regression).  ``regression_neuropose``
is the one this task mirrors.

Dataset Notes
~~~~~~~~~~~~~

* **Auto-fetch**: ``--download`` pulls NM000281 from NEMAR via
  :py:class:`neuralfetch.download.Eegdash`, under
  ``<DATA_DIR>/Emg2pose/download/nm000281/sub-*/...``.  Eegdash owns NEMAR
  metadata and transfer through its own ``nemar-py`` dependency, so
  NeuralFetch neither imports nor pins ``nemar``.  Users with an existing
  BIDS copy should symlink it into ``download/nm000281/``.  For reference,
  the upstream non-BIDS release is 431 GiB (plus a 600 MiB ``_mini`` tar for
  smoke tests).
* **BIDS-aware reader**: the Study reads via
  :py:func:`mne_bids.read_raw_bids` (``>= 0.19``); channel types and units
  come from the BIDS sidecars -- 16 ``EMG`` channels (``emg0``--``emg15``)
  and 20 ``MISC`` joint channels (``joint0``--``joint19``) -- so the study
  applies no numerical rescaling of its own.
* **Joint-angle units**: the BDF physical header declares ``uV`` for all 36
  channels, so EMG arrives from MNE in volts while the joint channels arrive
  1e6 times too small.  The task config restores radians with
  ``scale_factor: 1.0e+6`` on the target extractor.  This is a file-format
  correction and is kept in the task rather than hidden in the generic study
  reader.
* **Recording metadata**: ``participants.tsv`` maps each BIDS subject to its
  anonymized upstream ``original_user``; the per-session ``scans.tsv`` and
  the recording sidecars carry ``stage``, ``side``, ``source_file``,
  ``ValidSamples``, and the BDF zero-padding (``BDFPaddedSamples`` /
  ``BDFPaddedDuration``) applied to reach a whole number of BDF records.
  The task splits on ``user``, and ``stage`` / ``side`` ride along in
  ``summary_columns`` for per-stage and per-hand breakdowns at aggregation
  time.
* **IK failures**: the offline inverse-kinematics solver failed on 12.7% of
  frames, and the release marks those frames by writing an **all-zero joint
  vector** -- there is no separate annotation.  Upstream detects them with
  ``~np.all(np.isclose(joint_angles, 0), axis=-1)`` and, with
  ``skip_ik_failures`` (its datamodule default, for train and val/test
  alike), emits only windows lying inside contiguous resolved runs.  The
  study does the same: each recording contributes one event per resolved
  run, so no window straddles a failure.  Detection happens in radians, so a
  genuinely small angle is not mistaken for the marker.  Set
  ``Emg2pose(skip_ik_failures=False)`` to get one event per recording
  instead.
* **Windowing**: 5-s windows with a 5-s stride, so windows tile each
  event without overlap; runs shorter than one window are dropped.
* **Zero padding**: every BDF is zero-padded up to a whole number of
  one-second records, so the file runs past the data (``ValidSamples +
  BDFPaddedSamples`` is always a multiple of 2000).  The padding is all-zero,
  so the IK-failure test excludes it for free.  With
  ``skip_ik_failures=False`` the event is instead bounded at
  ``ValidSamples / SamplingFrequency``; that field is missing from 2,785 of
  the 25,253 sidecars, and for those the session's ``scans.tsv`` ``duration``
  column is used, which agrees exactly with
  ``ValidSamples / SamplingFrequency`` on all 22,468 recordings carrying
  both.

Paper splits
~~~~~~~~~~~~

The upstream ``metadata.csv`` -- one row per HDF5 recording -- carries the
fields below.  Only ``user``, ``stage``, ``side``, and the source file name
survive into the BIDS entities and sidecars; the split and generalization
assignments do not.  The study recovers them by joining ``metadata.csv`` on
that source file name, which every recording preserves as ``SourceFile``.

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Column
     - Description
     - In BIDS entities?
   * - ``user``
     - Anonymized user ID
     - yes (``participants.tsv``)
   * - ``session``
     - Recording session (several stages per session)
     - yes (BIDS ``ses-`` entity)
   * - ``stage``
     - Name of the movement stage
     - yes (``scans.tsv``, sidecar)
   * - ``side``
     - Hand side (``left`` / ``right``)
     - yes (``recording-`` entity)
   * - ``moving_hand``
     - Whether the hand is prompted to move
     - no
   * - ``held_out_user``
     - Whether the user is held out from training
     - no
   * - ``held_out_stage``
     - Whether the stage is held out from training
     - no
   * - ``split``
     - ``train`` / ``val`` / ``test``
     - no
   * - ``generalization``
     - ``user``, ``stage``, or ``user_stage``
     - no

``metadata.csv`` ships inside the release at
``sourcedata/emg2pose_metadata.csv`` and is also published standalone (~5 MiB)
at ``https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_metadata.csv``;
``--download`` fetches the standalone copy when the release's own is absent,
and ``Emg2pose(split_metadata=...)`` overrides both.  The join was checked
against the full release: **all 25,253 recordings match a metadata row**, and
the resulting counts reproduce the published totals exactly --
17,136 ``train`` / 1,950 ``val`` / 6,167 ``test``, with the test rows
splitting 3,790 ``user`` / 3,539 ``stage`` / 788 ``user_stage``.

``PredefinedSplit`` consumes the ``split`` column directly with
``valid_split_by: null``, so the paper's validation recordings are kept rather
than a fresh validation set being drawn out of train.  ``generalization`` is
carried in ``summary_columns``, so per-condition numbers fall out of the
aggregation.

.. note::

   When no metadata table can be found, the study logs a warning and simply
   omits the split columns -- a BIDS-only download stays usable for
   exploration, but ``neuralbench emg pose`` will fail at split time, because
   there is no defensible split to fall back on.

.. warning::

   The all-zero IK-failure test is implemented from the upstream definition
   and unit-tested, but has **not** been checked against a real NM000281 BDF
   -- confirming that the failure frames survive the BIDS conversion as
   exactly-zero joint channels needs a recording pulled on the cluster.

.. warning::

   emg2pose is released under CC-BY-NC-SA-4.0, and the UmeTrack hand model
   used for forward kinematics under CC-BY-NC-4.0.  Both are
   **non-commercial**.

References
~~~~~~~~~~

.. [Salter2024] Salter, Sasha, Richard Warren, Collin Schlager, Adrian Spurr,
   Shangchen Han, Rohin Bhasin, Yujun Cai, Peter Walkington, Anuoluwapo
   Bolarinwa, Robert Wang, et al. "emg2pose: A large and diverse benchmark
   for surface electromyographic hand pose estimation." *Advances in Neural
   Information Processing Systems* 37, Datasets and Benchmarks Track (2024).
   arXiv:2412.02725.
