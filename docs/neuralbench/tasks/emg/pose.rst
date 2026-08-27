Hand pose decoding
==================

| **Name**: pose
| **Category**: motor / hand-pose decoding
| **Dataset**: :py:class:`~neuralset.studies.Emg2pose` (NM000281, emg2pose)
| **Objective**: :bdg-dark:`20-joint angle regression`
| **Split**: Leave-users-out (cross-user)
| **Upstream**: `paper <https://arxiv.org/abs/2412.02725>`_, `code <https://github.com/facebookresearch/emg2pose>`_, `blog <https://ai.meta.com/blog/open-sourcing-surface-electromyography-datasets-neurips-2024/>`_

.. image:: https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_overview.png
   :alt: emg2pose overview: sEMG wristband recordings paired with motion-capture hand pose
   :width: 75%
   :align: center

Usage
~~~~~

.. code-block:: bash

   # Auto-fetch NM000281 via eegdash
   neuralbench emg pose -m eegnet --download

   # Local 2-epoch sanity check
   neuralbench emg pose -m eegnet --debug

   # Full benchmark run
   neuralbench emg pose -m eegnet

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

Each 5-s window is mapped to the joint-angle vector at the window's right
edge, so the readout is causal: pose at time *t* is predicted from the EMG
that precedes it.  Targets are radians; the default metrics are RMSE, MAE,
Pearson *r*, R², and normalized RMSE over the 20 outputs.

As compared to the original paper, the NeuralBench default configuration
predicts a single end-of-window pose rather than rolling a state-space
tracker across the whole session, and splits users at random rather than
reusing the paper's fixed held-out-user assignment, keeping turn-around
tractable.  The paper's ``BAD_IK`` masking and per-user aggregation are not
applied, so absolute errors are not directly comparable to the published
tracking numbers.

The upstream repository ships three reference experiments --
``tracking_vemg2pose``, ``regression_vemg2pose``, and
``regression_neuropose`` -- along with pre-trained checkpoints for vemg2pose
(tracking and regression) and NeuroPose (regression).  None of them are
ported here yet; ``eegnet`` and the constant baselines are what the task runs
against today.

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
* **Windowing**: 5-s windows with a 5-s stride, so windows tile each
  recording without overlap; incomplete trailing windows are dropped.
  Note that each BDF is zero-padded to a whole second (verified:
  ``ValidSamples + BDFPaddedSamples`` is always a multiple of 2000), so the
  final retained window of a recording can end on up to ~1 s of padding.
  The task does not yet crop to ``ValidSamples``; doing so is a small,
  worthwhile follow-up.

Paper splits
~~~~~~~~~~~~

The upstream ``metadata.csv`` -- one row per HDF5 recording -- carries the
fields below.  Only ``user``, ``stage``, ``side``, and the source file name
survive into the BIDS conversion; the split and generalization assignments do
not.

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Column
     - Description
     - In NM000281?
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

.. note::

   ``metadata.csv`` is published standalone (~5 MiB) at
   ``https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_metadata.csv``
   and is keyed by the HDF5 file name, which NM000281 preserves as
   ``source_file`` in ``scans.tsv`` and ``SourceFile`` in each sidecar.
   Joining on that column is therefore enough to recover the paper's exact
   splits and the three generalization conditions -- the intended follow-up
   to the random cross-user split shipped here.

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
