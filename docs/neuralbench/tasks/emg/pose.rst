Hand pose decoding
==================

| **Name**: pose
| **Category**: motor / hand-pose decoding
| **Dataset**: :py:class:`~neuralset.studies.Emg2pose` (NM000281, emg2pose)
| **Objective**: :bdg-dark:`20-joint angle regression`
| **Split**: Leave-users-out (cross-user)

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

Hand-pose regression from 16-channel surface EMG (one wristband at 2 kHz)
against the 20 joint angles of the UmeTrack hand skeleton, following
[Salter2024]_.  Each 5-s window is mapped to the joint-angle vector at the
window's right edge, so the readout is causal: pose at time *t* is predicted
from the EMG that precedes it.  Targets are radians; the default metrics are
RMSE, MAE, Pearson *r*, R², and normalized RMSE over the 20 outputs.

As compared to the original paper, the NeuralBench default configuration
predicts a single end-of-window pose rather than rolling a state-space
tracker across the whole session, and splits users at random rather than
reusing the paper's fixed held-out-user assignment, keeping turn-around
tractable.  The paper's ``BAD_IK`` masking and per-user aggregation are not
applied, so absolute errors are not directly comparable to the published
tracking numbers.

Dataset Notes
~~~~~~~~~~~~~

* **Auto-fetch**: ``--download`` pulls NM000281 from NEMAR via
  :py:class:`neuralfetch.download.Eegdash`, under
  ``<DATA_DIR>/Emg2pose/download/nm000281/sub-*/...``.  Eegdash owns NEMAR
  metadata and transfer through its own ``nemar-py`` dependency, so
  NeuralFetch neither imports nor pins ``nemar``.  Users with an existing
  BIDS copy should symlink it into ``download/nm000281/``.
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
* **Splits and metadata**: ``participants.tsv`` and the recording sidecars
  supply the source user, movement stage, hand side, source file, and valid
  sample count.  The task splits on ``user``, and ``stage`` / ``side`` are
  carried in ``summary_columns`` so per-stage and per-hand breakdowns are
  available at aggregation time.
* **Windowing**: 5-s windows with a 5-s stride, so windows tile each
  recording without overlap; incomplete trailing windows are dropped.

References
~~~~~~~~~~

.. [Salter2024] Salter, Sasha, et al. "emg2pose: A large and diverse
   benchmark for surface electromyographic hand pose estimation."
   *Advances in Neural Information Processing Systems* 37 (2024).
   arXiv:2412.02725.
