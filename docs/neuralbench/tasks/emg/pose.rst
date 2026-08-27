Hand pose decoding
==================

| **Name**: pose
| **Category**: motor / hand-pose decoding
| **Dataset**: :py:class:`~neuralset.studies.Salter2024Emg2pose` (NM000281, emg2pose)
| **Objective**: :bdg-dark:`20-joint angle trajectory regression`
| **Split**: The paper's ``train`` / ``val`` / ``test`` assignment

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
of the UmeTrack hand skeleton [Salter2024]_: 25,253 recordings over 193
participants, 370 hours and 29 movement stages, with 2 kHz sEMG paired with
motion-capture joint angles.  Each 5-s window is mapped to the joint-angle
trajectory over that window, in **degrees** -- the unit the paper reports
angular error in, so ``val/mae`` reads directly against its tables.

This is the paper's **regression** setting (``regression_neuropose``), a plain
sequence-to-sequence map.  Its **tracking** setting is not implemented: that
one feeds in the initial pose and conditions on the previous state at each
step, which is a model-side change rather than a configuration one.

Dataset Notes
~~~~~~~~~~~~~

* **Joint-angle units**: the BDF physical header declares ``uV`` for all 36
  channels, so MNE returns the 20 ``MISC`` joint channels 1e6 times too
  small.  The target's ``scale_factor`` restores radians and converts to
  degrees in one step.
* **IK failures**: the inverse-kinematics solver failed on 12.7% of frames.
  NM000281 marks them with ``BAD_IK`` annotations, and the study emits one
  event per contiguous resolved run so no window straddles a failure --
  matching upstream's ``skip_ik_failures``, its datamodule default.  The
  annotations are the only usable source: over 60 sampled recordings,
  upstream's all-zero test on the BDF joint samples agreed with them 87% of
  the time on average and as little as 32%, usually reporting no failures at
  all where the annotations mark 15% of the recording, because BDF
  quantization does not preserve the exact zeros that test relies on.  Pass
  ``skip_ik_failures=False`` for one event per recording instead.
* **Padding**: the BDF writer pads the final data record with edge values, so
  every span is clipped to the ``scans.tsv`` duration.  A recording whose
  session has no usable ``duration`` entry is rejected rather than bounded at
  the padded file length, which would feed constant joint angles to the loss
  as ground truth.
* **Splits**: the paper's ``split`` and ``generalization`` assignments are not
  BIDS entities.  They are joined from the upstream ``emg2pose_metadata.csv``
  on the HDF5 name each sidecar preserves as ``SourceFile``.  The table ships
  in the release under ``sourcedata/``; ``--download`` otherwise fetches the
  standalone copy.  Without it the split columns are absent and the task
  fails at split time rather than inventing a split.
* **Sampling rate**: ``NeuroPoseNet`` [Liu2021]_ decimates 2 kHz to its
  ``internal_sfreq`` and applies NeuroPose's original pooling schedule, where
  emg2pose instead keeps 2 kHz and widens that schedule (Section 3.5).  The
  receptive fields are comparable, but this route sees no EMG content above
  100 Hz.

.. warning::

   emg2pose is released under CC-BY-NC-SA-4.0, and the UmeTrack hand model
   used for forward kinematics under CC-BY-NC-4.0.  Both are
   **non-commercial**.

References
~~~~~~~~~~

.. [Liu2021] Liu, Yilin, Sai Zhang, and Mahanth Gowda. "NeuroPose: 3D hand
   pose tracking using EMG wearables." *Proceedings of the Web Conference
   2021*, 1471--1482.

.. [Salter2024] Salter, Sasha, et al. "emg2pose: A large and diverse benchmark
   for surface electromyographic hand pose estimation." *Advances in Neural
   Information Processing Systems* 37 (2024).  arXiv:2412.02725.
