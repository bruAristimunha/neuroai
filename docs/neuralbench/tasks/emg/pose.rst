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

   # Download the NM000281 release
   neuralbench emg pose -m neuropose --download

   # Full paper configuration
   neuralbench emg pose -m neuropose

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/emg/pose/config.yaml
      :language: yaml

Description
~~~~~~~~~~~

Hand-pose regression from 16-channel surface EMG against the 20 joint angles
of the UmeTrack hand skeleton [Salter2024]_: 25,253 recordings over 193
participants, 370 hours and 29 movement stages, with 2 kHz sEMG paired with
tracked joint angles. Each 5-s window is mapped to the 20-joint trajectory
in degrees, the unit used for the paper's reported angular error.

This is the paper's **regression** setting (``regression_neuropose``), a plain
sequence-to-sequence map.  Its **tracking** setting is not implemented: that
one feeds in the initial pose and conditions on the previous state at each
step, which is a model-side change rather than a configuration one.

Dataset Notes
~~~~~~~~~~~~~

* **IK failures and padding**: ``BAD_IK`` BDF annotations split a recording
  into resolved-label spans, and ``scans.tsv`` clips the padded BDF tail. A
  recording without an unpadded duration is rejected.
* **Splits**: the paper's ``split`` and ``generalization`` assignments are not
  BIDS entities.  They are joined from the upstream ``emg2pose_metadata.csv``
  on the HDF5 name each sidecar preserves as ``SourceFile``. The required
  ``sourcedata/emg2pose_metadata.csv`` must be materialized locally; there is
  no metadata fallback.

.. warning::

   emg2pose is released under CC-BY-NC-SA-4.0, and the UmeTrack hand model
   used for forward kinematics under CC-BY-NC-4.0.  Both are
   **non-commercial**.

References
~~~~~~~~~~

.. [Salter2024] Salter, Sasha, et al. "emg2pose: A large and diverse benchmark
   for surface electromyographic hand pose estimation." *Advances in Neural
   Information Processing Systems* 37 (2024).  arXiv:2412.02725.
