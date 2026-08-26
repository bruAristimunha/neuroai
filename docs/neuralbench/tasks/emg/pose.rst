EMG2Pose BIDS study
===================

| **Dataset**: :py:class:`~neuralset.studies.Emg2pose` (NEMAR NM000281)
| **Modality**: 16-channel surface EMG with 20 joint-angle targets
| **Status**: dataset integration; no default NeuralBench task yet

EMG2Pose is the NeurIPS 2024 hand-pose benchmark from Salter, Warren,
Schlager et al. NEMAR NM000281 is its EMG-BIDS conversion. Each BDF recording
contains ``emg0`` through ``emg15`` and ``joint0`` through ``joint19`` at
2 kHz. Every recording supplies a BIDS ``channels.tsv`` sidecar: 16 ``EMG``
channels in volts and 20 ``MISC`` joint-angle channels in radians. MNE-BIDS
uses that metadata directly without study-specific numerical rescaling. The
BDF encoder's physical header is ``uV`` for all 36 channels: EMG consequently
arrives from MNE in volts, while the joint targets require an explicit
``×1e6`` conversion in a paper-specific task to recover radians. That
conversion is deliberately not hidden in the generic study reader.

Why no default task configuration?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The BIDS release exposes the source user, movement stage, hand side, source
file, and valid sample count through ``participants.tsv`` and recording
sidecars. The paper's exact train/validation/test and held-out
user/stage/user-stage assignments remain defined by upstream ``metadata.csv``;
they are not derivable exactly from those BIDS fields. A generic random split,
plain MSE loss, or guessed target window would therefore be misleading.

The future official task must therefore consume a public structured split
artifact and implement the paper's valid-sample and ``BAD_IK`` handling,
5-second windows, tracking/regression rollout, paper-specific input
preprocessing (including the BDF joint-angle scale), and per-user aggregation.
Until then, :class:`~neuralset.studies.Emg2pose` is available for verified
BIDS loading and replication preparation, but ``neuralbench emg pose`` is not
advertised as a runnable benchmark.

Dataset access
~~~~~~~~~~~~~~

The study uses :py:class:`neuralfetch.download.Eegdash`; Eegdash owns NEMAR
metadata and transfer support through its own dependency on ``nemar-py``.
NeuralFetch does not import ``nemar`` directly and therefore does not pin it.
An existing Iceberg BIDS copy can be linked under
``download/nm000281/`` below the study directory.
