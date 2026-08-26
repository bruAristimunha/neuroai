EMG2Pose BIDS study
===================

| **Dataset**: :py:class:`~neuralset.studies.Emg2pose` (NEMAR NM000281)
| **Modality**: 16-channel surface EMG with 20 joint-angle targets
| **Status**: dataset integration; no default NeuralBench task yet

EMG2Pose is the NeurIPS 2024 hand-pose benchmark from Salter, Warren,
Schlager et al. NEMAR NM000281 is its EMG-BIDS conversion. Each BDF recording
contains ``emg0`` through ``emg15`` and ``joint0`` through ``joint19`` at
2 kHz. The study maps those channel names explicitly because the converted
recordings need not provide a ``channels.tsv`` sidecar. It restores the EMG
from MNE's volts to the upstream normalized scale; joint channels are already
in radians when read by MNE.

Why no default task configuration?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The paper's train/validation/test and user/stage/user-stage generalization
conditions come from the upstream ``metadata.csv``. BIDS entities and
``events.tsv`` do not encode those splits. That file is currently outside the
BIDS-only acquisition boundary, so a generic random split, plain MSE loss, or
guessed target window would be misleading.

The future official task must therefore consume a public structured split
artifact and implement the paper's valid-sample and ``BAD_IK`` handling,
5-second windows, tracking/regression rollout, and per-user aggregation.
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
