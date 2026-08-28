# NeuralBench EMG Pose Replication Design

## Purpose

Add the smallest safe NeuralBench extension that can run an **eval-only**
EMG2Pose/VEMG2Pose diagnostic against the sealed NEMAR `nm000281` subset.
The extension must preserve the released evaluator's trajectory, mask,
kinematics, split, and per-user aggregation semantics. It must never label
the resulting value as a paper reproduction or candidate-model parity.

This design also records the precise paper contracts for every EMG model in
the current queue. NeuroPose and SensingDynamics have no admissible dataset
or released implementation for their paper result, so their initial
NeuralBench deliverable is a strict compatibility report and shape validation,
not a substitute experiment on `nm000281`.

## Scope and non-goals

In scope:

- a dedicated `emg/pose` task namespace;
- an immutable-manifest reader for the 456 Test/User+Stage BDF recordings;
- exact official-checkpoint evaluation for `tracking_emg2pose.ckpt` and
  `tracking_vemg2pose.ckpt`;
- explicit trajectory/mask/initial-pose interfaces;
- per-user angle-error and landmark-distance reports; and
- paper dossiers and blocked-state tests for NeuroPose and SensingDynamics.

Out of scope:

- training any model, a generic pose-regression grid, or a normalized score;
- copying noncommercial upstream source, weights, UmeTrack assets, or data
  into NeuroAI;
- treating BDF data as identical to the private original HDF5 release;
- replacing the official evaluator; and
- any Braindecode model PR change, merge, or checkpoint conversion.

## Shared evidence boundary

`nm000281` is an internal NEMAR prerelease at commit
`d5a7508c09f0220a8699c440b0c5e60fee1b4dbf`, not a published immutable
dataset release. The selected split is 456 recordings from 20 held-out users
and six stages. It contains 54,063,381 valid samples (7.508802917 hours),
which differs from the paper's stated 7.0 hours. One Val/User/Unconstrained
record has BDF-to-HDF5 parity evidence; it does not establish equivalence for
all Test/User+Stage recordings.

Every artifact and CLI result must include these literal fields:

```text
evaluation_scope = "official checkpoint / transformed NEMAR prerelease"
paper_equivalence_established = false
candidate_model_parity_established = false
```

The only data path is the read-only Iceberg generation under
`paper_repro/emg2pose_nm000281_20260825_v1`. A user supplies its path, the
official runtime path, the checkpoint path, and the UmeTrack asset path at
execution time; none is downloaded, copied, or stored in the repository.

## Minimal architecture

```text
sealed manifest + read-only BDF directory
        |
        v
Nm000281PoseStudy ----> PoseWindowDataset ----> OfficialPoseAdapter
  fixed split/masks        (x, y, bad_ik, y0)       official checkpoint only
        |                                                |
        +-------------------- protocol metadata --------+
                                                         v
                                              OfficialPoseMetricsCallback
                                               AE deg + FK LD mm, grouped
                                               per user, SD ddof 0 and 1
                                                         |
                                                         v
                                                 immutable JSON/CSV evidence
```

The adapter is deliberately separate from NeuralBench's normal
`BrainModule`: its outputs are trajectories `(batch, 20, 11790)`, not
class logits or a fixed prediction vector. It delegates model inference,
`BAD_IK` handling, and UmeTrack landmark calculation to a pinned external
official evaluator. NeuralBench owns manifest validation, recording selection,
result serialization, and the comparison of its per-user rows with a provided
sealed baseline bundle.

## Paper protocol dossiers

### EMG2Pose — Salter et al. (2024)

| Item | Required contract |
| --- | --- |
| Input | 16-channel wrist sEMG at 2 kHz; one hand/session/stage recording |
| Labels | 20 UmeTrack joint angles on the same sample grid; invalid IK frames excluded |
| Tasks | Tracking: ground-truth initial pose; Regression: no initial pose |
| Cohort | 193 users: 158 train, 15 validation, 20 held-out test |
| Split | User, Stage, and User+Stage; never random windows |
| Metrics | Angle MAE in degrees and fingertip landmark distance in millimetres |
| Generic target | User+Stage AE `15.05 +/- 1.22` degrees; LD `20.69 +/- 1.39` mm |
| Result status | Not yet evaluated on the generic released checkpoint |

The first NeuralBench run therefore evaluates only the released generic
tracking checkpoint through the official evaluator. It must publish the
observed result, not infer the paper number from the VEMG run.

### VEMG2Pose — Salter et al. (2024)

| Item | Required contract |
| --- | --- |
| Core model | Causal 16-channel encoder plus state-conditioned decoder predicting velocity increments |
| Window | 11,790 effective samples: 1,790 left context and 10,000 supervised samples |
| Sampling | 2 kHz input; stride 10,000; no jitter during evaluation |
| Tracking state | Ground-truth pose at first supervised sample |
| Mask | Preserve `BAD_IK`; evaluate only official-valid frames |
| Aggregation | Mean within user, then mean and SD across 20 user means |
| Target | Test/User+Stage AE `11.0 +/- 1.0` degrees; LD `15.4 +/- 1.4` mm |

The current sealed reference is Margaret job 500932: AE
`11.037608559112806`, LD `15.375535790427591`, 3,562 windows, 450/456
contributing recordings. The adapter acceptance test compares all 20
per-user rows and aggregate values to that bundle. A match is still a
non-equivalent transformed-data diagnostic.

### Pos-MT — Hadidi et al. (2026), recipe-only extension

This paper uses the EMG2Pose corpus, but it is not the released VEMG2Pose
checkpoint comparator. Its causal encoder has convolutions `11/5` and `5/2`,
TDS subsampling stages `17/4` and `9/2`, and 64-dimensional features at
25 Hz. Features are interpolated to 50 Hz and concatenated with the prior
20-angle pose into a two-layer, 512-hidden-unit LSTM decoder. The decoder
head output scalar is part of the model definition: position tracking requires
`0.1`, position regression requires `1.0`; `0.01` can collapse to an almost
static solution. Regression velocity uses a 250-ms position warm-start and
multitask training weights are tracking `0.875` and regression `0.125`.

Training is AdamW (`weight_decay=0.01`, no LayerNorm decay), gradient clip
1.0, 10-epoch warmup `1e-8 -> 1e-3`, cosine decay to `1e-6` through epoch
150, and EMG rotation augmentation. The loss is joint L1 plus `0.01` times
valid-frame fingertip FK distance. It reports nine seeds per user before
cross-user aggregation. No official Pos-MT checkpoint/source has been found,
so this contribution records the specification only and does not claim a run.

### NeuroPose — Liu, Zhang, and Gowda (2021/2022)

| Item | Required contract |
| --- | --- |
| Input | Consumer Myo: 8 channels at 200 Hz, five-second/1,000-sample windows |
| Labels | 16 directly predicted flexion/extension angles plus five anatomy-derived DoF |
| Architecture | Conv2D encoder-decoder with three 3x2/4x2/2x2 stages, ResNet bridge, and nearest-neighbour decoder |
| Constraints | ROM normalization, bounded ReLU, anatomy equations, and temporal smoothness loss |
| Training | Adam `1e-3`; conv L2 `0.01`; dropout `0.05`; adaptive-BN transfer with roughly 90 s target-user data |
| Report | 12 users, 12 sessions; median angular error `6.24` degrees and P90 `18.33` degrees |

`nm000281` has 16 wrist channels at 2 kHz and UmeTrack labels. Resampling or
tiling it would make an explicitly non-paper transfer probe, not NeuroPose
replication. The original dataset, exact adaptation membership, source, and
checkpoint are unavailable. The initial NeuralBench test must reject the
`nm000281` dataset for a `neuropose_paper` request with that explanatory error.

### SensingDynamics — Sîmpetru et al. (2022/2024)

| Item | Required contract |
| --- | --- |
| Input | `(2, 5, 64, 192)`: raw 10-500 Hz and 20-Hz low-pass streams from five 64-electrode grids at 2,048 Hz |
| Cohort | 13 subjects; prompted and random sequences; subject-specific training |
| Outputs | 60 position values, 22 joint angles, or one force value |
| Network | Conv3D temporal detector, SMU, circular grid/electrode padding, Conv3D grid encoder/refiner, 512/512 MLP |
| Training | MAE, 50 epochs; post-hoc 150-ms moving average; Procrustes for position result |
| Target | Prompted post-Procrustes position MED `3.70 +/- 2.14` mm |

The 16-channel nm000281 wrist recording cannot represent the five physical
64-electrode grids. The original data, source, pretrained model, preprocessing
parameters, and aggregation are unavailable. The initial NeuralBench test
must reject `nm000281` for `sensingdynamics_paper`, rather than fabricate a
grid by repeating channels.

## Interfaces and acceptance checks

### Manifest and dataset interface

`Nm000281PoseStudy` accepts a manifest TSV, the selected BDF root, and the
expected source commit/tree and hashes. It returns only manifest rows, ordered
by relative BDF path, with user, stage, side, and split metadata. It fails
before reading BDF samples if any hash, count, or selected path differs.

`PoseWindow` has these fields:

```python
emg: Tensor          # (16, 11790), source numeric scale
joint_angles: Tensor # (20, 11790), radians
bad_ik: Tensor       # (11790,), bool
initial_pose: Tensor # (20,), radians, first supervised frame
user: str
bdf_path: str
```

### Evaluator interface

`OfficialPoseEvaluationConfig` is eval-only and requires explicit paths to
the official runtime, source revision, selected checkpoint, UmeTrack assets,
manifest, BDF root, and output directory. Its public output is only JSON,
CSV, and a SHA-256 completion manifest. It must reject a configuration that
sets `fit`, `train`, an optimizer, or a generic NeuralBench model YAML.

### Metric interface

`OfficialPoseMetricsCallback` consumes official-valid windows and reports:

- `angular_mae_degrees` per user and aggregate;
- `landmark_distance_mm` per user and aggregate;
- `sd_ddof_0`, `sd_ddof_1`, and `users`; and
- exact selected, contributing, and zero-window recording counts.

It must compare the canonical VEMG evaluation to an explicit reference bundle.
The accepted result is an exact value-level comparison, never an assertion
that the paper was replicated.

## Testing requirements

1. Synthetic BDF-independent fixture proves the manifest rejects extra,
   missing, reordered, or wrong-hash paths.
2. Synthetic trajectory fixture proves left context is not supervised,
   `BAD_IK` is excluded, and the first supervised pose becomes `initial_pose`.
3. Per-user metric fixture proves window pooling cannot replace per-user
   aggregation and verifies both ddof conventions.
4. Eval-only config fixture rejects training options and a missing external
   official runtime/checkpoint/UmeTrack path.
5. VEMG sealed-evidence fixture validates aggregate values and 20 per-user
   rows against a provided reference bundle without embedding the noncommercial
   checkpoint or dataset.
6. NeuroPose and SensingDynamics fixtures reject nm000281 for paper-mode runs
   with a reason that names the missing input contract.
7. Environment preflight verifies imports resolve from the current worktree's
   declared NeuralSet package rather than an unrelated path.

## Sources

The Exa semantic search was used for discovery; detailed protocol extraction
was checked against locally archived primary papers/LaTex sources.

- https://doi.org/10.48550/arxiv.2412.02725
- https://doi.org/10.48550/arxiv.2603.08212
- https://doi.org/10.48550/arxiv.2605.30127
- Local primary artifacts: `references/neuropose-www21.pdf`,
  `references/sensingdynamics-biorxiv.pdf`, and
  `references/pose-mt-arxiv-src/main.tex` in the dedicated EMG research
  checkout.

## Risks and explicit stops

The required official runtime and UmeTrack code carry noncommercial terms.
The adapter must remain an external-path integration; if the implementation
would require vendoring or redistributing them, work stops. A successful
official-checkpoint diagnostic does not authorize a Braindecode merge,
architecture port, weight conversion, or a paper-reproduction claim.
