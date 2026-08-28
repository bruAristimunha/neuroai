# NeuralBench EMG Pose Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an eval-only NeuralBench harness for official EMG2Pose/VEMG2Pose checkpoints on sealed nm000281 data, with explicit paper-mode rejections for NeuroPose and SensingDynamics.

**Architecture:** NeuralFetch owns manifest selection and validation. NeuralBench owns trajectory protocol validation, per-user aggregation, provenance, and a small external official-evaluator wrapper. No normal training task or checkpoint conversion is added.

**Tech Stack:** Python 3.12, Pydantic, pandas, PyTorch tensors, pytest, Ruff, NeuralFetch, NeuralBench, and a supplied external EMG2Pose runtime.

**Spec:** `docs/superpowers/specs/2026-08-26-neuralbench-emg-pose-design.md`

## Global Constraints

- Work only on `codex/emg-pose-neuralbench`.
- Every result records `evaluation_scope="official checkpoint / transformed NEMAR prerelease"`, `paper_equivalence_established=false`, and `candidate_model_parity_established=false`.
- No data, checkpoint, UmeTrack asset, or noncommercial source is downloaded into, copied to, or redistributed by this repository.
- Only eval-only official-checkpoint execution is supported. Training, random splits, generic model YAML, and model conversion are rejected.
- NeuroPose and SensingDynamics reject nm000281 paper mode with their missing input contract.

## File Structure

| Path | Responsibility |
| --- | --- |
| `scripts/setup_local_editables.sh` | Reproduce the CI editable-install environment. |
| `neuralfetch-repo/neuralfetch/studies/nm000281emg2pose.py` | Immutable selected-recording manifest validator. |
| `neuralfetch-repo/neuralfetch/test_nm000281emg2pose.py` | Synthetic manifest validation tests. |
| `neuralbench-repo/neuralbench/pose_protocol.py` | Pose window/request types, paper-mode validation, aggregation. |
| `neuralbench-repo/neuralbench/test_pose_protocol.py` | Protocol, metric, and documentation tests. |
| `neuralbench-repo/neuralbench/official_pose_runner.py` | Explicit external evaluator command/result wrapper. |
| `neuralbench-repo/neuralbench/test_official_pose_runner.py` | Command and sealed-result verification tests. |
| `docs/neuralbench/tasks/emg/pose.rst` | Public scope/caveat/paper-availability documentation. |
| `docs/neuralbench/tasks/tasks.rst` | Task index entry. |
| `experiments/emg-pose-neuralbench/run_official_pose_eval.sbatch` | Two-checkpoint Margaret evaluation only after static preflight. |

### Task 1: Reproduce CI's editable environment

**Files:**
- Create: `scripts/setup_local_editables.sh`
- Create: `neuralbench-repo/neuralbench/test_local_environment.py`
- Create: `neuralbench-repo/neuralbench/pose_protocol.py`

**Interfaces:**
- Produces `assert_local_neuroai_imports(expected_root: Path) -> None`.

- [ ] **Step 1: Write the failing test**

```python
def test_local_import_preflight_rejects_external_package(monkeypatch, tmp_path):
    monkeypatch.setattr("neuralset.__file__", "/outside/neuralset/__init__.py")
    with pytest.raises(RuntimeError, match="outside the current worktree"):
        assert_local_neuroai_imports(tmp_path)
```

- [ ] **Step 2: Verify RED**

Run: `source .venv/bin/activate && cd neuralbench-repo && pytest neuralbench/test_local_environment.py -q`

Expected: import failure because `pose_protocol` does not exist.

- [ ] **Step 3: Implement GREEN**

Implement `assert_local_neuroai_imports` to inspect `neuralbench`, `neuralset`, `neuralfetch`, and `neuraltrain` module origins and reject any origin outside `expected_root`. Add `scripts/setup_local_editables.sh` using the four strict editable installs from `.github/actions/setup-python-env/action.yml`.

- [ ] **Step 4: Verify GREEN**

Run: `scripts/setup_local_editables.sh && source .venv/bin/activate && cd neuralbench-repo && pytest neuralbench/test_local_environment.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/setup_local_editables.sh neuralbench-repo/neuralbench/pose_protocol.py neuralbench-repo/neuralbench/test_local_environment.py
git commit -m "test: guard neuralbench local editable imports"
```

### Task 2: Add immutable nm000281 manifest validation

**Files:**
- Create: `neuralfetch-repo/neuralfetch/studies/nm000281emg2pose.py`
- Create: `neuralfetch-repo/neuralfetch/test_nm000281emg2pose.py`

**Interfaces:**
- Produces `Nm000281Emg2poseManifest(manifest_path: Path, bids_root: Path, expected_manifest_sha256: str)`.
- `selected_recordings() -> pandas.DataFrame` exposes only ordered `bdf_path,user,stage,side,split` rows.
- `validate_selected_recordings() -> pandas.DataFrame` requires 456 unique relative paths, 20 users, and `user_stage` rows.

- [ ] **Step 1: Write the failing test**

```python
def test_manifest_rejects_digest_and_path_contract_violations(tmp_path):
    manifest = tmp_path / "selected.tsv"
    manifest.write_text("bdf_path\tuser\tstage\tsplit\n../escape.bdf\t01\t1\tuser_stage\n")
    study = Nm000281Emg2poseManifest(
        manifest_path=manifest, bids_root=tmp_path, expected_manifest_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="SHA-256|relative"):
        study.validate_selected_recordings()
```

- [ ] **Step 2: Verify RED**

Run: `source .venv/bin/activate && cd neuralfetch-repo && pytest neuralfetch/test_nm000281emg2pose.py -q`

Expected: import failure for `Nm000281Emg2poseManifest`.

- [ ] **Step 3: Implement GREEN**

Use streaming SHA-256 with 1 MiB chunks and `PurePosixPath` validation. Reject absolute paths, traversal, duplicates, missing BDFs, non-`user_stage` rows, non-456 cardinality, and non-20-user manifests. Do not implement `download`.

- [ ] **Step 4: Verify GREEN**

Run: `source .venv/bin/activate && cd neuralfetch-repo && pytest neuralfetch/test_nm000281emg2pose.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add neuralfetch-repo/neuralfetch/studies/nm000281emg2pose.py neuralfetch-repo/neuralfetch/test_nm000281emg2pose.py
git commit -m "feat: validate nm000281 pose manifests"
```

### Task 3: Encode pose protocol and paper availability

**Files:**
- Modify: `neuralbench-repo/neuralbench/pose_protocol.py`
- Create: `neuralbench-repo/neuralbench/test_pose_protocol.py`

**Interfaces:**
- Produces `PoseWindowSpec(input_samples=11790, left_context_samples=1790, supervised_samples=10000, channels=16, joints=20)`.
- Produces `PoseEvaluationRequest` with explicit external paths and `eval_only=True`.
- Produces `aggregate_pose_metrics(rows: pandas.DataFrame) -> dict[str, Any]`.
- Produces `validate_paper_mode(model_name: str, source_contract: Mapping[str, int]) -> None`.

- [ ] **Step 1: Write failing tests**

```python
def test_window_spec_supervises_only_final_10000_samples():
    assert PoseWindowSpec().supervised_slice == slice(1790, 11790)

def test_metrics_average_within_user_before_cross_user_mean():
    rows = pd.DataFrame({"user": ["a", "a", "b"], "ae": [1.0, 3.0, 9.0], "ld": [2.0, 6.0, 18.0]})
    assert aggregate_pose_metrics(rows)["angular_mae_degrees"]["mean"] == 5.5

def test_neuropose_paper_mode_rejects_nm000281_contract():
    with pytest.raises(ValueError, match="8 channels at 200 Hz"):
        validate_paper_mode("neuropose", {"channels": 16, "sample_rate": 2000})
```

- [ ] **Step 2: Verify RED**

Run: `source .venv/bin/activate && cd neuralbench-repo && pytest neuralbench/test_pose_protocol.py -q`

Expected: import failure for `PoseWindowSpec`.

- [ ] **Step 3: Implement GREEN**

Use frozen dataclasses. Enforce exact left context, `BAD_IK` mask length, 20-user aggregation, and both ddof values. Reject training/optimizer fields. Require 8 channels/200 Hz for NeuroPose and 320 channels/2,048 Hz/5 grids for SensingDynamics. Every constructed result carries false equivalence/parity flags.

- [ ] **Step 4: Verify GREEN**

Run: `source .venv/bin/activate && cd neuralbench-repo && pytest neuralbench/test_pose_protocol.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add neuralbench-repo/neuralbench/pose_protocol.py neuralbench-repo/neuralbench/test_pose_protocol.py
git commit -m "feat: add eval-only emg pose protocol"
```

### Task 4: Wrap, but do not vendor, official evaluation

**Files:**
- Create: `neuralbench-repo/neuralbench/official_pose_runner.py`
- Create: `neuralbench-repo/neuralbench/test_official_pose_runner.py`

**Interfaces:**
- Produces `build_official_pose_command(request: PoseEvaluationRequest) -> list[str]`.
- Produces `verify_sealed_pose_result(result_directory: Path, expected: Mapping[str, str]) -> dict[str, Any]`.

- [ ] **Step 1: Write failing tests**

```python
def test_runner_uses_only_explicit_external_paths(tmp_path):
    command = build_official_pose_command(_request(tmp_path, "tracking_vemg2pose"))
    assert "--checkpoint" in command
    assert "--train" not in command

def test_tampered_result_payload_is_rejected(tmp_path):
    (tmp_path / "aggregate_metrics.json").write_text("{}")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_sealed_pose_result(tmp_path, {"aggregate_metrics.json": "0" * 64})
```

Add a fixture for job 500932's aggregate values: AE `11.037608559112806`, LD `15.375535790427591`, 20 users, 3,562 windows, and false equivalence.

- [ ] **Step 2: Verify RED**

Run: `source .venv/bin/activate && cd neuralbench-repo && pytest neuralbench/test_official_pose_runner.py -q`

Expected: import failure for `build_official_pose_command`.

- [ ] **Step 3: Implement GREEN**

Require all paths to exist and be outside the checkout. Build only an argument list for the supplied evaluator; run it only in `run_official_pose_evaluation` through checked subprocess execution. Rehash payloads, require a completion manifest, and reject result JSON that claims equivalence or candidate parity.

- [ ] **Step 4: Verify GREEN**

Run: `source .venv/bin/activate && cd neuralbench-repo && pytest neuralbench/test_official_pose_runner.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add neuralbench-repo/neuralbench/official_pose_runner.py neuralbench-repo/neuralbench/test_official_pose_runner.py
git commit -m "feat: wrap official emg pose evaluation"
```

### Task 5: Document and preflight the two admissible diagnostics

**Files:**
- Create: `docs/neuralbench/tasks/emg/pose.rst`
- Modify: `docs/neuralbench/tasks/tasks.rst`
- Create: `experiments/emg-pose-neuralbench/run_official_pose_eval.sbatch`
- Create: `experiments/emg-pose-neuralbench/README.md`

**Interfaces:**
- Documents a Python eval-only API, not a stock `neuralbench emg pose` training command.
- Sbatch invokes generic EMG2Pose and VEMG2Pose only after path preflight.

- [ ] **Step 1: Write failing documentation and script tests**

```python
def test_pose_docs_and_sbatch_forbid_training():
    assert "paper_equivalence_established=false" in POSE_DOC.read_text()
    assert "NeuroPose" in POSE_DOC.read_text() and "blocked" in POSE_DOC.read_text()
    assert "--train" not in SBATCH.read_text()
    assert "tracking_emg2pose.ckpt" in SBATCH.read_text()
    assert "tracking_vemg2pose.ckpt" in SBATCH.read_text()
```

- [ ] **Step 2: Verify RED**

Run: `source .venv/bin/activate && cd neuralbench-repo && pytest neuralbench/test_pose_protocol.py::test_pose_docs_and_sbatch_forbid_training -q`

Expected: file-not-found failure.

- [ ] **Step 3: Implement GREEN**

Document all four paper contracts, the immutable data boundary, the downloaded-paper archive, exact VEMG diagnostic values, the generic EMG run target, and strict NeuroPose/SensingDynamics blocks. Sbatch must use read-only Iceberg/runtime paths, atomic output children, and one run per approved checkpoint.

- [ ] **Step 4: Verify GREEN**

Run: `source .venv/bin/activate && cd neuralbench-repo && pytest neuralbench/test_pose_protocol.py neuralbench/test_official_pose_runner.py -q && ruff check . && ruff format --check .`

Run: `source .venv/bin/activate && sphinx-build -W -b html ../docs ../docs/_build/html`

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add docs/neuralbench/tasks/emg/pose.rst docs/neuralbench/tasks/tasks.rst experiments/emg-pose-neuralbench neuralbench-repo/neuralbench
git commit -m "docs: add eval-only emg pose protocol"
```

## Plan self-review

- Tasks 2--4 map directly to manifest, protocol, and official-evaluator requirements in the spec.
- Task 5 records every paper's availability and creates only two data-compatible diagnostics.
- No task trains a model, creates a surrogate NeuroPose/SensingDynamics score, vendors upstream assets, or authorizes a model merge.
- `PoseEvaluationRequest` is the only runner request type; per-user aggregation remains isolated in `aggregate_pose_metrics`.

