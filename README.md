# Crossing Challenge Submission

This repository contains my entry for Gobblecube's Crossing Challenge.
The system predicts:
- pedestrian crossing intent in the next 2 seconds (`intent` in `[0,1]`)
- future pedestrian bounding boxes at `+0.5s`, `+1.0s`, `+1.5s`, `+2.0s`

Entrypoint required by the grader:

```python
def predict(request: dict) -> dict:
    ...
```

## Repo Contents

Core inference and grading:
- `predict.py` - grader entrypoint (`predict(request) -> dict`)
- `grade.py` - local scorer aligned with challenge scoring rules

Training and preprocessing:
- `prepare_data.py` - parquet to model-ready arrays
- `train_tabular_models.py` - tabular intent + direct trajectory regressors
- `train_residual_models.py` - residual trajectory regressors over constant-velocity
- `train_traj_seq.py` - sequence trajectory model training
- `train_model.py` - earlier GRU joint model experiment
- `traj_seq_model.py`, `model.py` - model definitions
- `tune_params.py` - blend/temperature tuning on a dev sample

Model artifacts used by inference:
- `tabular_models.pkl`
- `traj_residual_models.pkl`
- `traj_seq.pth`
- `pedestrian_predictor.pth` (legacy experiment artifact)

Data and docs:
- `data/train.parquet`
- `data/dev.parquet`
- `data/schema.md`

## Approach Summary

### Intent
- Tabular feature pipeline from bbox/ego/context history.
- `XGBClassifier` for `will_cross_2s`.
- Isotonic calibration plus mild temperature scaling at inference.

### Trajectory
- Constant-velocity baseline from recent bbox centers.
- Direct tabular regressors for future centers.
- Residual regressors that learn corrections over constant-velocity.
- Sequence trajectory model blended with tabular/residual outputs.

### Output reconstruction
Predicted future centers are converted to `[x1,y1,x2,y2]` using recent average bbox width/height from valid history.

## Reproduce

Install dependencies:

```bash
pip install -r requirements.txt
```

Train/update models:

```bash
python prepare_data.py
python train_tabular_models.py
python train_residual_models.py
python train_traj_seq.py
```

Optional parameter tuning:

```bash
python tune_params.py
```

Local grade:

```bash
python grade.py
```

Contract checks (if tests are present in your local copy):

```bash
python -m pytest tests/
```

## Docker

Build:

```bash
docker build -t my-crossing -f dockerfile .
```

Offline run check:

```bash
docker run --rm --network=none my-crossing python grade.py
```

## Submission Notes

This repo is set up for offline inference (`--network=none`) and row-by-row prediction through `predict()`.

Files that matter most for challenge scoring:
- `predict.py`
- `dockerfile`
- `requirements.txt`
- model artifacts (`*.pkl`, `*.pth`)
- `README.md`

## Current Local Validation

Latest local dev sample result (`python grade.py`, 5,000-row sample):
- `Score: 0.6936`
- `intent_term: 0.827` (`BCE: 0.2057`)
- `traj_term: 0.561` (`ADE: 28.0 px`)

## Next Improvements (if continuing)

- Stronger horizon-aware sequence trajectory head.
- Separate per-horizon residual model tuning.
- Calibration on a held-out split separate from dev tuning.
- Longer GPU sweeps for sequence trajectory training.
