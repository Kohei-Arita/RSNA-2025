# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an RSNA Intracranial Aneurysm Detection competition project designed with a three-tier architecture:
- **Research Layer (Google Colab)**: Data exploration, model training, and experimentation
- **Storage Layer (Google Cloud Storage + DVC)**: Data versioning and artifact management
- **Submission Layer (Kaggle Notebooks Only)**: Offline inference with Internet disabled

The project uses Hydra for configuration management, Weights & Biases for experiment tracking, and follows a strict separation between research and submission environments.

## 🚨 CRITICAL DATA POLICY

**NEVER USE DUMMY OR DEFAULT DATA**: This is a medical imaging competition with real patient data. Always use actual data from GCS (`gs://rsna2025-prod/`) via authenticated access. Never create, generate, or use placeholder/synthetic/dummy data for any purpose - including testing, debugging, or prototyping. All data must come from the official competition dataset stored in Google Cloud Storage.

## ⚠️ Current Implementation Status

**CRITICAL**: Most Python files in `src/rsna_aneurysm/` contain only comment stubs and require implementation:
- All core modules (dataset.py, model.py, etc.) need actual code
- DICOM geometry tests are skipped and must be implemented first
- Configuration files (configs/*.yaml) may be corrupted or incomplete
- The codebase is in skeleton/planning stage, not ready for execution

## Common Commands

### Environment Setup (Google Colab)
```bash
# Install dependencies
pip install -r env/requirements.txt

# Login to W&B
wandb login

# Setup Kaggle API
mkdir -p ~/.kaggle
cp /content/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Initialize DVC with GCS remote
dvc pull
```

### Data Management
```bash
# Download competition data
kaggle competitions download -c rsna-intracranial-aneurysm-detection -p data/raw
unzip -q data/raw/rsna-intracranial-aneurysm-detection.zip -d data/raw

# Sync with DVC remote (optional)
dvc pull  # Get shared data
dvc push  # Share new artifacts
```

### Training
```bash
# Basic training with Hydra configuration
python -m rsna_aneurysm.cli train \
  model=baseline_2d cv=patient_kfold train=base,fp16 data=rsna \
  paths=colab wandb.project=RSNA2025

# Multi-run experiments
python -m rsna_aneurysm.cli train \
  model=efficientnet,convnext cv=patient_kfold \
  --multirun
```

### Inference
```bash
# Standard inference
python -m rsna_aneurysm.cli infer \
  inference=base model=baseline_2d paths=colab \
  checkpoint_path=models/exp0001_fold0_best.ckpt
```

### Kaggle Submission Preparation
```bash
# Package precomputed data for Kaggle
make kaggle-prep

# Local dry run (CSV format for testing)
make kaggle-dryrun

# Collect wheels for offline installation
make wheels
```

### Testing
```bash
# Run specific test modules
python -m pytest tests/test_dataset.py -v
python -m pytest tests/test_transforms.py -v
python -m pytest tests/test_metrics.py -v
python -m pytest tests/test_inference.py -v

# Run all tests
python -m pytest tests/ -v
```

## Architecture & Code Structure

### Three-Tier Architecture Deep Dive

**Research Layer (Colab Environment)**:
- Full Google Cloud integration with `paths=colab` configuration
- Unlimited access to 311GB dataset via DVC (`dvc pull/push` workflows)
- Interactive development with W&B experiment tracking
- Complete Python ecosystem: `pip install -r env/requirements.txt`

**Storage Layer (GCS + DVC)**:
- Data versioning: DVC manages competition data via GCS remote
- Compression strategy: npz/float16 for volumes, parquet for metadata
- Artifact tracking: Models, preprocessed volumes, fold assignments
- Access pattern: Research pulls full data, submission uses precomputed subsets

**Submission Layer (Kaggle Constraints)**:
- Internet disabled, ~20GB input limit, 9-12 hour time budget
- Serving architecture: STDIN/STDOUT JSON API (`kaggle/kaggle_infer.py --serve`)
- Automatic downgrading: resolution→TTA→stride→candidates under time pressure
- Configuration: `paths=kaggle wandb=disabled inference=kaggle_fast`

### Core Components

**CLI Entry Point**: `src/rsna_aneurysm/cli.py`
- Unified command-line interface for train/infer/eval operations
- Uses Hydra for configuration composition and overrides
- Supports environment switching via configuration composition

**Data Pipeline**:
- `src/rsna_aneurysm/dataset.py`: DICOM processing and dataset implementation
- `src/rsna_aneurysm/datamodule.py`: PyTorch Lightning DataModule
- `src/rsna_aneurysm/dicom_utils.py`: DICOM reading and preprocessing utilities
- `src/rsna_aneurysm/transforms.py`: Data augmentation and preprocessing transforms

**Model Architecture**:
- `src/rsna_aneurysm/model.py`: PyTorch Lightning Module for training loop
- `src/rsna_aneurysm/loss.py`: Custom loss functions (Focal Loss, etc.)
- `src/rsna_aneurysm/metrics.py`: Evaluation metrics (AUC, Sensitivity, etc.)

**Training Infrastructure**:
- `src/rsna_aneurysm/optimizer.py`: Optimizer configurations
- `src/rsna_aneurysm/scheduler.py`: Learning rate schedulers

**Inference & Evaluation**:
- `src/rsna_aneurysm/inference.py`: Model inference utilities
- `src/rsna_aneurysm/oof_utils.py`: Out-of-fold prediction and evaluation
- `src/rsna_aneurysm/postprocess.py`: Post-processing (ensemble, NMS)
- `src/rsna_aneurysm/visualization.py`: Visualization tools (Grad-CAM, learning curves)

### Configuration System (Hydra)

**Composition Patterns**: Hydra enables sophisticated configuration mixing:
```bash
# Environment switching
train paths=colab     # Research environment with full capabilities
infer paths=kaggle wandb=disabled inference=kaggle_fast  # Submission constraints

# Multi-component mixing
train model=baseline_2d cv=patient_kfold train=base,fp16 data=rsna
#     ↑architecture  ↑CV strategy      ↑training modes   ↑dataset

# Multi-run experiments
train model=efficientnet,convnext cv=patient_kfold --multirun
```

**Main Config**: `configs/config.yaml` - Central defaults and composition rules

**Component Configs**:
- `configs/model/`: Architecture definitions (baseline_2d.yaml, efficientnet.yaml, etc.)
- `configs/data/`: Dataset configurations (rsna.yaml, cache.yaml, external.yaml)
- `configs/cv/`: Cross-validation strategies (patient_kfold.yaml, groupkfold.yaml)
- `configs/train/`: Training settings (base.yaml, fp16.yaml, swa.yaml, tta.yaml)
- `configs/aug/`: Data augmentation levels (light.yaml, medium.yaml, heavy.yaml)
- `configs/paths/`: Environment-specific paths (local.yaml, colab.yaml, kaggle.yaml)
- `configs/inference/`: Inference configurations (base.yaml, kaggle_fast.yaml)

### Experiment Management

**Structure**: Each experiment gets its own directory under `experiments/expXXXX/`
- `config.yaml`: Snapshot of Hydra configuration for reproducibility
- `training.ipynb`: Training execution notebook
- `evaluation.ipynb`: OOF generation and evaluation
- `inference.ipynb`: Inference execution
- `notes.md`: Experiment notes, W&B links, and retrospective

**Data Flow**:
- `outputs/`: Regenerable intermediate outputs (OOF predictions, test predictions)
- `reports/`: Permanent shared artifacts (figures for papers/presentations)
- `models/`: Trained model checkpoints (managed via DVC/W&B)

### Kaggle Submission Architecture

**Dual Environment Design**:
- Research (Colab): Full logging, visualization, iterative development
- Submission (Kaggle): Minimal inference, offline operation, time-constrained

**Key Files**:
- `kaggle/kaggle_infer.py`: Main inference server for Kaggle submission
- `kaggle/kaggle_utils.py`: Kaggle-specific utilities
- `kaggle/offline_requirements.txt`: Minimal dependencies for offline installation
- `tools/pack_precompute.py`: Packages precomputed data for Kaggle datasets
- `tools/verify_submission.py`: Validates submission format (local testing only)

**Time Budget Management Architecture**:
Automatic quality degradation under Kaggle's 9-12 hour time constraint:
```python
# Progressive quality reduction when time running out:
if eta_remaining < threshold:
    disable_tta()           # Remove test-time augmentation
    increase_patch_stride() # Coarser spatial sampling
    reduce_candidates()     # Fewer candidate detection points
    lower_resolution()      # Reduce input image resolution
```
Priority: "complete execution within time budget" over maximum accuracy.

**Serving API Contract**:
- Input: JSON with `series_id` via STDIN
- Output: 14 probability labels [0,1] per series via STDOUT
- Format: `aneurysm_present` + 13 anatomical location probabilities
- Constraint: Must work offline with precomputed data only

## Important Development Notes

### Environment Switching
- Use `paths=colab` for research environment
- Use `paths=kaggle wandb=disabled inference=kaggle_fast` for Kaggle environment
- Configuration composition allows mixing: `train=base,fp16` combines base training with mixed precision

### Data Constraints
- Competition data is ~311GB, managed via DVC + GCS remote
- Kaggle submission limited to ~20GB total input datasets
- Focus on essential precomputed features and model weights only
- Use compressed formats: npz/float16 for volumes, parquet for metadata

### Cross-Validation Strategy
- Patient-level splits to prevent data leakage
- CV fold assignments should be fixed and versioned for reproducibility
- All experiments must use consistent fold definitions

### Code Quality Requirements & Medical Imaging Constraints

**🚨 DICOM Geometry Validation (BLOCKING)**:
- **Requirement**: `tests/test_dicom_geometry.py` must pass before ANY training/inference
- **Critical validations**: RescaleSlope/Intercept, coordinate systems (LPS/RAS), voxel spacing consistency
- **Patient coordinate conversion**: mm coordinates ↔ voxel indices must be mathematically verified
- **Failure mode**: Training/inference operations HALT if geometry tests fail
- **Medical significance**: Incorrect geometry → wrong anatomical locations → patient harm

**Implementation Priority Order**:
1. **DICOM geometry tests** - blocking requirement, prevents all downstream work
2. **Core data pipeline** - dataset.py, dicom_utils.py, transforms.py with real GCS data
3. **Model architecture** - dual-head for aneurysm presence + 13 anatomical locations
4. **Cross-validation** - patient-level splits, no data leakage
5. **Kaggle packaging** - time budget management, automatic quality degradation

### Competition Submission Requirements
- **Format**: Server must respond with 14-label probabilities [0,1] per series_id
- **Labels**: `aneurysm_present` + 13 anatomical location labels
- **Contract**: Definition in `docs/SUBMISSION_CONTRACT.md` (note: may be outdated, needs sync with official API)
- **Serving**: Kaggle uses serving API, not CSV submission
- **Time Limit**: Must complete within 9-12 hours with automatic downgrading
- **Offline**: Must work with Internet disabled using precomputed data

### Development Workflow
1. **Start with tests**: Implement and pass `test_dicom_geometry.py` first
2. **Build data pipeline**: Implement DICOM loading and preprocessing
3. **Add model training**: Implement PyTorch Lightning training loop
4. **Test inference**: Verify end-to-end inference pipeline
5. **Package for Kaggle**: Use `make kaggle-prep` to prepare submission

This codebase is structured for competition reproducibility with clear separation of concerns between research and production inference environments.