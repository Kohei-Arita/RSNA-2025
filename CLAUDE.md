# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an RSNA Intracranial Aneurysm Detection competition project designed with a three-tier architecture:
- **Research Layer (Google Colab)**: Data exploration, model training, and experimentation
- **Storage Layer (Google Cloud Storage + DVC)**: Data versioning and artifact management
- **Submission Layer (Kaggle Notebooks Only)**: Offline inference with Internet disabled

The project uses Hydra for configuration management, Weights & Biases for experiment tracking, and follows a strict separation between research and submission environments.

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

### Core Components

**CLI Entry Point**: `src/rsna_aneurysm/cli.py`
- Unified command-line interface for train/infer/eval operations
- Uses Hydra for configuration composition and overrides

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

**Time Budget Management**:
The system implements automatic downgrading when approaching time limits:
1. Reduce TTA (Test Time Augmentation)
2. Increase patch stride (coarser sampling)
3. Reduce candidate count limit
4. Reduce input resolution
Priority is "complete execution within time budget" over maximum accuracy.

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

### Code Quality Requirements
- **Priority 1**: Implement DICOM geometry tests (`tests/test_dicom_geometry.py`) - currently skipped
- **Priority 2**: Implement core data pipeline (dataset.py, dicom_utils.py, transforms.py)
- **Priority 3**: Implement model architecture and training loop (model.py, loss.py)
- Geometric consistency (spacing, orientation, intensity scaling) must be verified before training
- Failure in geometry tests should halt training/inference operations

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