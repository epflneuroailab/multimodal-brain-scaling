# Multimodal Scaling Laws for Task & Data-Optimized Models of Visual Cortex

[![Project Page](https://img.shields.io/badge/Project%20Page-EPFL%20site-E60028.svg?logo=googlechrome&logoColor=white)](https://multimodal-brain-scaling.epfl.ch)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Results%20Dataset-FFD21E.svg?logo=huggingface)](https://huggingface.co/datasets/epfl-neuroai/multimodal-brain-scaling)
[![OpenReview](https://img.shields.io/badge/OpenReview-OQ6jQHJPTT-1A3D91.svg?logoColor=white)](https://openreview.net/forum?id=OQ6jQHJPTT)
[![ICML 2026](https://img.shields.io/badge/ICML-2026-0B5D1E.svg?logoColor=white)](https://icml.cc/virtual/2026/poster/64356)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)

Official code and result artifacts for:

> **Multimodal Scaling Laws for Task & Data-Optimized Models of Visual Cortex**<br>
> Abdulkadir Gokce, Yingtian Tang, Martin Schrimpf<br>
> International Conference on Machine Learning (ICML), 2026

Task-optimized neural networks are among the strongest in-silico models of visual cortex, but it is still unclear which resources most reliably improve model-to-brain alignment. This repository supports a unified study of alignment scaling across **8 neural datasets**, **600+ vision models**, and multiple recording modalities: macaque electrophysiology/spiking and human fMRI, EEG, and MEG.

We analyze three controllable axes:

- **Pretraining scale:** data, model size, and compute improve alignment but show clear saturation.
- **Neural fine-tuning:** hybrid task and neural supervision yields small but consistent alignment gains that transfer across datasets and modalities.
- **Mapping scale:** increasing paired stimulus-response samples for the model-to-brain readout produces the largest and most reliable gains.

Beyond the scaling analyses, we introduce a **subject-shared cross-attention readout** that ties learned queries across subjects while keeping subject- and ROI-specific output heads. It matches or exceeds per-subject linear decoding in several regimes while using up to an order of magnitude fewer parameters.

The codebase provides workflows for extracting model features, fitting linear and subject-shared cross-attention readouts, fine-tuning vision backbones with neural supervision, fitting scaling curves, and restoring published result tables.

## Repository Contents

```text
configs/                 Research configs for models, training, readouts, layer commitments, and curve fitting
scripts/                 End-to-end example shell workflows (train, extract, evaluate, fit curves)
src/mbs/core/            Shared CLI, config, and path helpers
src/mbs/metrics/         RSA, CKA, and Ridge-GCV metric implementations
src/mbs/modeling/        Shared LoRA and projection utilities
src/mbs/training/        Neural fine-tuning workflow
src/mbs/extraction/      Feature extraction workflow
src/mbs/evaluation/      Ridge and attention-probe evaluation workflows
src/mbs/analysis/        Scaling-law curve fitting
src/mbs/visualization/   Plotting utilities used in analysis
tests/                   Import, CLI, static, and utility smoke tests
```

This is a `src`-layout Python package with import root `mbs`. Research configs intentionally live at the repository root under `configs/`, so most workflows should be run from a cloned checkout or with explicit config paths.

## Installation

We recommend [`uv`](https://docs.astral.sh/uv/) for reproducible dependency management.

```bash
git clone https://github.com/epflneuroailab/multimodal-brain-scaling.git
cd multimodal-brain-scaling

uv sync --locked
```

Optional dependency groups (`training`, `evaluation`, `analysis`, `visualization`, `dev`) gate the per-workflow requirements. `uv sync` resolves the union of every group passed in a single invocation, so combine them as needed (it will also drop extras you omit). On macOS arm64, `scikit-learn-intelex` is skipped and evaluation falls back to standard scikit-learn.

```bash
uv sync --all-extras                                 # everything (recommended for reproduction)
uv sync --extra training --extra evaluation          # fine-tune + evaluate
uv sync --extra analysis --extra visualization       # fit + plot curves from released artifacts
uv sync --extra dev                                  # tests
```

When chaining workflows, sync all needed extras once, then run commands with
`--no-sync` so `uv` does not prune packages from extras omitted by a later
command. The example scripts do this automatically: the first script you run
prepares the combined environment, writes a readiness marker in `.venv`, and
later scripts reuse that environment without syncing again.

```bash
scripts/setup_env.sh                                 # optional explicit setup
uv run --no-sync mbs-download-artifacts --help
uv run --no-sync mbs-train --help
uv run --no-sync mbs-extract-features --help
uv run --no-sync mbs-evaluate-all-layers --help
uv run --no-sync mbs-evaluate-committed-layers --help
uv run --no-sync mbs-evaluate-attn-probe --help
uv run --no-sync mbs-fit-curves --help
```

By default, `scripts/setup_env.sh` and the example scripts install
`training evaluation analysis`. Override that set with `MBS_EXTRAS`, for
example `MBS_EXTRAS="training evaluation" scripts/setup_env.sh`, or use
`MBS_EXTRAS=all-extras` for every optional dependency. Set
`MBS_FORCE_SYNC=true` to refresh the environment deliberately.

If you have already synced and activated an environment with the corresponding extras, the `uv run ...` prefix can be omitted. Each script is also reachable as `python -m mbs.<module>` (e.g. `python -m mbs.training.train`) if you prefer module-style invocation.

## Published Artifacts

Published result tables and metadata are hosted on Hugging Face:

```text
https://huggingface.co/datasets/epfl-neuroai/multimodal-brain-scaling
```

Restore the released Parquet tables as local CSV files:

```bash
uv run --no-sync mbs-download-artifacts --artifacts-dir artifacts
```

The downloader restores tables such as:

- `pretraining_results_with_metadata.csv`
- `layer_search_results.csv`
- `finetuning_results.csv`
- `mapping_results.csv`

Raw neural datasets and image stimuli are not stored in this repository. Please obtain them from their original sources and licenses.

## Neural Benchmarks

The study evaluates visual model alignment on eight public neural benchmarks:

| Handle | Modality | Dataset |
| --- | --- | --- |
| `FZ-EP` | electrophysiology | Freeman & Ziemba 2013, macaque V1/V2 |
| `MH-EP` | electrophysiology | Majaj & Hong 2015, macaque V4/IT |
| `TVSD-EP` | electrophysiology | THINGS Ventral-stream Spiking Dataset |
| `T-fMRI` | fMRI | THINGS-fMRI |
| `NSD-fMRI` | fMRI | Natural Scenes Dataset |
| `T-MEG` | MEG | THINGS-MEG |
| `T-EEG1` | EEG | THINGS-EEG1 |
| `T-EEG2` | EEG | THINGS-EEG2 |

Evaluation reports held-out neural predictivity with Pearson correlation, noise-ceiling normalization, and optional RSA/CKA metrics. Layer commitments are selected with training-split cross-validation and stored in:

```text
configs/evaluation/layer_commitment/
```

## Model Sources

The primary controlled scaling analyses use the `scaling-primate-vvs` model suite, which contains 600+ supervised vision models with systematically varied architecture, parameter count, pretraining samples, and estimated compute. Additional `timm` models provide larger-scale and alternative pretraining regimes, including supervised, self-supervised, and image-language contrastive models.

For SPVVS checkpoints, set:

```bash
export SCALING_MODELS_CKPTS_DIR=/path/to/scaling_primate_vvs/checkpoints
```

If this variable is unset or a local checkpoint is missing, supported loaders fall back to checkpoint URLs from the model metadata.

## Example Workflows

Example scripts are intentionally lightweight and are meant to be edited or driven by environment variables.

### 1. Fine-tune a backbone with neural supervision

```bash
DATA_PATH_IMAGE=/path/to/imagenet \
DATA_PATH_NEURAL=/path/to/neural_data \
DATA_NEURAL_FILENAME=SachiMajajHong2015.h5 \
DATA_NEURAL_REGIONS=V4,IT \
OUTPUT_DIR=outputs/train_example \
bash scripts/train_example.sh
```

### 2. Extract features for a stimulus set

```bash
MODEL_ID=resnet50_imagenet_full \
DATA_ROOT=/path/to/stimuli.h5 \
DATASET_TYPE=h5 \
STIMULUS_SET_ID=object_images \
OUTPUT_DIR=outputs/features/resnet50_imagenet_full \
bash scripts/extract_example.sh
```

When `--committed_extraction_layers` is used, `STIMULUS_SET_ID` selects the stimulus-set key in `configs/evaluation/layer_commitment/committed_extraction_layers.json`. Feature extraction supports:

- HDF5 image datasets (`--dataset_type h5`)
- THINGS-style image folders (`--dataset_type things`)
- Brain-Score stimulus sets (`--dataset_type brain_score`, requires the evaluation extra)

Large activations can be compressed with fixed random projections, with cached projectors controlled by `--projector_cache` and `--use_cached_projectors`.

### 3. Evaluate committed layers

```bash
MODEL_ID=resnet50_imagenet_full \
FEATURES_DIR=outputs/features/resnet50_imagenet_full \
DATA_HDF5_PATH=/path/to/neural_benchmark.h5 \
OUTPUT_DIR=outputs/evaluation/resnet50_imagenet_full \
bash scripts/evaluate_example.sh
```

Evaluation workflows include:

- all-layer ridge sweeps: `mbs-evaluate-all-layers`
- committed-layer ridge evaluation: `mbs-evaluate-committed-layers`
- attention-based readouts: `mbs-evaluate-attn-probe`

### 4. Fit scaling curves

After restoring the released result tables (see [Published Artifacts](#published-artifacts)), fit a pretraining-compute scaling curve aggregated across architectures and benchmarks with:

```bash
RESULTS_CSV=./artifacts/pretraining_results_with_metadata.csv \
EXPERIMENT_CONFIG=configs/analysis/scaling_compute/architecture_average/benchmark_average.yaml \
OUTPUT_DIR=outputs/curve_fits \
bash scripts/fit_curves_example.sh
```

The shipped `configs/analysis/` tree reproduces every scaling-curve experiment in the paper: pretraining compute / samples / parameters scans (`scaling_compute/`, `scaling_samples/`, `scaling_parameters/`), neural fine-tuning scaling (`finetuning/`), and mapping-data scaling for linear and attention-probe readouts (`mapping/`). Each subdirectory has a `*-base.yaml` capturing shared filters and fitting hyperparameters; per-architecture and per-benchmark leaves inherit via `base_config:` and only override the filters they change. Swap the `EXPERIMENT_CONFIG` path to reproduce any cell of the paper's results.

## Configs

Important config groups:

```text
configs/models/spvvs/                      SPVVS model configs
configs/training/encoders/                 encoder training and fine-tuning configs
configs/training/decoders/                 neural decoder configs
configs/training/lora/                     LoRA configs
configs/evaluation/readouts/               low-rank, shallow-MLP, and attention-probe readout configs
configs/evaluation/layer_commitment/       committed layers used for evaluation
configs/analysis/scaling_compute/          pretraining compute (FLOPs) scaling fits
configs/analysis/scaling_samples/          pretraining sample-count scaling fits
configs/analysis/scaling_parameters/       model-size scaling fits
configs/analysis/finetuning/               neural fine-tuning data scaling fits
configs/analysis/mapping/                  mapping-data scaling fits (linear and attention probes)
```

Configs can inherit from `base_config` files. Paths are resolved relative to the config file being loaded.

## Development

Run the test suite with:

```bash
uv run --no-sync pytest
```

If `uv` is unavailable but a compatible environment is active:

```bash
python -m pytest
python -m compileall -q src tests
```

Some CLI smoke tests skip optional entry points when the corresponding extras are not installed.

## Citation

```bibtex
@inproceedings{gokce2026multimodal_brain_scaling,
  title     = {Multimodal Scaling Laws for Task \& Data-Optimized Models of Visual Cortex},
  author    = {Abdulkadir Gokce and Yingtian Tang and Martin Schrimpf},
  booktitle = {Forty-third International Conference on Machine Learning},
  year      = {2026},
  url       = {https://openreview.net/forum?id=OQ6jQHJPTT}
}
```

## License

This repository is released under the terms in [LICENSE](LICENSE). External datasets, model checkpoints, and benchmark sources may carry separate licenses and access requirements.
