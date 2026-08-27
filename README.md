# ExFMECG code repository

## Repository structure

```text
.
├── assets/
│   ├── task_labels.json                 # ordered task labels
│   └── concepts/                       # ordered runtime resources and statistics
├── examples/
│   ├── E07303.mat                      # public Georgia ECG source record
│   ├── E07303.hea                      # waveform and clinical metadata
│   ├── E07303.npy                      # preprocessed model input
│   ├── E07303.json                     # provenance and processing metadata
│   └── prepare_georgia_example.py      # deterministic preprocessing
├── scripts/
│   ├── config/                         # model and training configuration
│   ├── data/                           # waveform preprocessing utilities
│   ├── datasets/                       # manifest and runtime-resource loaders
│   ├── evaluation/                     # performance and operating-point metrics
│   ├── interpretability/               # output-analysis utilities
│   ├── models/                         # model and component modules
│   ├── runners/                        # distributed training runner
│   └── tasks/                          # training task interface
├── prepare_data.py                         # waveform-to-manifest preprocessing
├── prepare_phecodex.py                     # ICD-to-PheCodeX mapping
├── train.py                                # training entry point
├── quick_start.py                          # single-ECG inference
├── evaluate.py                             # cohort evaluation
└── LICENSE                                 # GNU AGPL v3 license
```

## Installation

Create a Python environment with PyTorch support appropriate for the local
hardware, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The code was tested on Ubuntu 22.04 with Python 3.10 and PyTorch 2.11.0.
Required packages are listed in `requirements.txt`. CUDA is optional for ECG
preprocessing and single-ECG inference; training and large-cohort evaluation
benefit from CUDA-capable GPUs.

The text encoder uses
[`emilyalsentzer/Bio_ClinicalBERT`](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT).
The files are downloaded on first use unless already present in the local model
cache.

Download the ExFMECG model weights from the
[Zenodo record](https://doi.org/10.5281/zenodo.22110788), then set their local
path for the commands below:

```bash
export EXFMECG_CHECKPOINT=/path/to/ExFMECG_model_weights.pth
```

## Input preparation

ExFMECG receives a 10-second, 12-lead ECG in millivolts with shape
`(12, 1024)`. Leads must be supplied in the standard order
`I, II, III, aVR, aVL, aVF, V1-V6`. `prepare_data.py` loads NumPy, MATLAB or
WFDB records, resolves lead orientation, converts voltage units, centre-crops
or zero-pads to 10 seconds, resamples to 1,024 points and writes one array per
ECG together with a streaming JSONL manifest.

Input metadata may be CSV, JSON or JSONL. The default column names are:

| Column | Description |
|---|---|
| `study_id` | unique ECG identifier |
| `ecg_path` | waveform path, relative to `--waveform-root` or absolute |
| `age` | age at ECG acquisition |
| `sex` | `female`, `male` or `unknown` |
| `sampling_rate` | source sampling rate in Hz |
| `unit` | `mV`, `uV` or `V` |
| `dataset_type` | training task code; use `0` for custom inference |
| `label_list` | positive labels separated by `#`, or a JSON list |
| `evaluable_labels` | optional labels that can be evaluated for this ECG |
| `report` | ECG report text used during training |

```bash
python prepare_data.py \
  --metadata metadata.csv \
  --waveform-root /path/to/raw/waveforms \
  --output-dir data/processed
```

Use `data/processed/manifest.jsonl` for training or evaluation. If all records
share a sampling rate or unit, use `--default-sampling-rate` and
`--default-unit` instead of metadata columns.

### Optional label mapping

Record-level ICD codes can be mapped with the official PheCodeX flat files.
Download
`phecodeX_ICD_CM_map_flat.csv` and `phecodeX_ICD_WHO_map_flat.csv` from the
[PheCodeX repository](https://github.com/PheWAS/PhecodeX), then run:

```bash
python prepare_phecodex.py \
  --diagnoses diagnoses.csv \
  --cm-map /path/to/phecodeX_ICD_CM_map_flat.csv \
  --who-map /path/to/phecodeX_ICD_WHO_map_flat.csv \
  --output mapped_phecodex.csv
```

`diagnoses.csv` should contain `record_id`, `icd_code` and `vocabulary_id`.
Column names can be changed with command-line options. The output contains one
row per record-PheCodeX mapping.

## Runtime resources

`assets/task_labels.json` contains the ordered task labels associated with the
model weights. `assets/concepts/` contains ordered definitions and
model-compatible statistics used by the released implementation. Training
loads the corresponding targets from the directory specified by
`EXFMECG_CONCEPT_MATRIX_ROOT`; the expected format is implemented by the
concept-matrix loader under `scripts/datasets/`.

## Training

The provided configuration defines the model and training objectives.
Prepare each dataset in the manifest format shown above.
Each record uses `dataset_type` to select an ordered label group from
`assets/task_labels.json`. When adapting the code to a new cohort, update the
task mapping, ordered labels and concept-target routing together.

Set the manifest, waveform and concept-matrix locations, then launch distributed
training. The number of processes and batch size can be adjusted for the
available hardware:

```bash
export EXFMECG_TRAIN_MANIFEST=/path/to/train/manifest.jsonl
export EXFMECG_ECG_ROOT=/path/to/train
export EXFMECG_CONCEPT_MATRIX_ROOT=/path/to/concept_matrices

torchrun --nproc_per_node=4 train.py \
  --options run.batch_size_train=128 run.output_dir=output/exfmecg
```

Training parameters are defined under `scripts/config/train/`; model settings
are defined under `scripts/config/model/`.
Checkpoints and the resolved configuration are written below the configured
output directory. Resume training with
`run.resume_ckpt_path=/path/to/checkpoint.pth`.

## Inference

The same checkpoint can be queried directly with disease, phenotype or outcome
descriptions. No target-specific output head is fitted for this query-based
inference. The example uses a public ECG from the PhysioNet/Computing in
Cardiology Challenge 2020 dataset. Provenance, licensing and deterministic
preprocessing are documented in `examples/README.md`.

```bash
python quick_start.py \
  --checkpoint "$EXFMECG_CHECKPOINT" \
  --ecg examples/E07303.npy \
  --age 81 \
  --sex female \
  --query "atrial fibrillation" \
  --query "sinus rhythm" \
  --top-concepts 10
```

The script returns one probability for each query. Use `--top-concepts` to
include the highest-probability binary concepts. Output is printed as JSON.

## Cohort evaluation

Queries are provided as a JSON list. A simple list uses the same text for the
model prompt and the manifest label:

```json
["heart failure", "chronic obstructive pulmonary disease"]
```

If the model prompt and evaluation label differ, use explicit `name` and
`prompt` fields:

```json
[
  {"name": "COPD", "prompt": "chronic obstructive pulmonary disease"}
]
```

Run query-based evaluation on a prepared manifest:

```bash
python evaluate.py \
  --checkpoint "$EXFMECG_CHECKPOINT" \
  --manifest data/processed/manifest.jsonl \
  --ecg-root data/processed \
  --queries queries.json \
  --output-dir output/evaluation \
  --bootstrap 1000
```

The evaluator writes sample-level predictions and standard discrimination and
operating-point metrics. Bootstrap confidence intervals, threshold selection
on a separate manifest and predefined thresholds are supported through
command-line options. Large manifests are processed in batches.

## Output analysis

Entry points and supporting utilities for examining ordered model outputs are
included in this repository.

## External data

External datasets retain their original access conditions and are not
redistributed by this repository. Users must obtain any required data from the
original providers.

## Acknowledgements

The implementation builds on the software organization of
[InstructBLIP](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip)
and earlier ECG-language modeling work in
[KED](https://github.com/control-spiderman/ECGFM-KED).

## License

The source code in this repository is distributed under the GNU Affero General
Public License v3.0. See `LICENSE`. Third-party models and datasets retain their
original licenses and access conditions.
