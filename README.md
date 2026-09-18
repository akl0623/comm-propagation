# From Routing to Diffusion: Process-Aware Modeling of Stimulation-Evoked Propagation in Human SEEG Networks

**Authors:** Yufei Yuan, Yinyu Lan, Simeng Wu, Xiangsen Liu, Xiao Zhang
**Affiliation:** School of Medicine and Health, Harbin Institute of Technology

---

## Overview

This repository contains analysis code for a process-aware framework that reconstructs
physiologically constrained candidate propagation paths from stimulation-evoked SEEG
responses and uses a hybrid routing–diffusion communication model to characterize
stage-wise communication dynamics.

## Repository Structure

```
├── src/
│   ├── path_search/              # Propagation path reconstruction
│   ├── communication_metrics/    # Four graph-theoretic communication metrics
│   ├── features/                # Endpoint features, Legendre expansion, PCES screening
│   ├── models/                   # Adaptive elastic-net, domain classifier
│   ├── controls/                 # OrderShuffle, MatchedRandom, SCNull, LatencyNull
│   ├── insilico/                 # Feature-displacement simulation, SR analysis
│   ├── cucr_preprocessing/       # CUCR electrode co-registration and CCEP processing
│   └── utils/                    # Cross-validation, bootstrap statistics
├── scripts/                      # Entry-point scripts for each analysis stage
├── config/                       # Frozen hyperparameters and screening thresholds
├── data/                         # Data acquisition instructions (no raw data included)
├── environment.yml               # Conda environment specification
└── requirements.txt             # Pip freeze (fallback)
```

## Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate tnsre

# Set single-threaded BLAS for reproducibility
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

Python 3.12.2, numpy 2.5.1, scikit-learn 1.8.0, scipy, pandas 3.0.0.
All random seeds are fixed (random_state=42). See `config/hyperparameters.json`
for the complete frozen hyperparameter specification.

## Data Availability

This repository does **not** include raw data. The three datasets used in this study
must be obtained separately:

### 1. F-TRACT (epilepsy cohort, group-level responses)
- Source: EBRAINS Knowledge Graph
- URL: https://ebrains.eu/
- Reference: M. Jedynak et al., "F-TRACT: a probabilistic atlas of anatomo-functional
  connectivity of the human brain (F-TRACT_P_01_v2307) [Data set]," EBRAINS, 2023.

### 2. CUCR (clinically uninvolved cortical reference)
- Source: Contact the authors of the original CUCR dataset
- Reference: S. Parmigiani et al., "Simultaneous stereo-EEG and high-density scalp EEG
  recordings to study the effects of intracerebral stimulation parameters,"
  *Brain Stimulation*, vol. 15, no. 3, pp. 664–675, May 2022.

### 3. HCP structural connectivity (group-averaged SC matrix)
- Source: BALSA (Washington University)
- URL: https://balsa.wustl.edu/
- Preprocessing reference: B. Q. Rosen and E. Halgren, "A Whole-Cortex Probabilistic
  Diffusion Tractography Connectome," *eNeuro*, vol. 8, no. 1,
  p. ENEURO.0416-20.2020, Jan. 2021.

After downloading, set the environment variable:
```bash
export PIPELINE_ROOT=/path/to/your/data/workspace
```

## Reproduction

The entry-point scripts in `scripts/` expect `PIPELINE_ROOT` to point to a directory
containing the downloaded datasets in the expected subdirectory structure.

Key parameters are frozen in `config/`:
- `config/hyperparameters.json`: ElasticNet grid, CV scheme, bootstrap seeds
- `config/screening_params.json`: PCES screening thresholds (variance ≥ 0.4155,
  DDVS Jaccard ≥ 0.80, Ksel = 128, SC density = 0.25)

## License

This project is licensed under the MIT License.
