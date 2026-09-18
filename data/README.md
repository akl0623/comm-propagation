# Data

This repository **does not include any raw data files**. Before running the
code, obtain the datasets listed below and place them according to the pinned
configuration in `config/`.

| Data | Contents | How to obtain | Used for |
|---|---|---|---|
| **F-TRACT group data** | Group-averaged structural connectivity (streamline count / tract length), cortico-cortical evoked potential (CCEP) onset latency, and stimulation-response probability matrices (360 ROIs, HCP-MMP1 atlas) | **EBRAINS** (f-tract.eu, F-TRACT project open data; registration and data agreement required) | Main analysis (Task C), path search, four communication metrics, sensitivity analyses |
| **CUCR data** | SEEG stimulation-response data from the second center (electrode coordinates, CCEP onset, response probability) | **Contact the original data holders** (collaborating hospital; subject to patient-privacy constraints, not distributed with this repository) | CUCR secondary analysis (`cucr_secondary_analysis_v1`; requires authorization) |
| **HCP atlas** | HCP-MMP1.0 360-ROI atlas, Yeo7 network parcellation, fsaverage ROI centroid distance matrix | **HCP** (humanconnectome.org) public atlas | Anatomical whitelist, Yeo7 network constraints (B1/B6 screens), Euclidean-distance guidance for navigation efficiency |
| **NER patient data** | Patient-level SEEG path data (Timone Hospital, Marseille) | Not released; see the original paper supplementary material for access requests | Task B domain classification / E2/E3 (steps explicitly SKIP when data are absent) |

## Directory Layout

```
data/ftract/    # F-TRACT derived matrices (SC, tract length, onset delay, probability)
data/cucr/      # CUCR derived matrices (median_onset_time_matrix, response_probability_matrix)
data/atlas/     # MMP1 -> Yeo7 mapping, anatomical whitelist, fsaverage distance matrix
data/ner/       # README only (patient data not released)
```

Scripts read input paths from the pinned configuration; if you need to relocate
data, set `PIPELINE_ROOT` (see the repository README) rather than editing
analysis logic.

## Privacy and License

- Redistribution of F-TRACT derived matrices requires compliance with the
  F-TRACT data agreement.
- CUCR / NER patient data **must not** be included in the public repository;
  related steps explicitly skip when data are absent.
