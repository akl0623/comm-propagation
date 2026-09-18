# CUCR Preprocessing

This directory contains the CUCR (clinically uninvolved cortical reference) data
preparation code: electrode co-registration, electrode-to-parcel assignment, and
CCEP (cortico-cortical evoked potential) detection. These steps turn raw CUCR
SEEG recordings into the parcel-level inputs consumed by the main analysis
pipeline (see `data/README.md` for the dataset itself).

**Note:** CUCR preprocessing is a *data preparation* step, not part of the core
analysis pipeline. It is documented here for completeness and reproducibility;
the frozen results used in the study were produced by this code.

## Pipeline Overview

| Step | Component | Script |
|---|---|---|
| 1. Structural MRI processing | coregistration | `batch_deal_smri.sh` |
| 2. Electrode coordinates | RAS -> LIA conversion | `ras_to_t1w_lia.py` |
| 3. Electrode-to-parcel mapping | brain region assignment | `electrode_to_parcel_mapping.py`, `batch_electrode_to_parcel_mapping.py` |
| 4. CCEP detection | z-score, onset detection, trial statistics | `ccep_detection_legacy.py`, `ccep_response_onset_statistics.py` |

### Step 1: Structural MRI processing (shell)

`batch_deal_smri.sh` loops over subjects sub-02..sub-36 and performs, per subject:

```bash
mri_info $INPUT_T1                          # orientation check (FreeSurfer)
fslreorient2std $INPUT_T1 $REORIENTED_T1    # reorient to standard (FSL)
flirt -in $REORIENTED_T1 -ref MNI152_T1_1mm.nii.gz \
      -omat struct2mni.mat -searchrx -30 30 -searchry -30 30 -searchrz -30 30
fslorient -swaporient $MNI_REGISTERED_T1    # RAS orientation (FSL)
recon-all -i $MNI_REGISTERED_T1 -s sub-XX_recon -all          # FreeSurfer
mri_surf2surf --hemi lh --srcsubject fsaverage --trgsubject sub-XX_recon \
      --sval-annot lh.HCP-MMP1.annot --tval sub-XX_lh.HCP-MMP1_on_MNI152_ICBM2009a_nlin.annot
mri_surf2surf --hemi rh --srcsubject fsaverage --trgsubject sub-XX_recon \
      --sval-annot rh.HCP-MMP1.annot --tval sub-XX_rh.HCP-MMP1_on_MNI152_ICBM2009a_nlin.annot
```

This projects the HCP-MMP1 parcellation onto each subject's cortical surface in
MNI space; the resulting per-subject annot files are consumed by Step 3.

### Step 2: Electrode coordinates

`ras_to_t1w_lia.py` converts electrode positions from RAS world coordinates
(BIDS `space-MNI152NLin2009aSym_electrodes.tsv`) to LIA world coordinates via the
T1w affine matrix (nibabel).

### Step 3: Electrode-to-parcel mapping

`electrode_to_parcel_mapping.py` (single subject, sub-01) and
`batch_electrode_to_parcel_mapping.py` (sub-02..sub-36):

- computes area-weighted centroids of every HCP-MMP1 parcel from the
  FreeSurfer annot + white surface geometry;
- assigns each good-status electrode to the nearest parcel centroid;
- averages all good electrodes within a parcel per trial, producing
  `sub-XX_task-ccepcoreg_run-YY_parcel_seeg_data.npy`
  (trials x parcels x timepoints);
- writes `electrode_to_parcel_mapping.csv` and `active_parcels.csv`.

Requires `FREESURFER_SUBJECTS_DIR` to point at the FreeSurfer subjects directory
(the one containing `sub-XX_recon`):

```bash
export FREESURFER_SUBJECTS_DIR=/path/to/freesurfer/subjects
```

### Step 4: CCEP detection

`ccep_detection_legacy.py` and `ccep_response_onset_statistics.py` implement the per-trial CCEP detection
on the parcel-level data:

- baseline window [-300, -50] ms, response window [0, 100] ms
  (time axis linspace(-300, 700, 1001));
- per-trial/per-channel z-score against the trial baseline (std <= 0 -> 0);
- significance: |z| > 5 in the response window;
- onset = first threshold crossing; peak = argmax from the crossing;
- pooled across the 36 subjects x 323 runs into the 360x360 median onset
  matrix (`median_onset_time_matrix_251114.txt`), the latency input of the
  CUCR path engine.

## External Dependencies

- **FSL 6.0+** (`fslreorient2std`, `flirt`, `fslorient`)
- **FreeSurfer 7.x** (`mri_info`, `recon-all`, `mri_surf2surf`)
- **MRtrix3** is not required by these scripts
- Python: numpy, pandas, nibabel, scipy (see `environment.yml`)

The Python scripts are thin wrappers/orchestrators around the FSL/FreeSurfer
command-line tools; no additional logic is included.
