# Analysis code for the revised manuscript

This archive contains the analysis scripts and frozen settings referenced by the manuscript. It contains **no participant recordings or held-out predictions**. The required F-TRACT data are subject to EBRAINS access terms; CUCR recordings require permission from the original data holders. The HCP structural inputs must also be obtained separately.

## What to run

| Manuscript analysis | Entry point |
|---|---|
| Primary process-versus-endpoint prediction | `scripts/run_main_prediction.py` |
| Full five-repeat, five-fold complexity-threshold analysis (T30–T70) | `scripts/run_sensitivity.py` (step 2) |
| Corrected significance test on saved held-out predictions | `scripts/run_linked_inference.py` |
| Prediction controls | `scripts/run_controls.py` |
| Cohort/domain comparison | `scripts/run_domain_classification.py` |
| Feature-displacement simulation | `scripts/run_insilico.py` |
| CUCR response detection | `scripts/run_cucr_detection.py` |
| Anatomical-circuit enrichment (Fig. 3) | `scripts/run_anatomical_enrichment.py` |

The enrichment entry applies Holm correction across the four F2 circuits (language/arcuate, thalamocortical, motor, visual). `PS_SCREEN=S14` selects the S14 sensitivity screen. The frozen whitelist, registries, and saved task-C outputs are authorized workspace inputs and are not bundled. The threshold entry re-runs T30–T70 under the same five-repeat × five-fold protocol; per-threshold p values are computed by the linked inference entry point.

`config/final_analysis.json` identifies the code, model support rule, cross-validation and statistical settings for the revised manuscript. The original 0-100 ms CUCR batch code retains its historical filename; the earlier 0-50 ms exploratory routine is not included.

Create the pinned environment using `environment.yml` and use single-threaded BLAS for numerical reproduction. With **no data**, these commands verify the package structure and the corrected statistical implementation:

```bash
python scripts/check_release.py
python scripts/run_linked_inference.py --self-test
python scripts/run_main_prediction.py --check
python scripts/run_sensitivity.py --check
python scripts/run_anatomical_enrichment.py --check
```

The revised inference implementation reproduced all five 100,000-draw threshold p values from the existing saved full-cross-validation predictions. The reported ROI-linked and observation-linked p values are produced by `src/inference/portable_inference.py` through `scripts/run_linked_inference.py`. To run it on authorized predictions, use `scripts/run_linked_inference.py --help` and supply the prediction file, stimulation-region mapping, draw count and output location. The main test shares one exchange decision per stimulation region across repeated predictions; a per-observation sensitivity option is separate. Threshold training also writes an internal 10,000-draw diagnostic; the manuscript's 100,000-draw p values come from the separate inference entry point.

The data-dependent entry points accept `--execute --data-root /authorized/workspace` and require the original inputs in a writable copy of that workspace. The source archive does not distribute participant recordings or automatically update GitHub or Zenodo.

## Terminology used in the original simulation scripts

`personalized` denotes the two targets assigned to the **limbic** group; `conventional` denotes the left and right **parahippocampal** targets. `control` denotes seven targets selected by the historical early-sensory label rule, including nonvisual regions; it should not be read as a purely visual control group. The code maps these groups before computing the final measure, **SR** (cosine similarity to the reference feature vector). Earlier **CR/SI** outputs belong to a previous exploratory formulation and are **not** the SR analysis retained in the revised manuscript. SR describes feature-space proximity, not clinical benefit or target superiority.

`LICENSE` contains the MIT terms. The CUCR scripts depend on FSL and FreeSurfer as well as authorized recordings; see their source for input conventions.
