# Finish the dementia MRI project: code and practical walkthrough

**Review: 28 October 2026. Start date: 26 October 2026.**

This package contains implementation code, not trained OASIS models or performance results. It supports a smaller review-ready version and the full proposed experiment. No clinical performance or completion time is guaranteed before your data and computer are tested.

## What you can realistically finish

**First target:** OASIS-1 MRI → 16 slices → frozen ResNet-18 → mean pooled features → trained logistic regression → held-out evaluation → local Streamlit demonstration. An age-only baseline is included.

**Full target:** CNN-only, ViT-only and fusion, each with 4, 8 and 16 slices, plus the same age baseline. Full mode is implemented, but ViT extraction is the most expensive optional stage for your review deadline.

On a normal PC, the small classifier is usually much cheaper to fit than extracting image features. The encoders are pretrained and frozen: you are not training ResNet or ViT from scratch. CPU execution is supported. Actual speed depends on CPU, RAM, storage and the cohort size. Use the five-participant pilot to decide; do not rely on a guessed number of hours. A machine with 8 GB RAM may need other apps closed and batch size 1; 16 GB gives more room. These are planning suggestions, not measured minimum specifications.

Present the CNN baseline as your completed first implementation if ViT is unfinished. Do not claim fusion results before running them. Your proposed research objective can remain the CNN–ViT comparison.

## Files and how they work

| File | Purpose |
|---|---|
| `config.yaml` | Your paths, geometry, slices and experiment settings |
| `pipeline.py` | Dataset matching, preprocessing, encoders, training, evaluation and prediction |
| `app.py` | Local Streamlit interface using the same prediction code |
| `requirements.txt` | Python dependencies; install torch/torchvision separately |
| `tests/test_pipeline.py` | Synthetic checks; never use their results as project accuracy |
| `VALIDATION.md` | What was actually tested before delivery |

The original OASIS data stays on your PC. The app selects local processed samples; it does not upload data or make medical decisions.

## Step 1 — Extract this ZIP and open the folder

Extract to a simple path such as `D:/dementia_starter`. Open that folder in VS Code. Open its terminal and make sure `pipeline.py` is in the current folder.

**Why:** all commands below are run from the project folder, and the app can then import the pipeline.

**Done when:** `dir` on Windows shows `pipeline.py`, `config.yaml`, `app.py` and `README.md`.

## Step 2 — Create an isolated environment

Use Python 3.11 if available. In Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip freeze > requirements-lock.txt
```

The CPU command is the straightforward route for a normal PC. For NVIDIA acceleration, choose the matching torch/torchvision command from https://pytorch.org/get-started/locally/ instead. Do not mix independently chosen torch and torchvision versions.

Throughout the rest of this README, **`python` means the virtual environment's Python**. Either replace it with `.\.venv\Scripts\python.exe` or activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import torch, torchvision, nibabel, sklearn; print('Imports OK'); print('CUDA:', torch.cuda.is_available())"
```

If activation is blocked by PowerShell policy, use the explicit executable path; no policy change is necessary. On macOS/Linux the equivalent interpreter path is `.venv/bin/python`, with the install command appropriate to that operating system.

**Why:** the environment keeps dependencies together. `CUDA: False` is expected for the CPU build and does not mean the project failed.

## Step 3 — Point the code at the downloaded dataset

Extract all OASIS archives first. Keep original folder structure and paired `.hdr` and `.img` files together. Export the clinical spreadsheet to CSV if only an XLS file is available; preserve ID, Age and CDR and leave missing values blank.

Edit these values in `config.yaml`:

```yaml
data_root: 'D:/OASIS1'
metadata_csv: 'D:/OASIS1/oasis_cross-sectional.csv'
output_dir: 'outputs'
device: cpu
batch_size: 2
```

Paths above are examples. Use your actual location. Forward slashes work on Windows. Set batch size to 1 if RAM is limited.

Default metadata columns are `ID`, `Age`, `CDR`. The code prints the actual columns; change `id_column`, `age_column` or `cdr_column` if needed. It expects original session IDs such as `OAS1_0001_MR1`; it deliberately stops if your metadata uses a different schema.

Default file search:

```yaml
image_glob: '**/PROCESSED/T88_111/*_t88_masked_gfc.hdr'
```

This selects the original brain-masked atlas-aligned intensity MRI, not RAW acquisitions or the `fseg` tissue-label volume. Original OASIS-1 provides Analyze `.hdr`/`.img` pairs. If you have only converted NIfTI files, verify how they were converted before changing the pattern. Do not point it at arbitrary MRI JPEGs.

**Why:** consistent image products and correct clinical matching matter more than adding another neural network.

## Step 4 — Build the participant table and fixed split

```powershell
python pipeline.py prepare
```

The command:

1. Finds selected MRI files and verifies their paired images exist.
2. Matches session IDs to the clinical table.
3. Keeps age >=60 and CDR in {0, 0.5, 1, 2}; logs missing values/exclusions.
4. Keeps the earliest eligible session per participant.
5. Assigns label 0 to CDR=0 and label 1 to CDR>0.
6. Checks exact duplicated image bytes under different IDs.
7. Saves a stratified 80% development / 20% test split.
8. Assigns development CV folds, reducing from five only if needed for class counts.

Files produced:

- `outputs/manifests/inventory.csv`
- `outputs/manifests/participants.csv`
- `outputs/manifests/exclusions.csv`
- `outputs/manifests/preparation.json`

**Why:** each participant is one independent example. Their slices are not separate patients. Saving the split prevents accidental changes between experiments.

The command refuses to overwrite an existing manifest. If you discover an actual metadata problem before evaluation, correct it and start a new output directory. Do not keep changing the split to improve accuracy.

**Done when:** you see counts for both groups in development and testing, and the selected MRI paths and CDR matches are correct in a manual spot-check.

## Step 5 — Verify image orientation and choose slices

```powershell
python pipeline.py inspect
```

This examines the first development participant. For another development participant:

```powershell
python pipeline.py inspect --subject OAS1_XXXX
```

Replace `OAS1_XXXX` with a real development participant ID from `participants.csv`.

Open the generated `outputs/qc/..._planes.png` and `..._geometry.json`. Inspect several development participants. The preview labels array axes rather than pretending anatomical directions are already verified. Compare the image with the OASIS documentation and a medical-image viewer; ask your guide to confirm if you are unsure. Analyze headers alone may not reliably specify anatomical orientation.

Then edit `geometry` in the configuration:

- `canonical`: keep false for the original convention if verified; if changed, inspect again.
- `axial_axis`: the verified axial array axis (often 2 for the intended original product; verify).
- `expected_shape`, `expected_zooms`: use the inspected geometry.
- `expected_affine`: copy the full 4×4 `effective_affine` from the JSON.
- `notes`: describe how you verified the plane and common image product.
- `verified`: set true **after** these checks.

Choose a brain-containing axial range and 16 unique positions. For example, **only if development previews support raw indices 40–130**, this prints a candidate list:

```powershell
python -c "import numpy as np; print(np.linspace(40,130,16,dtype=int).tolist())"
```

Paste the resulting list into `slice_indices`. The example bounds are not universal anatomical landmarks. Do not accept them blindly. The config already contains nested 4- and 8-slice subsets as positions within this 16-index list.

**Why:** wrong axes, blank slices and incompatible grids produce meaningless features. The code intentionally requires this one manual geometry decision before extraction. Reorientation does not perform image registration.

If a later scan fails geometry or quality checks, stop and inspect it using the same label-blind criteria. Do not relax checks just to make an unsuitable image pass. Record objective exclusions; rebuild a new study manifest before model development if exclusions change the cohort. Once test results are seen, do not remove hard cases to improve results.

## Step 6 — Understand the preprocessing code

`slices_array()` loads and validates the MRI, uses nonzero voxels as the mask for the intended already-brain-masked product, clips at the 1st/99th brain-intensity percentiles and scales to [0,1]. It keeps background zero and extracts the configured slices.

`tensor_inputs()` pads each slice to a square, resizes it to 224×224, repeats grayscale into three channels, and applies ImageNet channel normalisation. This is a documented MRI adaptation of the pretrained input recipe; it avoids centre-cropping away anatomy.

**Why:** the encoders need a fixed tensor shape and intensity handling. Their inputs have shape `[number_of_slices, 3, 224, 224]`. Repeated grayscale channels do not add new image information.

## Step 7 — Run a five-participant CPU pilot

```powershell
python pipeline.py extract --encoder cnn --limit 5
```

The first run downloads the official pretrained weights, so it needs internet access. The code prints model load/download time separately, then seconds per participant. Its default is float32, small batches, inference mode and one encoder in memory.

Estimate remaining work using:

**remaining participants × observed median seconds per participant**

For example, if your measured median were 12 seconds and 150 people remained, extraction would be approximately 30 minutes plus overhead. This is arithmetic illustrating the estimate, not a runtime prediction for your PC.

**Why:** CPU capability is best established by your actual pilot. The ResNet weights stay frozen. The expensive part is computing features, not learning millions of model weights.

**Done when:** five `outputs/features/cnn/<participant>.npz` files exist and the log reports shape `(16,512)` without errors. Rerunning skips compatible completed caches, so interruptions do not erase progress.

## Step 8 — Choose baseline or full mode BEFORE final evaluation

### Deadline-first path: default baseline

```powershell
python pipeline.py extract --encoder cnn
python pipeline.py train
```

This completes CNN extraction for development participants, then trains **CNN-16 and an age-only baseline**. It does not require ViT caches.

### Full comparison path: only if the pilot leaves enough time

Before training, time ViT too:

```powershell
python pipeline.py extract --encoder vit --limit 5
```

If feasible, complete both caches and use full mode:

```powershell
python pipeline.py extract --encoder cnn
python pipeline.py extract --encoder vit
python pipeline.py train --full
```

This trains all nine MRI combinations plus the age baseline. Choose one training path per output directory. `train` locks the run, and refuses to silently replace an already completed study.

To start the full study after a baseline run, create `config_full.yaml` with a new `output_dir`, then use `--config config_full.yaml` before every subcommand. The same seed and eligibility should preserve participant assignments if the cohort is unchanged; explicitly verify this. Do not portray a familiar test set as fresh validation after inspecting its earlier results. For your immediate review, prioritise the baseline rather than repeatedly restarting.

**What training actually does:**

```python
# One row per participant, after averaging that person's selected slices.
X_cnn = cnn_slice_features.mean(axis=1)   # N×512
X_vit = vit_slice_features.mean(axis=1)   # N×768
X_fusion = np.concatenate([X_cnn, X_vit], axis=1)  # N×1280
```

The logistic-regression pipeline learns from these rows and their CDR labels. It fits the scaler only inside training folds, tunes C over `[0.001,0.01,0.1,1.0]`, and refits the selected classifier using all development participants. The small C values provide stronger regularisation.

The age-only baseline asks whether age alone predicts the labels surprisingly well. It is not the app's MRI model.

**Output:** `outputs/results/development_comparison.csv`, per-model CV tables, saved `.joblib` models, `models/selected.json`, `models/experiments.json` and `models/lock.json`.

The selected app candidate uses highest development CV AUC; exact ties prefer simpler representation and fewer slices. It is not automatically the cheapest model, and the selected CV score is not a final unbiased performance estimate. Use measured costs to discuss trade-offs.

## Step 9 — Extract final-test features with the locked configuration

Baseline:

```powershell
python pipeline.py extract --encoder cnn --split test
```

Full mode additionally needs:

```powershell
python pipeline.py extract --encoder vit --split test
```

**Why:** test extraction is allowed only after training has locked the geometry, slices, weights, classifier protocol and participant manifest. The code rejects stale feature caches and changed configurations/versions.

Do not update packages halfway through the run. If packages change, restore the saved environment or create and document a new study run.

## Step 10 — Evaluate once

```powershell
python pipeline.py evaluate
```

This computes participant-level ROC-AUC, balanced accuracy, sensitivity, specificity, F1, confusion counts and stratified participant-bootstrap confidence intervals. Full mode also compares fusion AUC against each encoder using paired resampling.

Files:

- `results/test_metrics.csv`
- `results/test_predictions.csv`
- `results/test_roc.png`
- `results/paired_auc_differences.csv` (only populated for full comparisons)
- `results/evaluation_notes.json`

**Why:** this evaluates previously unseen participants. The intervals describe uncertainty conditional on these fitted models; they do not prove generalisation to other hospitals. Comparisons are descriptive and not corrected for testing multiple hypotheses.

The command refuses to rerun after successful evaluation. Read the saved results rather than repeatedly changing the model against the test set. Poor performance is a finding to analyse, not a reason to shuffle participants until accuracy improves.

## Step 11 — Measure real CPU cost

For a short review benchmark:

```powershell
python pipeline.py benchmark --subjects 2 --repeats 2
```

This measures actual image processing, not cached feature loading. It records encoder extraction time on development samples, model loading time, peak allocated GPU memory if applicable, and CPU resident memory after each run. CPU resident memory **is not peak RAM** and is labelled accordingly.

It also measures complete prediction including sequential model loading for every trained MRI configuration on one development sample. The sample is deliberately small for the review; report that limit. Full nine-model benchmarking takes longer than baseline benchmarking.

Files: `encoder_timings.csv`, `end_to_end_timings.csv` and `benchmark_environment.json` under `results/`.

**Why:** frozen encoders still require computation. The fusion system runs both branches, so a prediction gain has to be considered with its cost.

## Step 12 — Run a prediction directly

```powershell
python pipeline.py predict --image "D:/OASIS1/ACTUAL/PATH/scan_t88_masked_gfc.hdr"
```

Replace the example with a real path from your manifest. The matching `.img` must be in the same folder. Output includes model name, score, predicted group and elapsed time including model loading.

`predict_image()` does not read CDR or age to make the MRI prediction. It reloads the selected pipeline, processes the supplied MRI, extracts features, pools them and calls the trained classifier.

**Why:** this verifies the usable application path, rather than only running an experiment on cached features.

## Step 13 — Start the frontend

```powershell
python -m streamlit run app.py
```

Open the local address printed in the terminal. Choose a sample and click **Analyse MRI**. It shows slices, predicted CDR group, score and measured time. Development/test comparison tables are in expandable panels. Before final evaluation, the sample selector hides held-out participants.

**Why:** this is what you demonstrate at your review. The exact same preprocessing function is reused. The score is not a clinically validated probability of dementia.

This version intentionally uses supported local processed samples. An arbitrary screenshot or a raw hospital scan would not meet its input assumptions. Model loading is included in demo latency; subsequent optimisation can keep encoders in memory if RAM allows.

## Suggested deadline schedule

| Date | Priority | Evidence to save |
|---|---|---|
| 26 October | Install, match data, split, verify geometry, CNN pilot | Participant counts, QC montage, pilot timing |
| 27 October first work session | Complete CNN cache, train baseline, lock/evaluate | Real result CSVs and ROC plot |
| 27 October next session | Run app, benchmark, prepare slides and recording | Working demo, limitations, next steps |
| 28 October | Rehearse and present completed work | Local app plus backup screenshots/video |

Only add ViT/full mode if its measured extraction time fits with enough room for testing and presentation. Start with one good end-to-end path. Do not spend the last evening changing the architecture.

## What to explain in your review

“We match OASIS-1 MRI scans to recorded CDR labels and separate participants before training. A frozen pretrained ResNet-18 converts each selected MRI slice into features. We average features across slices and train a regularised logistic-regression classifier to predict the CDR group. We evaluate on held-out participants and demonstrate the pipeline through a local interface. The full study compares CNN, ViT and fused features under different slice budgets.”

If only baseline mode is complete, state that the ViT/fusion comparison is the next stage. Fill your slides only with values from your own output CSVs. If no evaluation is complete, show verified data/QC progress instead of fabricated accuracy.

## Common problems

| Message / symptom | Action |
|---|---|
| No intended MRI files found | Extract nested archives; check data_root and the real directory layout |
| Expected original session IDs | Check clinical file and ID column; do not strip visit suffixes before matching |
| Geometry not verified | Finish inspect; record notes, axis, affine and slice choices |
| Different shape/affine | Stop and verify image product/common grid; do not suppress validation |
| Empty or almost empty slice | Inspect brain coverage; decide slice range using development data only |
| Stale cache / changed lock | Restore matching config/environment; avoid mixing runs |
| CPU extraction slow | Batch size 1–2, close other apps, finish CNN baseline first |
| CUDA unavailable | CPU mode is valid; a CPU PyTorch install does not use the NVIDIA GPU |
| Windows process/worker issue | This package deliberately uses one classifier worker and no DataLoader workers |
| Pretrained download fails | Check internet/proxy; retry; successful feature files are retained |
| Training already locked | Keep completed results; choose a new output_dir for a different protocol |
| App differs from saved evaluation | Compare model, selected slices and preprocessing; verify same processed input |

## Sources

- OASIS-1: https://sites.wustl.edu/oasisbrains/home/oasis-1/
- Original release format: https://bpb-us-e2.wpmucdn.com/sites.wustl.edu/dist/6/4383/files/2024/03/oasis_cross-sectional_facts-bcc7a002dfb104f4.pdf
- ResNet-18: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
- ViT-B/16: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html
- Leakage prevention: https://scikit-learn.org/stable/common_pitfalls.html
- Orientation: https://nipy.org/nibabel/image_orientation.html

The choice of cohort, classifier, slice budgets and deadline-first baseline is our project design, not a claim that these sources prescribe this exact study.
