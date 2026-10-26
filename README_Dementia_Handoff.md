# Dementia MRI project — training and frontend handoff

**Start here.** This is a runnable research project, not a trained diagnostic product. Your friend supplies the authorised OASIS-1 download. The ZIP contains the pipeline, configuration, frontend, tests and documentation. No OASIS images, trained classifiers or claimed accuracy are included.

## 1. What we are building and why

**Title:** Evaluating CNN–ViT Feature Fusion for MRI-Based Dementia Status Classification.

Input: a supported, brain-masked, atlas-aligned structural MRI. Output: a participant-level prediction of **CDR = 0 versus CDR > 0**, with a model score. CDR comes from the clinical spreadsheet; it is the training target, not an input image feature. This is not confirmation of Alzheimer's disease or a replacement for clinical assessment.

We test whether combining representations from a pretrained CNN and ViT helps compared with either alone. Fusion itself is established work; our contribution is a controlled comparison, including computation and an age-confound baseline. We cannot promise fusion will win.

**Deadline scope:** run `review`: CNN-16, ViT-16, fusion-16 and age-only. Sixteen means 16 two-dimensional cross-sections from each person's 3D MRI. It gives one fixed, manageable protocol. It is not a proven optimum. The optional `full` suite adds 4 and 8 nested slices; do not claim a slice-budget study if you run only `review`.

## 2. Files and hardware

| File | Purpose |
|---|---|
| `pipeline.py` | Preparation, validation, feature extraction, training, evaluation, timing, prediction and return ZIP |
| `app.py` | Local Streamlit frontend using the same preprocessing and saved models |
| `config.yaml` | Paths, MRI geometry, slice selection and experiment settings |
| `requirements.txt` | Non-PyTorch dependencies |
| `requirements-tested.txt` | Versions used for the supplied CPU software checks; not a GPU installation recipe |
| `tests/test_pipeline.py` | Synthetic feature/data tests |
| `tests/smoke_cpu.py` | Synthetic MRI end-to-end CNN and frontend smoke test |
| `HANDOFF_CHECKLIST.md` | What your friend must verify and return |
| `VALIDATION.md` | What was actually checked before delivery |
| `COMPLETE_CODE_GUIDE.md` | This guide plus explanations and all source code |

RTX 3050 and 16 GB system RAM are a reasonable target for frozen extraction. GPU VRAM varies by 3050 model. Start with batch size 2; reduce to 1 on CUDA out-of-memory. Encoders run separately. A GPU accelerates feature extraction; the small logistic-regression classifiers train on CPU. CPU fallback is supported but slower. Runtime depends on disk, GPU, drivers and cohort size; estimate from the pilot rather than promising a duration.

Use 64-bit Python 3.11 or 3.12. The delivered checks use Python 3.12. Keep sufficient free space for your already-extracted OASIS release, Python, model weights and outputs. First use downloads pretrained weights; arrange internet before the review.

## 3. Install (Windows PowerShell)

Extract the ZIP. Open a terminal **inside `dementia_handoff`**. These commands assume Windows; Linux/macOS use `python3 -m venv .venv` and `source .venv/bin/activate` instead.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, use Command Prompt and `.venv\Scripts\activate.bat`, or invoke `.venv\Scripts\python.exe` directly. Do not disable system security policies.

Install PyTorch and torchvision together using the command from the [official installation selector](https://pytorch.org/get-started/locally/): choose Windows, Pip, Python and a CUDA build supported by your NVIDIA driver. Run `nvidia-smi` to inspect the GPU/driver. Do not use the CPU wheel command below for the intended GPU run.

If you intentionally want CPU-only:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then install the remaining libraries:

```powershell
python -m pip install -r requirements.txt
python pipeline.py doctor --models
```

**Why:** the doctor records versions, reports CUDA and tests real pretrained CNN/ViT forward passes (512 and 768 output values). On your RTX system it should report CUDA available and the GPU name. If it reports CPU, fix the PyTorch/driver installation before the long run. A locally installed CUDA toolkit is not a substitute for a CUDA-enabled PyTorch build.

Keep this environment unchanged during extraction, training and evaluation. The pipeline detects changed versions and refuses incompatible caches/models. Record the final environment:

```powershell
python -m pip freeze > friend_environment_freeze.txt
```

## 4. Supply the correct OASIS-1 files

Use the original [OASIS-1 release](https://sites.wustl.edu/oasisbrains/home/oasis-1/) under its access agreement. You need:

1. The clinical/demographic table with `ID`, `Age`, `CDR` (CSV, XLSX or XLS).
2. For all intended participants, the processed `PROCESSED/T88_111/` **brain-masked** image product ending `_t88_masked_gfc.hdr`, plus its matching `.img` file.
3. The release documentation, to verify anatomical orientation and image products.

Do not download OASIS-2/3/4 for this implementation. Do not substitute JPG screenshots, Kaggle-repackaged slices, unmasked files or raw repeated acquisitions. Keep `.hdr` and `.img` together with the same filename stem. Extract all intended archives under one root; nested disc directories are supported.

Example (your real filenames may differ):

```text
D:/OASIS1/
  oasis_cross-sectional.xlsx
  disc1/OAS1_0001_MR1/PROCESSED/T88_111/
    OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc.hdr
    OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc.img
```

Edit only the initial path values in `config.yaml` first:

```yaml
data_root: 'D:/OASIS1'
metadata_file: 'D:/OASIS1/oasis_cross-sectional.xlsx'
output_dir: 'outputs'
batch_size: 2
device: auto
experiment_suite: review
```

Use the real metadata filename; it need not be renamed. Forward slashes avoid Windows YAML escape problems. The root must contain extracted files, not only archives.

## 5. Build and inspect the participant table

```powershell
python pipeline.py prepare
```

**What happens:** match image sessions to metadata IDs; retain age >= 60 and recorded CDR in 0, 0.5, 1 or 2; set label 0 for CDR=0 and 1 for CDR>0; keep one earliest eligible session per person; check image pairs and duplicate content; create a fixed stratified 80/20 development/test split and development CV folds.

**Why:** the unit is a person, not a slice. Slices from one person must never appear in both training and testing. Older-adult restriction reduces a simple young-versus-old shortcut, but does not remove all confounding.

Read `outputs/manifests/` including `participants.csv`, exclusions and cohort summary. Check counts for both labels, the number of matched images, and excluded reasons. Missing CDR is not CDR zero. A partial download can silently reduce the research cohort; investigate missing-file exclusions before proceeding. The script requires at least eight participants in each group, but this is a software minimum, not an adequate scientific sample-size claim.

Do not edit the generated participants CSV: it is sealed with a hash. If paths or objective exclusions need changing before experiments, correct configuration, use a new output directory, and run preparation again. Optional exclusions CSV has columns `session_id,reason`; set `exclusion_file` before preparation and document objective reasons, never model performance.

## 6. Verify anatomy and choose the 16 slices — required manual step

```powershell
python pipeline.py inspect
```

Open the `outputs/qc/*_planes.png` and corresponding geometry JSON. The montage shows each array axis at several indices. Choose a development participant from `participants.csv` and inspect additional development examples:

```powershell
python pipeline.py inspect --subject OAS1_0001
```

Replace that example ID with an actual development `subject_id` from your manifest (no `_MR1` suffix). **An array axis number is not automatically an anatomical direction.** Analyze headers may not fully establish orientation. Confirm axial anatomy using release documentation and, if needed, a knowledgeable supervisor/MRI viewer. Inspect several development scans without using their labels to optimise slices. A transverse/axial view is a horizontal cross-section of the head; sagittal is side-on and coronal is front-on.

Choose a consistent brain-containing axial range that samples relevant brain anatomy and avoids near-empty end slices. No universal index range can be safely supplied without seeing your actual image product. The following is a command template: replace all capitalised placeholders; it is not directly runnable as written.

```text
python pipeline.py configure-geometry --subject DEVELOPMENT_ID --axis VERIFIED_AXIS_NUMBER --first FIRST_VOXEL_INDEX --last LAST_VOXEL_INDEX --notes "How orientation and range were verified"
```

**What happens:** the command records measured shape, spacing and affine from that development image, records your verification notes, and generates 16 evenly spaced unique indices between the supplied bounds. All subjects use these same indices. An affine describes the mapping between voxel positions and physical space; matching shape alone is insufficient.

**Why:** consistent anatomical sampling is needed for fair comparison. The program deliberately stops until this is done. Do not simply set `verified: true` to bypass the check. Reorientation is not registration. This package relies on the supplied atlas-aligned product and does not implement raw-MRI registration or skull stripping.

```powershell
python pipeline.py audit --split development
```

Inspect the selected-slice previews and audit CSV in `outputs/qc`. Resolve inconsistent geometry, empty images or corrupt pairs before feature extraction. If a consistent release has harmless header differences, investigate and document them; do not relax the checks merely to make the command pass. A failed audit must not be silently fixed by deleting difficult participants. After test outcomes are seen, exclusions or preprocessing changes would compromise the held-out claim.

## 7. Pilot the two frozen encoders

```powershell
python pipeline.py extract --encoder cnn --split development --limit 2
python pipeline.py extract --encoder vit --split development --limit 2
```

**Frozen** means the pretrained encoder weights do not change. ResNet-18 converts each slice into 512 numerical features; ViT-B/16 converts it into 768. Their original classification heads are removed. These are general pretrained representations, not pretrained dementia diagnoses.

The MRI is normalised within its nonzero brain mask using 1st/99th percentiles; selected slices are padded to a square, resized to 224×224, repeated into three channels and normalised for the encoders. Repeating grayscale does not add anatomical information; it supplies the expected input shape.

Each cache holds 16 slice feature vectors per participant. Fingerprints bind features to image bytes, preprocessing, weights, code and versions. Commands resume verified existing caches, so the two pilot participants will not be recomputed.

Read median seconds/person and estimate remaining time from uncached participants. First model download/load time is separate. If CUDA runs out of memory, set batch size 1 and retry. Close games and other GPU applications. A smaller batch changes memory demand, not the number of slices or labels.

## 8. Extract all development features

```powershell
python pipeline.py extract --encoder cnn --split development
python pipeline.py extract --encoder vit --split development
```

**Why save features:** classifier experiments then run quickly without repeatedly processing MRI images. Do not rename or mix caches from different runs. These feature files may still encode participant information; share under the dataset agreement.

## 9. Train and lock the experiment

```powershell
python pipeline.py train --suite review
```

Training uses only development participants. For each model, it fits a `StandardScaler` and class-balanced, L2-regularised logistic regression. Scaling is learned inside every training fold. Cross-validation selects `C` from 0.001, 0.01, 0.1 and 1. The final classifier is refitted on all development participants.

| Model | Participant input | What is trained |
|---|---|---|
| CNN-16 | Mean of 16 CNN vectors: 512 values | Scaler and logistic regression |
| ViT-16 | Mean of 16 ViT vectors: 768 values | Scaler and logistic regression |
| Fusion-16 | Concatenate both means: 1,280 values | Scaler and logistic regression |
| Age-only | Recorded age | Independent confound baseline |

Mean pooling averages each feature position across slices; concatenation appends the CNN and ViT vectors. We do not average labels or mix different people. The MRI classifiers do not receive CDR, MMSE or age as inputs.

The highest development CV AUC selects the frontend's default MRI model. Exact ties prefer CNN, then ViT, then fusion, and fewer slices where applicable. Age-only is not eligible as the MRI frontend model. This selection is based on AUC, not a formally optimised cost objective. Runtime is measured separately.

Review `outputs/results/development_comparison.csv` and `outputs/models/selected.json`. Training saves all models, CV results, configuration and a protocol lock. Do not modify code, slice positions, threshold, manifests or library versions after this point.

**Optional full study:** choose `python pipeline.py train --suite full` instead of the review command, before any test evaluation. It uses the same caches for all three representations at 4, 8 and 16 nested slices plus age-only. The extraction cache still contains 16 slices; benchmarking explicitly measures the smaller subsets. This is the path for claiming the complete slice-budget experiment.

**Deadline fallback:** if ViT extraction cannot finish, decide before testing to use `--suite cnn`, extract only CNN features, and honestly report a CNN baseline with fusion pending. Do not present unrun comparisons.

## 10. Evaluate once on untouched participants

After the protocol is locked:

```powershell
python pipeline.py audit --split test
python pipeline.py extract --encoder cnn --split test
python pipeline.py extract --encoder vit --split test
python pipeline.py evaluate
```

For the CNN-only fallback, omit test ViT extraction. Fixed test preprocessing checks are for detecting invalid files, not choosing new slices or tuning against outcomes.

The program reports participant-level ROC-AUC, balanced accuracy, sensitivity, specificity, F1 and confusion counts. It writes confidence intervals from stratified participant bootstrap sampling and paired fusion-versus-single-encoder AUC differences where those models exist. All models use the same test people and threshold 0.5.

**Important interpretation:** confidence intervals may be wide with a small test cohort. Bootstrap intervals are conditional on this fitted model/split; they do not capture every source of training variability. Multiple exploratory comparisons are not adjusted. A CI crossing zero for a paired AUC difference does not establish fusion superiority. The age-only result is a warning about possible confounding, not proof that age confounding has been eliminated. Single-dataset performance is not clinical/external validation.

Evaluation refuses a second run once results exist. If a genuine bug requires a corrected study, preserve the original run, document the change and its effect on test independence. Merely changing the output folder does not make previously inspected participants unseen again.

## 11. Check prediction consistency and measure computation

```powershell
python pipeline.py verify-prediction
python pipeline.py benchmark --subjects 2 --repeats 2
```

The first command compares a live development MRI prediction against its cached-feature prediction (tolerance 1e-5). Resolve discrepancies before demonstration.

The benchmark uses development samples and separates model loading, preprocessing, warm encoder time, end-to-end prediction time and available memory measurements. CUDA timings synchronize the device. CPU RSS is an after-run process reading, **not peak CPU memory**. GPU peak allocated memory is PyTorch allocator memory, not total card usage. Record GPU/CPU, versions, sample count and repeat count with results. Two samples/two repeats is a quick pilot; increase when time allows. A shorter input does not reduce the size of pretrained model weights.

## 12. Launch the frontend

```powershell
python -m streamlit run app.py --server.address 127.0.0.1
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`. The app provides:

- A menu of actually trained MRI models, defaulting to the development-selected model.
- Local dataset sample selection, or upload of one supported processed `.hdr` + `.img` pair / `.nii` / `.nii.gz` volume.
- Preprocessed MRI slices, predicted CDR group, a model score and elapsed time.
- Development comparison and, after evaluation, held-out metrics and ROC curves.

Uploads must match the training product's geometry and preprocessing. The app does not accept arbitrary MRI screenshots or perform raw clinical scan processing. Images shown are input slices, not explanations of diseased regions. The score is not a clinically calibrated disease probability.

Run the app once while internet is available to cache encoder weights before the review. Keep it local; do not publicly upload OASIS scans. If transferred to another computer, original sample paths will not exist: use the supported upload mode with authorised data or demonstrate on the training computer. Preserve the locked manifest bytes and recreate matching software versions. PyTorch CPU and CUDA builds have different version strings; the strict version check may require running on the original machine rather than bypassing it.

## 13. What to give back to the project owner

```powershell
python pipeline.py package-review
```

Send `outputs/Review_Return_Package.zip` to the authorised project collaborator. It includes code, configuration, fitted classifiers, result tables/plots, sealed manifests and a dependency freeze. It excludes MRI images, feature caches and QC image montages. It still contains participant metadata; follow OASIS terms. It does not bundle pretrained encoder weight files, so cache/download those on the demo system.

Also provide the filled `HANDOFF_CHECKLIST.md`, an app screenshot or screen recording using permitted data, and a note describing actual exclusions, selected slice range, GPU and training date. Do not send the original dataset publicly.

## 14. Results to use in the review

| Output | Use |
|---|---|
| `manifests/participants.csv` | Cohort, labels and reproducible subject splits |
| `manifests/` summary/exclusions | Actual included counts and reasons |
| `results/development_comparison.csv` | Selection without using test performance |
| `results/test_metrics.csv` | Held-out metrics with uncertainty |
| `results/confusion_*.png` | Actual error counts for each model |
| `results/test_roc.png` | ROC comparison |
| `results/paired_auc_differences.csv` | Paired fusion improvements and intervals |
| `results/encoder_timings.csv` | Warm extraction timing and memory fields |
| `results/end_to_end_timings.csv` | Demonstration latency including model loading |
| `results/prediction_consistency.json` | Saved-feature/live consistency |
| `models/training_protocol.json` | Exact suite, CV settings and environment |

The application is an exploratory MRI-to-CDR-group classifier. The experiment assesses whether fusion is useful; the frontend demonstrates that trained classifier. A working frontend alone is not evidence of accuracy.

## 15. Troubleshooting

| Message/symptom | Action and reason |
|---|---|
| No matching images | Check extraction, root, exact masked file suffix and matching pairs; do not change to raw files |
| Missing metadata columns | Use the official clinical table; set column names in config to actual headers |
| Excel dependency missing | Install requirements including openpyxl (XLSX) and xlrd (XLS) |
| Too few participants | Inspect missing archives/exclusions; do not fabricate records or split slices as people |
| Geometry not verified | Complete inspection and configure-geometry; do not bypass |
| Geometry/affine mismatch | Investigate mixed products/orientation with documentation; stop before training |
| CUDA unavailable | Check driver and install the correct CUDA PyTorch/torchvision pair in this venv |
| CUDA out-of-memory | Batch size 1, close GPU programs; CPU fallback if necessary |
| Download fails | Obtain internet access for official encoder weights; retry, never silently use random weights |
| Missing cache | Extract the requested encoder for the requested split; review suite needs both |
| Stale cache / changed lock | Restore original code/config/environment; do not delete validation guards |
| Training already locked | Continue to evaluation; do not rerun training to chase test accuracy |
| Test already evaluated | Use saved results; don't repeatedly choose models from test performance |
| App no local samples | Use supported upload or original training system |
| Low accuracy / fusion worse | Report honestly; inspect development errors/confounding; no promised 95% result |

## 16. Validation and expert review

Three distinct things are needed: (1) software tests, (2) unbiased participant-level evaluation, and (3) expert/clinical validation. This package supplies code and synthetic software checks. Your friend supplies actual OASIS evaluation. No expert endorsement, external hospital validation or clinical usability has been obtained.

Ask the faculty supervisor or an appropriately qualified imaging clinician to review the selected image product, axial orientation, slice coverage, target definition and interpretation of errors. Record who reviewed what and when, with their actual comments. Do not claim expert approval without that review. External validation would require a separately suitable cohort and compatible preprocessing, beyond the deadline baseline.

## 17. Official references

- [OASIS-1 dataset and access](https://sites.wustl.edu/oasisbrains/home/oasis-1/)
- [OASIS-1 release facts](https://bpb-us-e2.wpmucdn.com/sites.wustl.edu/dist/6/4383/files/2024/03/oasis_cross-sectional_facts-bcc7a002dfb104f4.pdf)
- [PyTorch installation](https://pytorch.org/get-started/locally/)
- [ResNet-18 pretrained model](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- [ViT-B/16 pretrained model](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html)
- [NiBabel image orientation](https://nipy.org/nibabel/image_orientation.html)
- [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html)
