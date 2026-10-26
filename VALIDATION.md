# Validation record — 26 October 2026

This package is starter research software. No real OASIS scans were supplied or used during these checks.

Passed:
- Five synthetic automated tests: participant matching/splitting, baseline training/evaluation, full ten-configuration dispatch and paired comparisons, feature concatenation/stale-cache rejection, metric/geometry checks, and synthetic NIfTI loading/normalisation.
- Actual CPU forward passes with official pretrained ResNet-18 and ViT-B/16 weights on artificial image tensors. Checked expected dimensions, finite outputs, repeated-inference agreement, differing outputs for differing slices, and frozen parameters.
- CPU integration on 20 artificial 3D NIfTI volumes: preparation, inspection, CNN feature extraction, baseline classifier training, held-out extraction and evaluation, model reload and direct inference. Direct prediction matched the cached-feature classifier score within 1e-6.
- Baseline component and end-to-end benchmark paths.
- Streamlit AppTest: config load, sample selection and Analyse button; no app errors; three output metrics rendered.

Not verified:
- Your actual OASIS files, Analyze orientation conventions in your copy, clinical metadata mapping, or scan quality.
- Real MRI classification performance; synthetic scores must never be reported as project results.
- Runtime/RAM on your PC, GPU execution, or a complete real-data ViT/fusion study.
- Pixel-level browser visual review. Frontend interaction was checked with Streamlit's testing API.

The shipped defaults intentionally require geometry verification before feature extraction. Install matching dependencies and record your own `requirements-lock.txt`; version changes invalidate a locked run.

Test environment versions:

```json
{
  "python": "3.12.14",
  "pipeline": "1.0.0",
  "numpy": "2.3.5",
  "pandas": "2.2.3",
  "nibabel": "5.4.2",
  "torch": "2.14.1+cpu",
  "torchvision": "0.29.1+cpu",
  "scikit-learn": "1.8.0"
}
```
