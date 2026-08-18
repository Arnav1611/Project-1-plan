# Anxiety-Modulated Neuroimaging Biomarkers for ASD Severity
## A complete implementation guide for a 2-person, 8-week course project

---

## 0. Read this first: the verdict

Your proposal document is a **12-month funded research programme written for a grant committee**. You have **8 weeks, 2 people, no funding, no MRI scanner, and no prior background in the field.** Those are different things.

The good news: the *core* of the project is genuinely doable in 8 weeks, because other people have already done the single hardest part for you — the neuroimaging preprocessing. Preprocessed ABIDE data is free, open, and downloadable with one line of Python.

The bad news: three specific things in the proposal cannot be done as written. I've flagged them in Section 2 so you can raise them with Dr. Subashini *before* you commit, rather than discovering them in week 6.

**What you should actually build:**

> A multimodal machine-learning pipeline on ABIDE data that (a) classifies ASD vs. typically-developing controls from resting-state functional connectivity and structural MRI features, (b) stratifies subjects by anxiety comorbidity and tests whether anxiety-related brain features improve severity prediction, (c) runs a statistical mediation analysis on the anxiety → brain → severity path with honest causal caveats, and (d) explains the model's decisions using SHAP.

That's a solid, defensible, publishable-at-workshop-level course project. It is Phases 1–4 of your proposal's roadmap compressed, with Phase 5 dropped.

---

## 1. Background: what is the problem, and what are we solving?

### 1.1 The clinical problem, in plain language

**Autism Spectrum Disorder (ASD)** is a neurodevelopmental condition involving differences in social communication and the presence of restricted or repetitive behaviours. It is a *spectrum*: two autistic people can look completely different clinically. One might need round-the-clock support; another might have a PhD and a family. This variation is called **heterogeneity**, and it is the central problem in autism research.

Right now, ASD is diagnosed and its severity rated **behaviourally** — a trained clinician observes the person and scores them on instruments like the ADOS. This works, but it has real limitations:

- It needs a scarce, expensively-trained clinician, and waiting lists run to years.
- Scores depend partly on who is doing the rating and how the child behaves that day.
- It tells you *what* the person does, not *why* their brain works differently.
- It can't easily separate "core autism" from co-occurring conditions.

So the field wants **biomarkers**: objective, measurable biological signals that track the condition. If you could look at a brain scan and say something reliable about severity, you'd have a second opinion that doesn't depend on clinician availability.

### 1.2 Where anxiety comes in

Autistic people have very high rates of anxiety. Prevalence estimates vary enormously depending on how you measure — the widely-cited review by White et al. (2009) reports a range from roughly 11% to 84% across studies. Your proposal quotes the 84% top end; be ready for a reviewer to ask about that, and quote the range instead.

Anxiety matters here for a reason that is easy to miss but is the actual intellectual core of your project:

**Anxiety and autism look similar in brain scans.**

Both involve the amygdala. Both involve the anterior cingulate. Both involve altered connectivity in the default mode network. So when a study reports "autistic brains show amygdala hyperactivation," there's an unanswered question hiding inside it:

> Is that an *autism* signature — or is it an *anxiety* signature that showed up because most of the autistic participants also happened to be anxious?

This is a **confounding** problem, and it's a genuine gap. Most ASD neuroimaging studies simply do not stratify by anxiety. If anxiety is driving part of what we call the "autism brain signature," then:

- Biomarker models are learning the wrong thing.
- Treatments targeting those circuits may be treating anxiety, not autism.
- Two autistic people with identical brain scans could need completely different interventions.

### 1.3 What we are actually solving

Three questions, in increasing ambition:

| # | Question | How we answer it |
|---|---|---|
| Q1 | Can brain features distinguish ASD from controls at all? | Supervised classification. This is the *baseline* — it replicates known work and proves your pipeline is correct. |
| Q2 | Does accounting for anxiety change the picture? | Stratify the cohort into ASD+anxiety / ASD-only / control. Compare which brain features matter in each group. Test whether adding anxiety improves severity prediction. |
| Q3 | Is anxiety *causally upstream* of the brain differences that predict severity? | Mediation analysis on the path anxiety → brain feature → severity. **With heavy caveats** (see 2.3). |

Q1 is guaranteed to produce results. Q2 is the novel contribution. Q3 is the ambitious, caveat-heavy part that makes it a "causal AI" project.

### 1.4 How we do it, at a glance

```
Brain scans (already preprocessed by other researchers, free)
        ↓
Turn each brain into a row of numbers  ← this is "feature extraction"
        ↓
Attach each person's anxiety score and severity score
        ↓
Train models to predict severity from brain numbers + anxiety
        ↓
Ask the model which numbers mattered  ← this is "explainability / XAI"
        ↓
Statistically test the anxiety → brain → severity path  ← "mediation"
```

---

## 2. Honest audit: three things in the proposal that don't work as written

Raise these with your supervisor early. Finding a real problem in a proposal and proposing a fix is *good* research behaviour, not a failure.

### 2.1 FLAG 1 — The named anxiety scales are not in ABIDE

Your proposal says anxiety severity will come from **SCARED, STAI, RCADS, and CBCL-Anxiety** available in ABIDE phenotypic metadata.

I checked the official ABIDE II Phenotypic Data Legend. Here's what's actually there:

| Instrument in ABIDE II | Actual column names | In ABIDE I? |
|---|---|---|
| MASC (Multidimensional Anxiety Scale for Children) — child self-report | `MASC_TOTAL_T`, `MASC_SOCIAL_TOTAL_T`, `MASC_PHYSICAL_TOTAL_T`, `MASC_HARM_TOTAL_T`, `MASC_SEP_T`, `MASC_ADI_T` | ❌ No |
| CBCL 6–18 — parent report | `CBCL_6-18_ANXIOUS_T` (Anxious/Depressed), `CBCL_6-18_ANXIETY_T` (DSM Anxiety Problems), `CBCL_6-18_INTERNAL_T` | ❌ No |
| BASC-2 Parent Rating Scales | `BASC2_PRS_ANXIETY_T`, `BASC2_PRS_INTERNAL_T` | ❌ No |
| Conners Parent Rating Scales | `CPRS_ANX_SHY` | ❌ No |
| CASI-4R / CSI-4 diagnostic cutoffs | `CASI_GAD_CUTOFF`, `CASI_SOCIAL_PHOBIA_CUTOFF`, `CASI_SEPARATION_CUTOFF`, `CSI_GAD_SEVERITY`, `CSI_SOCIAL_PHOBIA_SEVERITY` | ❌ No |
| Free-text psychiatric comorbidity | `NONASD_PSYDX_LABEL` + `NONASD_PSYDX_ICD9CODE` | ✅ Yes, as `COMORBIDITY` |

**SCARED, STAI and RCADS do not appear in the ABIDE legend at all.** Fix your proposal text to name MASC / CBCL / BASC-2 instead.

**And the bigger issue:** every one of the good anxiety instruments is ABIDE **II** only, and is only collected at *some* sites. A published study on amygdala volume and anxiety in ASD (Biological Psychiatry: CNNI, 2021) found only **three ABIDE-II sites** with usable CBCL anxiety data — Kennedy Krieger Institute, NYU, and Georgetown — and had to add their own University of Maryland sample to reach 294 participants.

So your realistic anxiety sample is **low hundreds at best, not 1,100.** You must audit this in week 1 before designing anything else. Section 5.1 gives you the script.

### 2.2 FLAG 2 — ABIDE II has no official preprocessed release

This is the sharpest practical constraint, and it creates a genuine dilemma:

- **ABIDE I** has a beautiful, free, fully preprocessed release (the Preprocessed Connectomes Project). Openly accessible, no login. → **But it has no proper anxiety scales.**
- **ABIDE II** has the anxiety scales. → **But you would have to preprocess the fMRI yourself.**

Preprocessing raw fMRI with fMRIPrep takes roughly 4–12 hours *per subject* and needs Linux, Docker, and a FreeSurfer licence. For 200 subjects on two student laptops, that is not an 8-week task. Do not attempt it.

**The resolution** is a two-track design (Section 5) — run your main ML pipeline on preprocessed ABIDE I with a coarse anxiety label, and run a smaller dimensional-anxiety analysis on the ABIDE II sites that have MASC/CBCL using **structural** features only, which are far cheaper to compute.

### 2.3 FLAG 3 — "Causal" mediation on cross-sectional data cannot establish direction

Your proposal states the aim as the *first causal model* of anxiety → fMRI/sMRI alterations → ASD severity.

Here is the problem, and it is not a technical one you can fix with a better model. All three variables — anxiety, brain measure, severity — are measured **at the same moment, in the same scan session.** There is nothing in the data that says anxiety came *first*.

The exact same numbers are equally consistent with:
- anxiety → brain change → severity (your hypothesis)
- severity → anxiety → brain change
- brain difference → both anxiety and severity (a common cause)
- some unmeasured third factor driving all three

Mediation analysis will happily give you an ACME estimate and a p-value for any of these orderings. The statistics cannot arbitrate. Direction comes from **design** — longitudinal measurement or experimental manipulation — not from software.

**What to do:** run the mediation, report it properly, and in your write-up say something like *"consistent with a mediation model in which..."* rather than *"anxiety causally drives..."*. State the identifying assumptions explicitly (no unmeasured confounding, correct temporal ordering, no mediator–outcome confounding). Then, to strengthen it:

1. Fit the **reversed** model (severity → brain → anxiety) and report that too. If both fit equally well, say so. This is the single most honest and most impressive thing you can do.
2. Run a **sensitivity analysis** for unmeasured confounding.
3. Always adjust for age, sex, full-scale IQ, site, and head motion.

Examiners reward this kind of rigour heavily. Overclaiming causality is the most common way these projects get torn apart in a viva.

### 2.4 FLAG 4 (bonus) — Phase 5 trimodal fusion is impossible, not just hard

Your roadmap has "Concatenate Phase 2 gut microbiome causal features with Phase 3 neuroimaging features; train ensemble model."

Feature concatenation requires **the same individuals** to have both data types. The public ASD microbiome datasets and ABIDE are completely disjoint populations — no ABIDE participant has a stool sample in a microbiome study. You cannot concatenate features across different people. There is no statistical trick that fixes this.

Reframe it honestly as: *"a proposed architecture for trimodal fusion, pending a future cohort with concurrent collection."* You can even implement the architecture and demonstrate it on simulated data as a proof-of-concept. Just don't claim it as a result.

---

## 3. Glossary — every term you need

### 3.1 Clinical / diagnostic

| Term | What it means |
|---|---|
| **ASD** | Autism Spectrum Disorder. |
| **TD / TDC** | Typically Developing (Control) — the non-autistic comparison group. |
| **Comorbidity** | A second condition occurring alongside the first (here: anxiety alongside autism). |
| **ADOS** | Autism Diagnostic Observation Schedule. A clinician-administered play/interview session, scored. The near-gold-standard diagnostic tool. |
| **ADOS Calibrated Severity Score (CSS)** | ADOS raw scores rescaled 1–10 so they're comparable across ages and language levels. **This is your best severity outcome variable.** ABIDE I: `ADOS_GOTHAM_SEVERITY`. ABIDE II: `ADOS_2_SEVERITY_TOTAL`. |
| **ADI-R** | Autism Diagnostic Interview–Revised. A long structured parent interview. |
| **SRS / SRS-2** | Social Responsiveness Scale. A 65-item questionnaire giving a continuous measure of autistic social traits. Better *coverage* in ABIDE than ADOS, so often a more practical severity outcome. |
| **MASC** | Multidimensional Anxiety Scale for Children. Child self-report anxiety measure. Present in ABIDE II only. |
| **CBCL** | Child Behavior Checklist. Parent-report questionnaire covering many problem domains, including Anxious/Depressed and DSM-oriented Anxiety Problems subscales. |
| **BASC-2** | Behavior Assessment System for Children, 2nd ed. Another broad parent-report scale with an Anxiety subscale. |
| **T-score** | A standardised score with mean 50 and SD 10 in the reference population. A T-score of 70 is 2 SD above average and usually the "clinically significant" threshold. All the CBCL/MASC/BASC anxiety columns are T-scores — convenient, because they're already comparable across ages. |
| **FIQ / VIQ / PIQ** | Full-scale / Verbal / Performance IQ. **Always** use FIQ as a covariate; it correlates with both brain structure and symptom scores. |
| **Phenotype / phenotypic data** | Everything non-imaging: age, sex, IQ, questionnaire scores, diagnosis, medication. |

### 3.2 Neuroimaging

| Term | What it means |
|---|---|
| **MRI** | Magnetic Resonance Imaging. Uses magnetic fields to image the body. No radiation. |
| **sMRI (structural MRI)** | One high-resolution 3D snapshot of brain **anatomy**. Answers "what shape and size is this brain?" Usually a **T1-weighted** image. |
| **fMRI (functional MRI)** | A *movie* of brain **activity** — many 3D volumes over time (e.g. one every 2 seconds for 6 minutes). Answers "which regions are active, and which activate together?" |
| **BOLD signal** | Blood-Oxygen-Level-Dependent signal. What fMRI actually measures: local blood oxygenation changes, used as an indirect proxy for neural activity. |
| **rs-fMRI (resting-state)** | fMRI while the person lies still doing nothing. You study spontaneous fluctuations. **Almost all of ABIDE is this.** |
| **Task fMRI** | fMRI while the person performs a task (e.g. viewing fearful faces). Better for anxiety questions, but rare in ABIDE. |
| **Voxel** | A 3D pixel. ABIDE preprocessed data is 3×3×3 mm per voxel; a brain has ~50,000–100,000 of them. |
| **ROI (Region of Interest)** | A named brain region, e.g. left amygdala. |
| **Atlas / parcellation** | A map that divides the brain into ROIs. This is the key dimensionality reduction step: 100,000 voxels → 200 ROIs. |
| **AAL** | Automated Anatomical Labeling atlas, 116 anatomical regions. Interpretable names. |
| **CC200 / CC400** | Craddock atlases, 200 or 400 regions derived from functional clustering. Standard for ABIDE connectivity ML. |
| **Harvard-Oxford (HO)** | Anatomical atlas including good subcortical structures — the one to use if the **amygdala** matters to you, which it does here. |
| **MNI space** | A standard reference brain coordinate system. Warping everyone into MNI space makes brains comparable across people. |
| **Functional connectivity (FC)** | How strongly two regions' time series move together — typically Pearson correlation. If amygdala and insula rise and fall together, they're functionally connected. |
| **Connectome / connectivity matrix** | The full N×N table of FC values between every pair of ROIs. For CC200 that's 200×200. |
| **DMN (Default Mode Network)** | A network active at rest, involved in self-referential thought and rumination. Reliably altered in both ASD and anxiety. |
| **ALFF / fALFF** | Amplitude / fractional amplitude of Low-Frequency Fluctuations. How much a voxel's signal fluctuates in the 0.01–0.1 Hz band. A per-voxel "activity level" measure. |
| **ReHo** | Regional Homogeneity. How similar a voxel's time course is to its immediate neighbours' — local coherence. |
| **VMHC** | Voxel-Mirrored Homotopic Connectivity. Correlation between a voxel and its mirror-image voxel in the opposite hemisphere. |
| **Cortical thickness** | Distance between the inner and outer surfaces of the cortical sheet, in mm. A classic sMRI feature; reduced in frontal regions in ASD. |
| **Grey matter volume** | Volume of neuron-cell-body-rich tissue. |
| **Sulcal depth / gyrification** | How deeply folded the cortex is. |
| **FreeSurfer** | The standard software for extracting cortical thickness, surface area and subcortical volumes from a T1 scan. `recon-all` is its main command and takes ~6–10 hours **per subject**. |
| **FSL / AFNI / SPM / ANTs / CIVET** | Other major neuroimaging toolboxes. |
| **fMRIPrep** | Modern, robust, standardised fMRI preprocessing pipeline. Excellent, but heavy: Docker + hours per subject. |
| **CPAC / CCS / DPARSF / NIAK** | The four pipelines used to produce the ABIDE Preprocessed release. **CPAC** is the most commonly used in ML papers. |
| **BIDS** | Brain Imaging Data Structure. A community standard folder/naming convention. Tools like fMRIPrep require it. |

### 3.3 Preprocessing steps (what was already done for you)

| Step | Why it's needed |
|---|---|
| **Slice timing correction** | An fMRI volume isn't captured all at once — slices are acquired sequentially. This aligns them to a common time point. |
| **Motion correction / realignment** | People move in the scanner. Each volume gets rigidly re-aligned to a reference. |
| **Framewise Displacement (FD)** | A per-timepoint number quantifying how much the head moved. **Critical:** motion creates spurious connectivity, and autistic children move more than controls — so motion can *masquerade* as a group difference. Always exclude high-motion subjects and always include mean FD as a covariate. The PCP download script only ships subjects with mean FD < 0.2. |
| **Spatial normalisation** | Warping each brain into MNI space. |
| **Smoothing (FWHM)** | Slight blurring to boost signal-to-noise. FWHM 6 mm = the blur kernel's full width at half maximum. |
| **Band-pass filtering (0.01–0.1 Hz)** | Keeping only slow fluctuations, where resting-state signal lives; discarding fast noise and slow drift. |
| **Global signal regression (GSR)** | Removing the whole-brain average signal. Controversial — it removes noise but can distort connectivity. This is why PCP ships four variants: `filt_global`, `filt_noglobal`, `nofilt_global`, `nofilt_noglobal`. Common default for ML: **`filt_noglobal`**. |
| **ICA-AROMA** | An automated, ICA-based method for identifying and removing motion-related noise components. |
| **Nuisance / confound regression** | Statistically removing motion parameters, white-matter and CSF signals, and drift from the time series. |

### 3.4 Machine learning

| Term | What it means |
|---|---|
| **Features (X)** | The input numbers. Here: connectivity values, cortical thicknesses, anxiety score, age. |
| **Label / target (y)** | What you're predicting. Diagnosis (classification) or severity score (regression). |
| **Classification** | Predicting a category (ASD vs TD). |
| **Regression** | Predicting a continuous number (an ADOS severity score of 7.3). |
| **p >> n** | More features than subjects. A CC200 connectome gives **19,900** features per person; you'll have maybe 800 people. This is the defining statistical difficulty of your project and it is why plain deep learning tends to overfit badly here. |
| **Overfitting** | The model memorises the training set instead of learning a generalisable pattern. Symptom: great training accuracy, poor test accuracy. |
| **Regularisation (L1 / L2)** | Penalising large model weights to prevent overfitting. **L1 (Lasso)** also drives weights to exactly zero, so it selects features. **L2 (Ridge)** shrinks them smoothly. |
| **Train / validation / test split** | Train = fit the model. Validation = tune choices. Test = final honest evaluation, touched **once**. |
| **Cross-validation (k-fold)** | Split data into k parts; train on k−1, test on the held-out one; rotate; average. Gives a more stable estimate than a single split. |
| **Stratified k-fold** | k-fold that preserves the class ratio in every fold. Always use this for imbalanced data. |
| **Leave-One-Site-Out CV (LOSO)** | Train on all imaging sites except one, test on that one. **This is the honest evaluation for multi-site data and you must report it.** Standard k-fold lets the model exploit site-specific quirks and inflates your numbers. |
| **Data leakage** | Any information from the test set influencing training. The #1 cause of fake high accuracy. Feature selection, scaling, ComBat harmonisation, and imputation must ALL be fit inside the training fold only. |
| **Batch / site effect** | Systematic differences between scanners and sites. A model can learn "this is a Caltech scan" instead of "this is an autistic brain." |
| **ComBat / neuroCombat** | The standard statistical method for removing site effects while preserving the biological effect you care about. |
| **Class imbalance** | Unequal group sizes. Handle via `class_weight='balanced'`, stratification, and by reporting AUC rather than raw accuracy. |
| **Logistic regression** | A linear classifier outputting a probability. Your first baseline, always. |
| **SVM (Support Vector Machine)** | Finds the separating boundary with the widest margin. Linear SVMs work notably well on connectome features. |
| **Random Forest** | Many decision trees on random subsets, votes averaged. Robust, little tuning, gives feature importances. |
| **Gradient boosting / XGBoost** | Trees built sequentially, each correcting the last one's errors. Usually the strongest model on tabular data. |
| **MLP** | Multi-Layer Perceptron: a plain fully-connected neural network. |
| **CNN** | Convolutional Neural Network. Exploits spatial structure via sliding filters. Natural for 3D sMRI images. |
| **GNN** | Graph Neural Network. Operates on graphs. A connectome **is** a graph — ROIs are nodes, connectivity strengths are edges — which is why GNNs are the current favourite for this data. |
| **GCN** | Graph Convolutional Network: each node updates itself by averaging its neighbours' features. |
| **GAT** | Graph Attention Network: same, but it *learns* how much to weight each neighbour. |
| **Transformer / attention** | Architecture that learns which parts of the input to attend to. Powerful, data-hungry. |
| **Multimodal fusion** | Combining data types. **Early fusion** = concatenate raw features. **Late fusion** = train separate models, combine predictions. **Intermediate fusion** = separate encoders, merge learned representations. Your dual-branch CNN+GNN design is intermediate fusion. |
| **Confusion matrix** | Table of true/false positives/negatives. |
| **Sensitivity / Recall** | Of the true ASD cases, what fraction did you catch? |
| **Specificity** | Of the true controls, what fraction did you correctly clear? |
| **Precision** | Of those you called ASD, what fraction really were? |
| **F1** | Harmonic mean of precision and recall. |
| **AUC-ROC** | Area under the sensitivity-vs-(1−specificity) curve. 0.5 = coin flip, 1.0 = perfect. **The main metric to report** — it's threshold-independent and robust to imbalance. |
| **XAI / SHAP** | Explainable AI. **SHAP** (SHapley Additive exPlanations) assigns each feature a fair credit share for a given prediction, based on cooperative game theory. Lets you say "this prediction was driven by amygdala–insula connectivity." |

### 3.5 Causal inference

| Term | What it means |
|---|---|
| **Correlation vs causation** | Ice cream sales correlate with drownings. Both are caused by summer. Correlation alone can't tell you what to intervene on. |
| **DAG** | Directed Acyclic Graph. Boxes = variables, arrows = assumed causal direction, no loops. A DAG makes your assumptions explicit and testable. Draw it by hand; don't let an algorithm invent it. |
| **Confounder** | A common cause of both your exposure and outcome. Must be adjusted for. (Age, sex, IQ, site, head motion here.) |
| **Mediator** | Sits *on* the causal path: A → M → B. **You must NOT adjust for a mediator** if you want the total effect — that's a classic error. |
| **Moderator** | Changes the *strength* of an effect without being on the path. Statistically, an interaction term. |
| **Mediation analysis** | Decomposes a total effect into the part flowing through the mediator and the part that doesn't. |
| **ACME** | Average Causal Mediation Effect — the indirect effect, i.e. through the mediator. |
| **ADE** | Average Direct Effect — the part not through the mediator. |
| **Proportion mediated** | ACME / total effect. |
| **Bootstrap** | Resample your data with replacement thousands of times to get confidence intervals without distributional assumptions. Standard for mediation, since indirect effects aren't normally distributed. |
| **SEM** | Structural Equation Modelling. Fits a whole system of equations at once; handles latent variables. Python: `semopy`. R: `lavaan` (more mature). |
| **Causal Bayesian Network** | A DAG plus conditional probability distributions. What your Phase 2 microbiome work used. |
| **Identifiability** | Whether a causal quantity can be estimated from your data *at all*, given your assumptions. Cross-sectional mediation is generally **not** identifiable in the strict sense — hence Flag 3. |

---

## 4. Datasets: what exists, what you can actually get

### 4.1 The recommendation, up front

| Dataset | Access | Anxiety data? | Preprocessed? | Verdict |
|---|---|---|---|---|
| **ABIDE I Preprocessed (PCP)** | **Fully open, no login** | ❌ Only free-text `COMORBIDITY` | ✅ **Yes, beautifully** | ⭐ **PRIMARY — build everything here** |
| **ABIDE I FreeSurfer sMRI derivatives** | Open (Zenodo / PCP) | n/a | ✅ Yes, precomputed stats | ⭐ **Use for the sMRI branch** |
| **ABIDE II** | Free NITRC account | ✅ MASC, CBCL, BASC-2 | ❌ No official release | ⚠️ Phenotype + T1 only |
| **NDAR / NDA** | Institutional DUA, NIH cert | ✅ Rich | Partial | ❌ Approval takes months. Not in 8 weeks. |
| **IMPAC** | Challenge archive | ❌ | ✅ | ⚠️ Optional extra cohort |
| **OpenNeuro** | Open | Varies | Varies | ⚠️ Small N; only if you want task-fMRI |

### 4.2 ABIDE I Preprocessed — your primary dataset

**What it is:** 1,112 datasets — 539 ASD and 573 controls — from 17 international sites, released 2012. The Preprocessed Connectomes Project then ran it through four pipelines (CPAC, CCS, DPARSF, NIAK) × four denoising strategies, and published the outputs. It is openly accessible without restriction and hosted on a public AWS S3 bucket.

**Why this changes your project:** you skip preprocessing entirely. The derivatives you want — ROI time series — are tiny text files.

**Derivatives available:** `alff`, `falff`, `reho`, `vmhc`, `lfcd`, `degree_binarize`, `degree_weighted`, `eigenvector_binarize`, `eigenvector_weighted`, `dual_regression`, `func_preproc`, `func_mask`, `func_mean`, and the ROI time series: `rois_aal`, `rois_cc200`, `rois_cc400`, `rois_dosenbach160`, `rois_ez`, `rois_ho`, `rois_tt`.

> ⚠️ **Critical practical warning:** ROI time series files (`rois_*`) are `.1D` text files — a few hundred KB each, so the whole cohort is ~1–2 GB. The 4D `func_preproc` volumes are `.nii.gz` at hundreds of MB each — **hundreds of gigabytes** for the full cohort. **Download `rois_*`, not `func_preproc`.** Getting this wrong will burn a week and your disk.

**How to get it (one line):**

```python
from nilearn.datasets import fetch_abide_pcp

abide = fetch_abide_pcp(
    data_dir="./data",
    pipeline="cpac",              # ccs | cpac | dparsf | niak
    band_pass_filtering=True,     # -> "filt_"
    global_signal_regression=False,  # -> "noglobal"  => filt_noglobal
    derivatives=["rois_cc200", "rois_ho"],
    quality_checked=True,         # keeps only subjects passing QC raters
)
# abide.phenotypic  -> pandas-style record array of all phenotype columns
# abide.rois_cc200  -> list of arrays, each (timepoints x 200)
```

Verified against nilearn 0.14.0. `quality_checked=True` filters on the PCP quality-assessment rater columns and will drop you to roughly 800–870 usable subjects. That's expected and correct — one published 3D-CNN study reports 774 ABIDE-I subjects passing QA by all functional raters (379 ASD / 395 TD).

**Phenotype file:** `Phenotypic_V1_0b_preprocessed1.csv`, which adds a `FILE_ID` column linking phenotype rows to imaging filenames, plus PCP quality metrics (`func_mean_fd`, `func_perc_fd`, `anat_snr`, `func_dvars`, and others). Use `func_mean_fd` for motion exclusion.

**Key ABIDE I columns for you:**
- `DX_GROUP` — 1 = Autism, 2 = Control
- `DSM_IV_TR` — 0 none, 1 Autism, 2 Asperger, 3 PDD-NOS
- `AGE_AT_SCAN`, `SEX` (1 = male, 2 = female), `FIQ`, `VIQ`, `PIQ`
- `ADOS_GOTHAM_SEVERITY` — calibrated severity 1–10 (**your best severity target**)
- `SRS_RAW_TOTAL` — continuous autistic-trait severity (**better coverage than ADOS**)
- `COMORBIDITY` — free text; **your only route to an anxiety label in ABIDE I**
- `CURRENT_MED_STATUS`, `MEDICATION_NAME` — note: anxiolytics/SSRIs here are informative
- `SITE_ID` — needed for LOSO CV and ComBat

### 4.3 ABIDE I structural (sMRI) features — without running FreeSurfer

This is the second shortcut that makes multimodal work feasible.

**Option A — PCP structural derivatives.** The PCP states that structural preprocessing and cortical measure calculation was performed with three pipelines: ANTs, CIVET, and FreeSurfer. Check the PCP download page for what's directly downloadable.

**Option B — the community FreeSurfer v6 release (recommended).** A public GitHub/Zenodo project (`dfsp-spirit/abide_preproc_smri_freesurfer6`) ran `recon-all` in FreeSurfer v6 on all **1,035 ABIDE I** subjects and published the derivatives: aseg and aparc stats (subcortical volumes, cortical thickness, surface area per region), local gyrification index, brain surfaces, and total brain measures. The repo itself contains an aggregated CSV of total/regional brain volume measures; the big surface data sits on Zenodo in ~6–16 GB chunks. Licensed CC BY-NC-SA under the ABIDE I usage agreement — **cite it, and check the licence before use.**

For your purposes you want the **aparc/aseg stats tables only** — a few MB. That gives you per-subject cortical thickness and volume for every ROI, joinable to the PCP functional data on subject ID. That is your entire sMRI branch, for free.

> Verify the subject-ID convention matches (`SUB_ID` vs zero-padded `FILE_ID`) before joining. Mis-joins here silently destroy a project.

### 4.4 ABIDE II — where the anxiety data lives

**Access:** free NITRC account, then accept the data usage agreement. Do this in week 1; approval is quick but not instant.

**Get the phenotype CSV first, before any imaging.** `ABIDEII_Composite_Phenotypic.csv` from the ABIDE II page. It's small. Audit it (Section 5.1) and *then* decide whether imaging is worth downloading.

**Sites known to carry CBCL anxiety data:** Kennedy Krieger Institute (KKI), NYU, and Georgetown (per the 2021 amygdala-volume study). MASC coverage differs — audit, don't assume.

**Severity columns:** `ADOS_2_SEVERITY_TOTAL` (1–10), `SRS_TOTAL_T`, `SRS_TOTAL_RAW`.

**Reality check to keep you calibrated:** that 2021 study, with 294 participants and a well-designed analysis, found **no statistically significant** amygdala-volume differences between autism-with-anxiety, autism-without-anxiety, and non-autistic groups. Published, peer-reviewed, null. If your 8-week project finds a huge effect, your first instinct should be to look for a bug or leakage — not to celebrate.

### 4.5 Datasets to deliberately skip

- **NDAR / NDA** — requires an institutional Data Use Certification signed by a designated official plus NIH credentialing. Realistically months. Your proposal lists it; cut it or mark it future work.
- **IMPAC (Paris-Saclay autism challenge)** — usable as an optional replication cohort, no anxiety data. Only if you're ahead of schedule.
- **OpenNeuro task-fMRI** — only if you specifically want amygdala habituation paradigms. Individual datasets are typically 20–60 subjects and each needs its own preprocessing. High effort, low N. Not for 8 weeks.
- **ABCD** — huge and excellent, but a full DUA process. No.

---

## 5. Plan A: the recommended pipeline, week by week

**Design principle: build a guaranteed deliverable first, then add the novel part.** If week 6 goes badly you still have a complete, working, reportable project.

### WEEK 1 — Setup and the audit gate

Nothing else matters until you know how many subjects have all three of: imaging, an anxiety measure, and a severity measure.

**Day 1–2:** environment + NITRC account.

```bash
pip install nilearn scikit-learn pandas numpy matplotlib seaborn \
            xgboost shap pingouin statsmodels neuroHarmonize
```

Use **Google Colab** (free tier is fine for weeks 1–5; Pro helps in week 6 for the GNN). Set up a shared GitHub repo on day 1.

**Day 3–5: THE AUDIT.** Write one script that answers, for both ABIDE I and ABIDE II:

```python
import pandas as pd

pheno = pd.read_csv("Phenotypic_V1_0b_preprocessed1.csv")   # ABIDE I

# 1. How many usable subjects after QC and motion exclusion?
usable = pheno[(pheno.func_mean_fd < 0.2)]
print(usable.DX_GROUP.value_counts())

# 2. Severity coverage - which target is actually usable?
for col in ["ADOS_GOTHAM_SEVERITY", "SRS_RAW_TOTAL", "ADOS_TOTAL"]:
    if col in usable.columns:
        n = usable[usable.DX_GROUP == 1][col].notna().sum()
        print(f"{col}: {n} ASD subjects have a value")

# 3. Anxiety signal in the free-text comorbidity field
anx_terms = ["anxiety", "anxious", "gad", "generalized anxiety",
             "social phobia", "separation anxiety", "specific phobia", "panic"]
com = usable["COMORBIDITY"].fillna("").str.lower()
usable["ANX_FLAG"] = com.apply(lambda s: int(any(t in s for t in anx_terms)))
print(pd.crosstab(usable.DX_GROUP, usable.ANX_FLAG))
print("\nNon-empty comorbidity entries:", (com.str.strip() != "").sum())
print(com[com.str.strip() != ""].value_counts().head(40))
```

Then the same for ABIDE II:

```python
ph2 = pd.read_csv("ABIDEII_Composite_Phenotypic.csv")
anx_cols = ["MASC_TOTAL_T", "CBCL_6-18_ANXIOUS_T", "CBCL_6-18_ANXIETY_T",
            "BASC2_PRS_ANXIETY_T", "CPRS_ANX_SHY",
            "CASI_GAD_CUTOFF", "CSI_GAD_SEVERITY"]
for c in anx_cols:
    if c in ph2.columns:
        print(f"{c:28s} n={ph2[c].notna().sum():4d}")
        print(ph2[ph2[c].notna()].SITE_ID.value_counts().to_string(), "\n")

# The number that decides your project:
both = ph2[ph2["CBCL_6-18_ANXIETY_T"].notna()
           & ph2["SRS_TOTAL_T"].notna()]
print("ABIDE II with anxiety AND severity:", len(both))
```

**The gate decision:**
- **≥120 ABIDE II subjects with anxiety + severity** → run Track 2b (dimensional anxiety, sMRI features).
- **<120** → run Track 2a (binary anxiety label from ABIDE I comorbidity text) and treat ABIDE II as a small descriptive supplement.

Write the audit numbers into your report. "We audited N=X and found Y" is a real finding and shows methodological maturity.

**Deliverable:** an audit notebook + a one-page cohort decision memo.

### WEEK 2–3 — Feature extraction

**Functional branch (ABIDE I, the main event):**

```python
import numpy as np
from nilearn.connectome import ConnectivityMeasure, sym_matrix_to_vec
from nilearn.datasets import fetch_abide_pcp

abide = fetch_abide_pcp(data_dir="./data", pipeline="cpac",
                        band_pass_filtering=True,
                        global_signal_regression=False,
                        derivatives=["rois_ho"],  # Harvard-Oxford: has amygdala
                        quality_checked=True)

ts_list = abide.rois_ho          # list of (timepoints x n_rois) arrays

# Correlation connectomes
corr = ConnectivityMeasure(kind="correlation", standardize="zscore_sample")
mats = corr.fit_transform(ts_list)          # (n_subjects, R, R)

# Fisher r-to-z: makes correlations closer to normally distributed
z = np.arctanh(np.clip(mats, -0.999999, 0.999999))
for m in z:
    np.fill_diagonal(m, 0)   # arctanh(1) = inf on the diagonal, so zero it out

# Vectorise the upper triangle -> feature matrix
X_fc = sym_matrix_to_vec(z, discard_diagonal=True)
print(X_fc.shape)   # (n_subjects, 6105) for a 111-region atlas: 111*110/2
```

Also compute a **tangent-space** version — it consistently outperforms raw correlation on ABIDE:

```python
tang = ConnectivityMeasure(kind="tangent", vectorize=True, discard_diagonal=True)
X_tan = tang.fit_transform(ts_list)
```

> ⚠️ `kind="tangent"` learns a reference mean from the data it's fitted on. To be strictly leakage-free, fit it on training folds only. For a course project, fitting once and noting the caveat is defensible — **but say so explicitly.**

**Structural branch:** load the precomputed FreeSurfer aparc/aseg stats, join on subject ID, keep cortical thickness per region + subcortical volumes + total intracranial volume (as a covariate/normaliser).

**Anxiety-focused ROI subset.** Alongside the full connectome, build a small, interpretable feature set from your proposal's Table 2.1 — amygdala, ACC, mPFC, insula, hippocampus, DMN regions. Roughly 12–20 ROIs → ~100–200 connectivity features. **This small set is often where your real result lives**, because it's statistically tractable, it's hypothesis-driven, and it's what the mediation analysis needs.

**Quality control:** drop mean FD > 0.2; drop QC-rater failures; drop subjects with excessive missing phenotype. Log every exclusion with a count — you need a CONSORT-style flow diagram in the report.

**Deliverable:** `features_fc.npy`, `features_smri.csv`, `cohort.csv`, and a documented exclusion table.

### WEEK 4–5 — Classical ML baselines

Do these **before** any deep learning. In this data regime they are frequently competitive with, or better than, neural networks, and they're what makes your deep-learning comparison meaningful.

Three tasks:
- **Task A:** ASD vs TD (binary). Sanity check.
- **Task B:** ASD+anxiety vs ASD-only vs TD (3-class). The novel bit.
- **Task C:** severity regression on `ADOS_GOTHAM_SEVERITY` or `SRS_RAW_TOTAL`, ASD subjects only.

Model ladder, in order: Logistic Regression (L2) → Linear SVM → Random Forest → XGBoost.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, cross_val_score

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("select", SelectKBest(f_classif, k=1000)),   # INSIDE the pipeline = no leakage
    ("clf", LogisticRegression(penalty="l2", C=1.0, max_iter=5000,
                               class_weight="balanced")),
])

# 1) Standard stratified 10-fold - the optimistic number
cv1 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
print("10-fold AUC:", cross_val_score(pipe, X, y, cv=cv1, scoring="roc_auc").mean())

# 2) Leave-One-Site-Out - the honest number
cv2 = LeaveOneGroupOut()
print("LOSO AUC:", cross_val_score(pipe, X, y, groups=sites,
                                   cv=cv2, scoring="roc_auc").mean())
```

**Report both numbers, always.** The gap between them *is* a result: it quantifies how much of your accuracy was site-specific.

**Site harmonisation with ComBat** — same discipline: fit on train, apply to test.

```python
from neuroHarmonize import harmonizationLearn, harmonizationApply
model, X_train_h = harmonizationLearn(X_train, covars_train)  # covars: SITE, AGE, SEX
X_test_h = harmonizationApply(X_test, covars_test, model)
```

Run the whole thing **with and without** ComBat and report both. Never fit ComBat on the full dataset before splitting — that is leakage, and it's the single most common flaw in student neuroimaging-ML projects.

**Deliverable:** a results table — model × task × (10-fold, LOSO) × (with/without ComBat) — with AUC, sensitivity, specificity, F1, and confusion matrices.

### WEEK 6 — Deep learning (dual-branch, the proposal's DAGMNet-inspired model)

Only start this once week 5's table is complete and committed.

**Architecture:**

```
sMRI features (~150)  →  MLP [128 → 64]  ──┐
                                            ├→ concat → Dense 64 → Dense 32 → softmax
FC connectome (graph) →  GCN/GAT [64 → 32] ─┤
                                            │
Anxiety + age + sex + FIQ (~5) → Dense 16 ──┘
```

**Graph construction:** nodes = ROIs. Node features = that ROI's connectivity profile row (or its ALFF/ReHo/thickness). Edges = thresholded connectivity — keep the top 10–20% strongest by absolute value, or take k-NN with k=10. Threshold choice matters; report it and try two.

**Practical warning:** PyTorch Geometric installation is genuinely painful and version-sensitive. Budget half a day, or **skip it** — a GCN layer is a two-line matrix operation in plain PyTorch:

```python
import torch, torch.nn as nn

class SimpleGCNLayer(nn.Module):
    """H' = act(  D^-1/2 A D^-1/2  H  W )"""
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, A_norm, H):        # A_norm: (B,R,R)  H: (B,R,d_in)
        return torch.relu(self.lin(torch.bmm(A_norm, H)))
```

That is a legitimate GCN. You do not need the library.

**Training discipline with n≈800:** small networks (tens of thousands of params, not millions), dropout 0.3–0.5, weight decay, early stopping on a validation fold, and **the same LOSO evaluation** as the baselines. If the deep model doesn't beat XGBoost, **report that honestly** — it's the expected outcome at this sample size and a genuinely useful finding, not a failure.

**Deliverable:** trained model, training curves, comparison against week-5 baselines.

### WEEK 7 — Causal mediation + explainability

**Step 1 — draw the DAG by hand.** Boxes: Anxiety, BrainFeature, Severity, and confounders Age, Sex, FIQ, Site, MeanFD. Arrows per your hypothesis. Do NOT use a causal-discovery algorithm to invent the graph — with this sample size it will output noise you can't defend.

**Step 2 — run the mediation.**

```python
import pingouin as pg

df = cohort[["anxiety_score", "amygdala_dmn_fc", "srs_total",
             "age", "sex", "fiq", "mean_fd"]].dropna()

res = pg.mediation_analysis(
    data=df,
    x="anxiety_score",       # exposure
    m="amygdala_dmn_fc",     # mediator (brain)
    y="srs_total",           # outcome (severity) - must be CONTINUOUS
    covar=["age", "sex", "fiq", "mean_fd"],
    n_boot=5000, seed=42,
)
print(res)
```

Verified against pingouin 0.6.1. Note the constraint: **the outcome must be continuous**, which is exactly why `SRS_RAW_TOTAL` or `ADOS_GOTHAM_SEVERITY` is your outcome rather than a mild/moderate/severe class.

Read the output rows: the `Indirect` row is your **ACME**; `Direct` is the **ADE**. Significance = bootstrap CI excluding zero. Compute proportion mediated = indirect / total.

**Step 3 — the honest additions that will win you marks:**
1. Run the **reverse** model: `x="srs_total"`, `m=brain`, `y="anxiety_score"`. Report both. If both are "significant", say plainly that cross-sectional data cannot distinguish them.
2. Test several mediators (amygdala, ACC, insula, DMN) and **correct for multiple comparisons** (FDR / Benjamini-Hochberg).
3. Add site as a covariate, or run within-site and meta-analyse.
4. Write out your identifying assumptions as a numbered list.

**Step 4 — SHAP.**

```python
import shap, xgboost as xgb

model = xgb.XGBClassifier(max_depth=3, n_estimators=300,
                          learning_rate=0.05).fit(X_tr, y_tr)
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_te)
shap.summary_plot(sv, X_te, feature_names=feat_names, max_display=25)
```

Use SHAP on the **small, interpretable ROI feature set**, not the 20,000-feature connectome — SHAP over 20k features produces an unreadable plot and no insight. Then map the top features back to named ROI pairs and check them against the literature in your proposal's Section 2. Agreement is a validation signal; disagreement needs discussing, not hiding.

**Deliverable:** mediation table (forward + reverse), DAG figure, SHAP summary + top-20 ROI-pair table.

### WEEK 8 — Write-up

Freeze the code. No new experiments. Details in Section 11.

---

## 6. ML concepts you're using, and *why*

### 6.1 Why a connectivity matrix becomes a feature vector

An FC matrix is symmetric with a meaningless diagonal, so all information lives in the upper triangle: R×(R−1)/2 values. R=200 → **19,900 features**. With ~800 subjects that's ~25 features per subject. This is severe `p >> n`.

Consequences you must design around:
1. Unregularised models will overfit completely.
2. Feature selection or dimensionality reduction is mandatory.
3. Deep learning has little advantage — it needs far more data than this.
4. **A smaller atlas is often better.** Harvard-Oxford (~111 regions) gives ~6,000 features and includes the amygdala. Use it.

### 6.2 Why Fisher r-to-z

Correlations are bounded in [−1, 1], so their sampling distribution is skewed near the ends. `arctanh(r)` maps them to an unbounded, roughly normal scale — which is what linear models and t-tests assume. One line, always do it.

### 6.3 Why tangent space beats correlation

Correlation matrices live on a curved manifold (symmetric positive-definite matrices), not in flat Euclidean space, so Euclidean operations on them are slightly wrong. Tangent-space embedding projects them onto a flat space around a reference point where Euclidean geometry is valid. In practice this gives a few points of AUC for free. `ConnectivityMeasure(kind="tangent")`.

### 6.4 Why leave-one-site-out is non-negotiable

Every site has a different scanner, sequence, population, and recruitment pattern. Some sites have more ASD subjects than others. With standard k-fold, subjects from the same site appear in both train and test, so a model can learn "scanner signature → site → base rate of ASD at that site" and score well without learning anything about autism.

LOSO breaks this. Your LOSO number will be lower. That lower number is the real one.

For context on why this matters: a multi-site harmonisation study analysing 2,226 ABIDE I+II subjects across 39 site-samples notes that earlier work found ASD-vs-control accuracy under 60% from anatomical measures alone, and attributes much of the variability in published performance to inter-site batch effects.

### 6.5 Why a GNN, conceptually

A connectome is literally a graph. Feeding a flattened 19,900-vector to an MLP throws away all of that structure — the model doesn't know that feature #4,182 and feature #4,183 share a brain region. A GNN keeps the topology: each ROI updates its representation by aggregating its neighbours', so the model reasons over networks rather than over an arbitrary list of numbers. GAT extends this by *learning* which neighbours matter, which also gives you interpretable attention weights.

Whether this actually helps at n≈800 is an empirical question. Recent work has questioned whether graph deep learning genuinely outperforms simpler baselines on functional connectomes. **Test it, don't assume it.** "We tested GNNs and they did not beat regularised logistic regression at this sample size" is a legitimate, well-received finding.

### 6.6 Why SHAP rather than raw feature importance

Random Forest's built-in importance is biased toward high-cardinality features and doesn't tell you the *direction* of an effect. SHAP gives per-subject, signed, additively-consistent attributions grounded in game theory — so you can say "for this participant, elevated amygdala–insula connectivity pushed the prediction toward higher severity." That's the kind of statement a clinician can engage with, which is the whole point of the "clinically translatable" claim in your proposal.

### 6.7 The realistic accuracy you should expect

Set expectations now, with your supervisor, in writing.

| Task | Honest expectation (LOSO) | What papers often claim |
|---|---|---|
| ASD vs TD, rs-fMRI FC | **AUC 0.65–0.75** | 0.90–0.97 |
| ASD vs TD, sMRI only | **AUC 0.55–0.65** | 0.85+ |
| 3-class with anxiety | **Accuracy 0.45–0.60** | — |
| Severity regression | **R² 0.05–0.20** | — |

Your proposal cites DAGMNet at 96.80% AUC. Treat that number with real caution. Very high ABIDE accuracies frequently come from some combination of: no leave-one-site-out validation, feature selection performed before cross-validation, hyperparameters tuned on the test set, single favourable train/test splits, or single-site subsets. Getting 0.72 AUC with airtight LOSO methodology is **better science** than 0.96 with leakage, and a good examiner knows it.

Put this table in your report as an "expected performance" section. If you then hit 0.73, you've met your stated target rather than missed an imaginary one.

---

## 7. Alternatives, ranked by feasibility

| # | Approach | Effort | Risk | Novelty | Verdict |
|---|---|---|---|---|---|
| **A** | **ABIDE I preprocessed FC + precomputed FreeSurfer sMRI + comorbidity-derived anxiety label; classical ML → GNN; mediation + SHAP** | Medium | **Low** | Medium | ⭐ **DO THIS** |
| **B** | A, plus ABIDE II dimensional anxiety (MASC/CBCL) on 3–4 sites, sMRI-only features | Med-High | Medium | **High** | ⭐ **Stretch goal — add if week 1 audit is favourable** |
| C | fMRI-only, no structural branch | Low | Very low | Low | Safe fallback if you fall behind. Drop the "multimodal" claim. |
| D | sMRI-only (thickness/volume) | Low | Low | Low | Weakest signal (~0.55–0.65 AUC). Only as a comparison arm. |
| E | Phenotype-only ML (no imaging at all) | Very low | Very low | Very low | Excellent as a **baseline to beat** — proves imaging adds something. Include this. |
| F | Preprocess ABIDE II raw yourself with fMRIPrep | **Very high** | **Very high** | Medium | ❌ Not in 8 weeks. Don't. |
| G | Task-fMRI amygdala habituation from OpenNeuro | High | High | High | ❌ N too small, per-dataset preprocessing. |
| H | NDAR/NDA with rich anxiety measures | Blocked | — | High | ❌ DUA takes months. |
| I | Trimodal with Phase 2 microbiome | Impossible | — | — | ❌ Disjoint subjects. See Flag 4. |
| J | 3D CNN on raw `func_preproc` volumes | High | High | Low | ❌ Hundreds of GB download, needs serious GPU. |

**Recommended: A as your committed scope, B as the stretch goal, E as a mandatory baseline, C as your fallback.**

The reason A works is that every expensive step has already been done by someone else and published openly. You are doing the *analysis*, which is the part that's actually yours.

---

## 8. Splitting the work between two people

Pair on week 1 (the audit is too important to delegate). Then split by *layer*, not by *phase*, so neither of you is ever blocked waiting on the other.

**Person A — Data & Features ("upstream")**
- Week 1: ABIDE I download, phenotype audit, NITRC registration
- Week 2–3: FC extraction, atlas comparison, QC, exclusion logging
- Week 3: FreeSurfer stats join, sMRI feature table
- Week 4: ComBat harmonisation module
- Week 5–6: graph construction for the GNN, data loaders
- Week 7: figures — connectome plots, brain visualisations, cohort flow diagram
- Week 8: methods + data sections of the report

**Person B — Models & Statistics ("downstream")**
- Week 1: joint audit; environment, repo, CV scaffolding
- Week 2–3: build the CV harness against *synthetic* data so it's ready the moment features land
- Week 4–5: the full baseline ladder, results tables
- Week 6: dual-branch deep model
- Week 7: mediation analysis, SHAP, statistical tests
- Week 8: results + discussion sections

**Shared, non-negotiable:**
- One GitHub repo, branches, actual pull requests
- A 30-minute sync twice a week with written notes
- One shared `RESULTS.md` where every number is logged with the date and commit hash that produced it
- **Person B never touches feature code; Person A never touches model code.** Merge conflicts in a notebook will cost you a day.

**Repo layout:**
```
asd-anxiety-phase3/
├── README.md
├── requirements.txt
├── data/                 # gitignored
├── notebooks/
│   ├── 01_phenotype_audit.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_baselines.ipynb
│   ├── 04_deep_fusion.ipynb
│   ├── 05_mediation.ipynb
│   └── 06_shap.ipynb
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── harmonize.py
│   ├── models.py
│   └── evaluate.py
├── results/
│   ├── tables/
│   └── figures/
└── RESULTS.md
```

---

## 9. Pitfalls that kill these projects

Ordered by how often they actually happen.

1. **Downloading `func_preproc` instead of `rois_*`.** Hundreds of GB, a week gone. Check the derivative name twice.
2. **Data leakage via preprocessing outside CV.** Scaling, feature selection, ComBat, and imputation fitted on all data before splitting. Symptom: suspiciously good results. Fix: everything inside a `sklearn` `Pipeline`.
3. **Reporting only k-fold, never LOSO.** Your reviewer will ask. Have the number ready.
4. **Ignoring head motion.** Autistic children move more. Motion inflates apparent group differences. Exclude FD > 0.2 *and* covary mean FD.
5. **Ignoring site.** Without harmonisation or LOSO you may be classifying scanners.
6. **Subject-ID join errors.** `SUB_ID` vs zero-padded `FILE_ID`. Always assert row counts before and after every merge.
7. **Adjusting for the mediator when you want the total effect.** Conceptual error that silently invalidates the analysis.
8. **Claiming causality from cross-sectional data.** See Flag 3. Use hedged language.
9. **Starting deep learning in week 2.** You'll have no baseline to compare against and no idea whether the pipeline is even correct.
10. **Chasing DAGMNet's 96.8%.** You will not reproduce it with honest validation. Trying will consume weeks.
11. **3-class severity with tiny per-class counts.** Check `value_counts()` first. If severe-anxiety-ASD has n=23, don't build a 3-class classifier on it — do regression, or binary high/low splits.
12. **No exclusion log.** You'll be asked exactly how you got from 1,112 to your final N, and you must be able to answer with a table.
13. **Both people editing the same notebook.** Notebook merge conflicts are unresolvable in practice.
14. **Writing the report in week 8.** Draft methods in week 3, while you still remember what you did and why.

---

## 10. Report structure

Mirror your proposal's structure so the continuity is obvious.

1. **Abstract** — one paragraph, lead with the actual numbers.
2. **Introduction** — Section 1 of this guide, expanded. The anxiety-confound argument is your hook.
3. **Related work** — the references in your proposal, plus what you added. Explicitly note that prior ASD neuroimaging-ML work does not stratify by anxiety.
4. **Data** — ABIDE I/II, access route, the audit results, exclusion flow diagram, final cohort table by group and site.
5. **Methods** — features, harmonisation, models, CV scheme, mediation specification with assumptions, SHAP.
6. **Results** — the model × task × validation table; mediation forward *and* reverse; SHAP top features; the phenotype-only baseline for comparison.
7. **Discussion** — what the findings mean; agreement/disagreement with the literature.
8. **Limitations** — cross-sectional causality, sparse anxiety coverage, site effects, sample size, no external validation, anxiety instruments differing across sites. **Write this section generously.** A thorough limitations section is read as competence, not weakness.
9. **Future work** — Phase 4 trimodal architecture, longitudinal designs, prospective collection.

**Figures to produce:** cohort flow diagram; group-mean connectome matrices per group; ROC curves (10-fold vs LOSO overlaid); the model comparison bar chart; the DAG; mediation path diagram with coefficients; SHAP summary; a brain plot of the top ROIs.

---

## 11. Reading list, in the order to read it

**Read first (week 1):**
- The Nilearn tutorials on functional connectivity — hands-on, directly reusable code.
- The ABIDE Preprocessed site (`preprocessed-connectomes-project.org/abide/`) — read the pipelines and derivatives pages properly. One hour here saves days.
- The ABIDE I and ABIDE II Phenotypic Data Legends. Yes, actually read them. This is where your Flag 1 came from.

**Read in week 2–3:**
- Di Martino et al. (2014), ABIDE I — the dataset paper. *Molecular Psychiatry*.
- Di Martino et al. (2017), ABIDE II — *Scientific Data* 4:170010.
- Craddock et al. (2013) — the Preprocessed Connectomes / Neuro Bureau initiative.
- Abraham et al. (2017), deriving reproducible ASD biomarkers from rs-fMRI — the canonical careful-methodology ABIDE ML paper. Read this before you build any model.

**Read in week 4–6:**
- Nielsen et al. (2013), multisite FC classification of autism — *Frontiers in Human Neuroscience*. On why multi-site is hard.
- The multi-site harmonisation paper on ABIDE (*NeuroImage: Clinical*, 2022) — batch effects and ML discrimination.
- Any recent review on graph deep learning for functional connectomes, ideally one that questions whether GNNs actually beat baselines.

**Read in week 7:**
- Imai, Keele & Tingley on causal mediation — the ACME/ADE framework your analysis implements.
- VanderWeele on mediation assumptions and sensitivity analysis — this is what lets you write Flag 3 defensibly.
- Lundberg & Lee (2017), SHAP — *NeurIPS*.

**From your own proposal, worth reading properly:**
- White et al. (2009) on anxiety in ASD — the prevalence range.
- Kerns & Kendall (2012) on classifying anxiety in ASD — the distinction between typical and atypical ASD-related anxiety is directly relevant to how you define your anxiety label.
- The 2021 *Biological Psychiatry: CNNI* amygdala-volume-and-anxiety paper — the closest published work to your Track 2b, and it reports null findings. Read it to calibrate.

---

## 12. Week-by-week checklist

| Week | Person A | Person B | Gate |
|---|---|---|---|
| 1 | ABIDE I download; NITRC account | Repo, env, CV scaffolding | ✅ **Audit memo: how many subjects have anxiety + severity + imaging?** |
| 2 | FC extraction, HO + CC200 | CV harness on synthetic data | ✅ Feature matrix exists with correct shape |
| 3 | sMRI join; QC; exclusion log | Phenotype-only baseline | ✅ Cohort locked, exclusions documented |
| 4 | ComBat module | Baseline ladder, Tasks A/B/C | ✅ First real AUC, both 10-fold and LOSO |
| 5 | Graph construction | Full results table | ✅ **Complete reportable results exist. Project is now safe.** |
| 6 | Data loaders, figures | Dual-branch deep model | ✅ Deep vs classical comparison |
| 7 | Brain figures, DAG figure | Mediation + SHAP | ✅ Mediation forward + reverse |
| 8 | Methods/data write-up | Results/discussion write-up | ✅ Report, deck, repo frozen |

**The single most important line in this table is the week-5 gate.** Once you're past it you have a complete project, and everything after is upside.

---

## 13. Five things to raise with your supervisor this week

1. The proposal names SCARED/STAI/RCADS; ABIDE actually has MASC/CBCL/BASC-2, ABIDE II only. Can we update the instrument list?
2. ABIDE II has no preprocessed fMRI release. Do we accept a two-track design — functional analysis on ABIDE I, dimensional anxiety on ABIDE II structural?
3. Cross-sectional mediation can't establish direction. Can we frame Q3 as "consistent with" and report the reverse model too?
4. Phase 5 trimodal fusion needs subjects with both microbiome and MRI. No public cohort has this. Reframe as future work?
5. What performance would count as success? We'd like to propose LOSO AUC ≥ 0.70 for ASD vs TD as the target, and explain why the 96% figures in the literature aren't a fair benchmark.

---

---

## Appendix — what was verified vs. what to check yourself

**Verified by execution** (nilearn 0.14.0, pingouin 0.6.1, scikit-learn, torch 2.13):
- `fetch_abide_pcp` signature and argument names
- The full connectivity → Fisher-z → vectorise chain, including output shapes
- Tangent-space embedding
- The `Pipeline` + `StratifiedKFold` + `LeaveOneGroupOut` cross-validation code
- `pingouin.mediation_analysis` — output rows are `['M ~ X', 'Y ~ M', 'Total', 'Direct', 'Indirect']`; read the **`Indirect`** row for your ACME and **`Direct`** for the ADE
- The plain-PyTorch GCN layer

**Verified against official documentation:** the ABIDE II Phenotypic Data Legend (every anxiety column name in Flag 1 was read directly from it); the PCP derivative and pipeline names; ABIDE I/II sample sizes; ABIDE I open access vs. ABIDE II NITRC login.

**Not verifiable without downloading — check these yourself in week 1:**
- The **actual per-site counts** of non-missing MASC/CBCL/BASC-2 anxiety values. This is your project's key unknown. Everything in Section 5's gate decision depends on it.
- How densely populated ABIDE I's free-text `COMORBIDITY` field really is, and what vocabulary sites used. Print the raw value counts before trusting any keyword parser.
- Whether the community FreeSurfer stats tables join cleanly to the PCP phenotype file on subject ID.
- Current hosting arrangements for the PCP S3 bucket and the Zenodo structural chunks.

*Guide prepared for a 2-person, 8-week implementation of Phase 3, August 2026.*
