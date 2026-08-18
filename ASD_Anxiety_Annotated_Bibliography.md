# Annotated Bibliography — ASD Anxiety-Neuroimaging Project
## Every paper you need, organised by theme, with reading priority

---

## 0. How to use this list

There are ~80 papers here. **You will not read 80 papers in 8 weeks, and you shouldn't try.**

Each entry is tagged:

| Tag | Meaning | How many |
|---|---|---|
| 🔴 **CORE** | Read the full paper. Non-negotiable. | 12 |
| 🟡 **SKIM** | Read abstract + methods + figures. | ~25 |
| ⚪ **CITE** | Cite for context; abstract is enough. | rest |

A realistic target is **12 CORE papers read properly + 25 skimmed**. That is a strong course-project literature base.

**Verification status:** every entry marked ✅ VERIFIED had its authors, journal, year and DOI checked against PubMed/publisher records during this search. Entries marked ⚠️ UNVERIFIED came from your proposal or from reference lists and should be checked before you cite them.

---

## 1. ⚠️ Corrections to your proposal's reference list

Fix these before submission. Getting a citation wrong in a reference list is the kind of thing an examiner notices immediately.

### Error 1 — DAGMNet: wrong authors, wrong journal, wrong DOI

**Your proposal says:**
> Li, Z. et al. (2025). DAGMNet: Dual-Branch Attention-Pruned GNN for Multimodal sMRI and fMRI Fusion in Autism Prediction. PMC12467520. doi:10.3390/diagnostics15172232

**Actually:** ✅ VERIFIED
> Wang, L., Li, X., Yuan, J., & Chen, Y. (2025). DAGMNet: Dual-Branch Attention-Pruned Graph Neural Network for Multimodal sMRI and fMRI Fusion in Autism Prediction. **Biomedicines**, 13(9), 2168. doi:**10.3390/biomedicines13092168**. Published 5 Sept 2025. PMC12467520 ✓ (the PMC ID is the only correct part).

The journal is *Biomedicines*, not *Diagnostics*. First author is Lanlan Wang (Central South University), not "Li, Z."

### Error 2 — The anxiety fMRI paper: wrong first author AND misdescribed findings

**Your proposal says:**
> Ikeda, T. et al. (2025). The role of anxiety in modulating temporal processing and sensory hyperresponsiveness in ASD: an fMRI study.

**Actually:** ✅ VERIFIED
> **Atsumi, T.**, Ide, M., Chakrabarty, M., et al. (2025). *Scientific Reports*, 15, 17674. doi:10.1038/s41598-025-02117-5. PMID 40399452.

The DOI is right; the first author is wrong. More importantly, **your proposal misstates what the paper found.**

| Your proposal claims | What the paper actually reports |
|---|---|
| Anxiety modulates processing "via amygdala–supramarginal gyrus functional connectivity" | The amygdala–left supramarginal gyrus connectivity finding was in the **TD group**, not the ASD group |
| "Causal mediation analysis confirming an anxiety → sensory circuit pathway" | Anxiety **mediated** the link between right **angular gyrus** activation and sensory hyperresponsiveness in the ASD group — a different region and a different path |
| (implied large study) | **n = 25 ASD + 25 TD.** Task-based fMRI (visual temporal order judgment cued by facial emotion), not resting state |

Fix the mechanism description. As written, a reviewer who knows the paper will assume you didn't read it.

### Error 3 — The anxiety prevalence figure needs a range

Your proposal states anxiety affects "up to 84%" of autistic individuals, citing White et al. (2009). That review reports a **range of roughly 11–84%** depending on measurement method and sample. Quoting only the ceiling is technically defensible but looks like cherry-picking. Write "estimates range from about 11% to 84% depending on assessment method (White et al., 2009)."

### Error 4 — Incomplete reference

> "Twinned neuroimaging analysis for ASD classification. Scientific Reports (2024). doi:10.1038/s41598-024-71174-z"

No authors listed. Complete this or drop it.

### Error 5 — Unverified

> Imoh, P.O. et al. (2025). Leveraging AI-Driven Neuroimaging Biomarkers... PMC12346713.

Could not confirm. Check the PMC ID resolves before citing.

### ✅ Correct in your proposal

- White, S.W. et al. (2009). *Clinical Psychology Review*, 29(3), 216–229. doi:10.1016/j.cpr.2009.01.003
- Kerns, C.M. & Kendall, P.C. (2012). *Clinical Psychology: Science and Practice*, 19(4), 323–347.
- Peralta-Marzal, L.N. et al. (2024). *Scientific Reports*, 14, 814. doi:10.1038/s41598-023-50601-7 ✅ VERIFIED — authors, volume, article number and DOI all correct.

---

## 2. THEME A — The datasets (read these first)

🔴 **CORE** — you cannot write a methods section without these three.

1. ✅ **Di Martino, A., Yan, C.-G., Li, Q., et al. (2014).** The autism brain imaging data exchange: towards a large-scale evaluation of the intrinsic brain architecture in autism. *Molecular Psychiatry*, 19(6), 659–667.
   → **The ABIDE I dataset paper.** 1,112 datasets, 539 ASD / 573 controls, 17 sites. Cite this every time you mention ABIDE I. 🔴

2. ✅ **Di Martino, A., O'Connor, D., Chen, B., et al. (2017).** Enhancing studies of the connectome in autism using the autism brain imaging data exchange II. *Scientific Data*, 4, 170010. doi:10.1038/sdata.2017.10. PMC5349246
   → **The ABIDE II dataset paper.** 17 new cross-sectional collections: 487 ASD + 557 controls from 16 institutions; ABIDE I+II combined give 2,156 unique cross-sectional datasets. Critically, this is the paper that documents the added psychiatric/comorbidity variables — **the reason your anxiety data exists at all.** 🔴

3. ✅ **Craddock, C., Benhajali, Y., Chu, C., et al. (2013).** The Neuro Bureau Preprocessing Initiative: open sharing of preprocessed neuroimaging data and derivatives. *Neuroinformatics* (Frontiers abstract).
   → **The ABIDE Preprocessed / PCP citation.** Four functional pipelines (CCS, CPAC, DPARSF, NIAK) × four denoising strategies; structural measures via ANTs, CIVET and FreeSurfer. You must cite this plus the specific pipeline you used. 🔴

4. ⚪ **ABIDE I & ABIDE II Phenotypic Data Legends** (fcon_1000.projects.nitrc.org/indi/abide/)
   → Not a paper, but read them like one. This is where I found that your proposal's anxiety instruments don't exist in the dataset. Documentation, `ABIDEII_Data_Legend.pdf`. 🔴

5. ⚪ **Nielsen, J.A., Zielinski, B.A., Fletcher, P.T., et al. (2013).** Multisite functional connectivity MRI classification of autism: ABIDE results. *Frontiers in Human Neuroscience*, 7, 599.
   → The first big multi-site ABIDE classification attempt. Cite for "multi-site is hard." 🟡

---

## 3. THEME B — ASD classification from rs-fMRI: the methodological canon

This is where your ML pipeline comes from. **Read #6 before you write any model code.**

6. ✅ **Abraham, A., Milham, M.P., Di Martino, A., Craddock, R.C., Samaras, D., Thirion, B., & Varoquaux, G. (2017).** Deriving reproducible biomarkers from multi-site resting-state data: An autism-based example. *NeuroImage*, 147, 736–745. doi:10.1016/j.neuroimage.2016.10.045. PMID 27865923
   → 🔴 **THE most important methods paper for your project.** N=871 ABIDE subjects, fully automatic connectome pipeline, prediction on subjects from **unseen sites**. Reports ~67–68% accuracy — better than prior state of the art at the time. Also shows accuracy improves with more subjects, and that atlas/region-definition choice matters enormously for biomarker discovery. Open access preprint: arXiv:1611.06066.
   → **Use this as your performance benchmark, not DAGMNet.** When your supervisor asks why you're not hitting 96%, this is your answer.

7. ⚪ **Heinsfeld, A.S., Franco, A.R., Craddock, R.C., Buchweitz, A., & Meneguzzi, F. (2018).** Identification of autism spectrum disorder using deep learning and the ABIDE dataset. *NeuroImage: Clinical*, 17, 16–23.
   → Denoising-autoencoder + MLP, ~70% accuracy on ABIDE I. Heavily cited as the "deep learning on ABIDE" baseline. 🟡

8. ⚪ **Arbabshirani, M.R., Plis, S., Sui, J., & Calhoun, V.D. (2017).** Single subject prediction of brain disorders in neuroimaging: Promises and pitfalls. *NeuroImage*, 145, 137–165.
   → 🟡 Read the pitfalls section. This is where the leakage, small-sample and overfitting failure modes are catalogued. Cite it in your Limitations section.

9. ⚪ **Dvornek, N.C., Ventola, P., Pelphrey, K.A., & Duncan, J.S. (2017).** Identifying autism from resting-state fMRI using long short-term memory networks. *MLMI/MICCAI Workshop*, 10541, 362–370.
   → Sequence-model alternative: LSTM on ROI time series rather than static FC. Worth knowing as an "alternative approach" for your report. ⚪

10. ⚪ **Dvornek, N.C., et al. (2018).** Combining phenotypic and resting-state fMRI data for autism classification with recurrent neural networks. *ISBI*. PMID 30288208
    → Directly relevant to your dual-branch design: how to fuse phenotype with imaging. 🟡

11. ⚪ **Wang, Y., et al.** Multi-site clustering and nested feature extraction (MC-NFE) for fMRI-based ASD detection.
    → Handles site heterogeneity via clustering. ⚪

12. ⚪ **Rangaprakash, D., et al.** VAE deep learning with domain adaptation, transfer learning and harmonization for diagnostic classification from multi-site neuroimaging data.
    → 🟡 Notable finding for you: domain adaptation performed **comparably to ComBat**, and adding external healthy-control data (HBN, AOMIC) via transfer learning pushed ABIDE-II accuracy to ~73.8%. Also states plainly that independent-test-set accuracy on ABIDE **rarely exceeds 70%.** Excellent citation for your expected-performance section.

---

## 4. THEME C — Graph neural networks for brain connectomes

Your proposal's CNN-GNN architecture sits in this lineage.

13. ✅ **Parisot, S., Ktena, S.I., Ferrante, E., Lee, M., Guerrero, R., Glocker, B., & Rueckert, D. (2018).** Disease prediction using graph convolutional networks: Application to Autism Spectrum Disorder and Alzheimer's disease. *Medical Image Analysis*, 48, 117–130.
    → 🔴 **The population-graph GCN paper.** Subjects are nodes; edges encode phenotypic + imaging similarity. 70.4% on ABIDE, 80% on ADNI. Code: github.com/parisots/population-gcn.
    → **Important conceptual distinction for your report:** there are *two* completely different ways to build a graph here — (a) **population graph**: nodes = people (Parisot); (b) **brain graph**: nodes = ROIs (BrainGNN). Your proposal implies (b). Know which you're doing and say so explicitly.

14. ⚪ **Kipf, T.N., & Welling, M. (2017).** Semi-supervised classification with graph convolutional networks. *ICLR*.
    → 🔴 The original GCN. Read it — it's short, and the layer equation is exactly the two-line implementation in your project guide. Cite for your GCN layer.

15. ⚪ **Veličković, P., et al. (2018).** Graph attention networks. *ICLR*.
    → 🟡 The GAT paper. Cite if you implement attention.

16. ✅ **Li, X., Zhou, Y., Dvornek, N., Zhang, M., Gao, S., Zhuang, J., Scheinost, D., Staib, L.H., Ventola, P., & Duncan, J.S. (2021).** BrainGNN: Interpretable brain graph neural network for fMRI analysis. *Medical Image Analysis*, 74, 102233.
    → 🔴 **The brain-graph GNN with built-in interpretability.** ROI-level nodes, ROI-aware pooling, identifies salient regions. This is the closest published architecture to what you're building, and its interpretability approach complements your SHAP plan.

17. ⚪ **Kawahara, J., Brown, C.J., Miller, S.P., et al. (2017).** BrainNetCNN: Convolutional neural networks for brain networks; towards predicting neurodevelopment. *NeuroImage*, 146, 1038–1049.
    → 🟡 Edge-to-Edge, Edge-to-Node, Node-to-Graph layers designed specifically for connectivity matrices. **Use as your deep-learning baseline** — it's the standard comparator in this literature.

18. ⚪ **Jiang, H., et al. (2020).** Hi-GCN: A hierarchical graph convolution network for graph embedding learning of brain network and brain disorders prediction. *Computers in Biology and Medicine*.
    → Combines brain-graph and population-graph levels. 🟡

19. ⚪ **Rakhimberdina, Z., & Murata, T. (2020).** Population graph-based multi-model ensemble method for diagnosing autism spectrum disorder. *Sensors*. ⚪

20. ⚪ **Zhang, et al. (2022).** Exploring interpretable graph convolutional networks for autism spectrum disorder diagnosis (FSL-BrainNet). *Int. J. Computer Assisted Radiology and Surgery*. doi:10.1007/s11548-022-02780-3
    → Jointly learns node features and clean graph structure; identifies salient regions and subnetworks. 🟡

21. ⚪ **A heterogeneous graph convolutional attention network method for classification of autism spectrum disorder.** *BMC Bioinformatics* (2023). PMC10536734
    → 🟡 Useful for its related-work section, which surveys the GCN-for-ASD lineage compactly. A good shortcut for your own related-work section.

22. ✅ **"Rethinking Functional Brain Connectome Analysis: Do Graph Deep Learning Models Help?"** (2025). arXiv:2501.17207
    → 🔴 **Read this one.** The skeptical counterpoint: it questions whether graph deep learning actually beats simpler baselines on functional connectomes. Confirms ABIDE via PCP is publicly accessible without restriction. **If your GNN doesn't beat XGBoost, this paper is your defence** — you'll be reporting a known phenomenon, not a failure.

23. ⚪ **Enhanced Graph Convolutional Network with Chebyshev Spectral Graph and Graph Attention for ASD Classification** (2025). arXiv:2511.22178
    → Recent multi-branch GCN + GAT on ABIDE I with multimodal + phenotypic data: 74.82% accuracy, AUC 0.82. 🟡 **A far more realistic target than 96.8%** — and it's recent, so it's a fair current benchmark to cite.

---

## 5. THEME D — Multimodal sMRI + fMRI fusion

24. ✅ **Wang, L., Li, X., Yuan, J., & Chen, Y. (2025).** DAGMNet: Dual-Branch Attention-Pruned Graph Neural Network for Multimodal sMRI and fMRI Fusion in Autism Prediction. *Biomedicines*, 13(9), 2168. doi:10.3390/biomedicines13092168. PMC12467520
    → 🔴 **Your proposal's architectural template.** Three components: adaptive shared/modality-specific representation learning, edge-sensitive pooling for multi-atlas refinement, and dynamic graph pruning to counter GNN oversmoothing. Reports 91.59% accuracy / 96.80% AUC on ABIDE-I, plus ADNI validation. Inference ~0.021 s per subject.
    → **Read it critically.** Specifically check: did they use leave-one-site-out? Was feature selection inside cross-validation? The gap between their 96.8% and Abraham's 67% is not explained by architecture alone.

25. ⚪ **Aghdam, M.A., Sharifi, A., & Pedram, M.M. (2018).** Combination of rs-fMRI and sMRI data to discriminate autism spectrum disorders in young children using deep belief networks. *Journal of Digital Imaging*, 31, 895–903. ⚠️ UNVERIFIED (from your proposal — plausible, but check)
    → Early multimodal fusion precedent. 🟡

26. ⚪ **Song, et al. (2025).** Dual transformer model combined with GCN for sMRI and fMRI processing.
    → Reported max accuracy 79.47%. 🟡 Another realistic-number data point.

27. ⚪ **A Hierarchical Feature Extraction and Multimodal Deep Feature Integration-Based Model for ASD Identification** (2025).
    → Multiscale rs-fMRI extraction; explicitly notes it excludes behavioural/molecular data — i.e. **exactly the gap your anxiety layer fills.** 🟡 Good citation for your novelty argument.

28. ⚪ **An Adaptive Transfer Learning Framework for Multimodal Autism Spectrum Disorder Diagnosis** (2025). *Life*, 15(10), 1524.
    → Fuses behavioural + genetic + sMRI; Hybrid CNN-GNN on sMRI, gradient boosting on genetic data (86.6%). ⚪

29. ⚪ **Multi-modal Multi-kernel Graph Learning for Autism Prediction and Biomarker Discovery** (2025). ⚪

30. ⚪ **Multi-Site rs-fMRI Domain Alignment for ASD Auxiliary Diagnosis Based on Hyperbolic Space** (2025). arXiv:2502.05493
    → Useful comparison table of BrainNetCNN / Hi-GCN / population-GCN baselines. 🟡

---

## 6. THEME E — Anxiety in autism: the clinical foundation

31. ✅ **White, S.W., Oswald, D., Ollendick, T., & Scahill, L. (2009).** Anxiety in children and adolescents with autism spectrum disorders. *Clinical Psychology Review*, 29(3), 216–229. doi:10.1016/j.cpr.2009.01.003
    → 🔴 Your prevalence citation. Source of the 11–84% range. Already in your proposal, correctly. 

32. ✅ **Kerns, C.M., & Kendall, P.C. (2012).** The presentation and classification of anxiety in autism spectrum disorder. *Clinical Psychology: Science and Practice*, 19(4), 323–347.
    → 🔴 **Read this properly — it directly determines your anxiety label design.** It distinguishes *typical* anxiety presentations from *atypical, ASD-related* ones (e.g. anxiety about special interests, unusual phobias). This matters because if you define "anxiety" from a CBCL cutoff, you may capture only the typical presentations and miss the atypical ones — a limitation you should name explicitly.

33. ⚪ **Kerns, C.M., Renno, P., Storch, E.A., Kendall, P.C., & Wood, J.J. (eds.).** *Anxiety in Children and Adolescents with Autism Spectrum Disorder: Evidence-Based Assessment and Treatment.* See especially Kent, R. & Simonoff, E., Ch. 2, "Prevalence of Anxiety in Autism Spectrum Disorders." 🟡

34. ⚪ **Rosen, T.E., Mazefsky, C.A., Vasa, R.A., & Lerner, M.D. (2018).** Co-occurring psychiatric conditions in autism spectrum disorder. *International Review of Psychiatry*, 30(1), 40–61.
    → 🟡 Broad comorbidity overview. Good for your introduction.

35. ⚪ **Vasa, R.A., Kerns, C.M., et al.** Priorities for Advancing Research on Youth with ASD and Co-occurring Anxiety. *Journal of Autism and Developmental Disorders* (2018). doi:10.1007/s10803-017-3320-0
    → 🟡 **A researcher-consensus list of open questions in exactly your area.** Mine this for your "future work" section and to justify your gap statement.

36. ⚪ **Rosen, et al.** Development of the Parent-Rated Anxiety Scale for youth with ASD. *JAACAP*.
    → Relevant to why generic scales (CBCL/MASC) are imperfect in ASD. ⚪

37. ⚪ **Vaillancourt, T., et al. (2017).** The Screen for Child Anxiety-Related Emotional Disorders is sensitive but not specific in identifying anxiety in children with high-functioning ASD. PMC5539181
    → 🟡 **Directly relevant methodological caveat.** Shows SCARED correlates with CBCL Anxious/Depressed but has specificity problems in ASD due to symptom overlap. Cite when you justify your choice of anxiety instrument.

38. ⚪ **Uljarević, M., Hedley, D., Rose-Foley, K., et al.** Anxiety and Depression from Adolescence to Old Age in ASD. *Journal of Autism and Developmental Disorders*. ⚪

39. ⚪ **Yarger, H.A., & Redcay, E. (2020).** A conceptual model of risk and protective factors associated with internalizing symptoms in ASD: A scoping review. *Development and Psychopathology*, 32, 1254–1272. 🟡

---

## 7. THEME F — Anxiety and the brain in autism ⭐ YOUR CORE NOVELTY LITERATURE

**This section is the most important in the document and is almost entirely absent from your proposal's reference list.** These are the papers your contribution must be positioned against.

40. ✅ **Herrington, J.D., Miller, J.S., Pandey, J., & Schultz, R.T. (2016).** Anxiety and social deficits have distinct relationships with amygdala function in autism spectrum disorder. *Social Cognitive and Affective Neuroscience*, 11(6), 907–914. doi:10.1093/scan/nsw015. PMID 26865425
    → 🔴🔴 **THE single most important paper for your entire framing. Read it twice.**
    → 81 youth with ASD + 67 non-ASD controls, face-recognition fMRI paradigm. Only the ASD subgroup with **low** anxiety (n=28) showed reduced amygdala activation relative to controls. Within the ASD group, anxiety symptoms correlated **positively** with amygdala activity while core ASD symptoms correlated **negatively**. The authors conclude that the long-reported amygdala hypoactivation in ASD can be *masked* by comorbid anxiety, and that amygdala activity is a hybrid social/emotional signal.
    → **This is empirical proof that your project's premise is correct.** Anxiety genuinely confounds ASD brain findings, and two decades of literature may be partly measuring anxiety. Build your introduction around this paper. (Note: they used SCARED — in their own sample, not ABIDE.)

41. ✅ **Yarger, H.A., Nordahl, C.W., & Redcay, E. (2021).** Examining associations between amygdala volumes and anxiety symptoms in autism spectrum disorder. *Biological Psychiatry: Cognitive Neuroscience and Neuroimaging*. doi:10.1016/j.bpsc.2021.10.010. PMID 34688922 (erratum: PMC9058996)
    → 🔴 **The closest published study to your Track 2b — and it is a null result.**
    → n = 123 autistic + 171 non-autistic children aged 5–13, from University of Maryland plus **three ABIDE-II sites: Georgetown (GU), Kennedy Krieger (KKI), and NYU** — these are precisely the sites with CBCL anxiety data. Amygdala volumes were residualised for site, hemispheric volume, age, sex and FSIQ using **ComBat**.
    → Findings: clinically significant anxiety did **not** differentiate amygdala volumes across groups; no association between amygdala volume and anxiety scores within the autism sample; meta-analytic and **Bayes factor** estimates supported the null. Age, sex and autism severity did not moderate.
    → **Read this before you promise your supervisor any structural finding.** It also hands you your methods template: same sites, same instrument, same harmonisation approach.

42. ✅ **Andrews, D.S., Aksman, L., Kerns, C.M., Lee, J.K., et al. (2022).** Association of amygdala development with different forms of anxiety in autism spectrum disorder. *Biological Psychiatry*, 91(11), 977–987.
    → 🔴 **The developmental/longitudinal counterpart.** Crucially, it separates *traditional* DSM anxiety from *ASD-distinct* anxiety — and finds different amygdala trajectories. Directly operationalises the Kerns & Kendall distinction (#32) in neuroimaging.
    → **This is your strongest argument for why a single binary "anxious/not anxious" label is inadequate**, and a great limitation to discuss.

43. ⚪ **"Anxiety–Amygdala Associations: Novel Insights From the First Longitudinal Study of Autistic Youth With Distinct Anxiety."** *Biological Psychiatry* (2022) commentary/report.
    → 🟡 Reinforces that **longitudinal design is what licenses causal language** — the core of Flag 3 in your project guide.

44. ⚪ **Herrington, J.D., Maddox, B.B., Kerns, C.M., et al. (2017).** Amygdala volume differences in autism spectrum disorder are related to anxiety. *Journal of Autism and Developmental Disorders*, 47, 3682–3691.
    → 🟡 Reports a **positive** structural finding, in contrast to Yarger et al. (#41). Cite both — the disagreement is itself worth discussing.

45. ⚪ **Juranek, J., Filipek, P.A., Berenji, G.R., et al. (2006).** Association between amygdala volume and anxiety level: MRI study in autistic children. *Journal of Child Neurology*, 21, 1051–1058.
    → ⚪ The earliest study in this line; found enlarged right amygdala associated with higher CBCL anxious/depressed scores. Cite for historical framing.

46. ⚪ **"Functional brain abnormalities associated with comorbid anxiety in autism spectrum disorder."** *Development and Psychopathology* (2020), 32, 1273–1286.
    → 🔴 **Functional (not structural) anxiety-comorbidity findings in ASD — the closest match to your fMRI branch.** Track this one down and read it.

47. ⚪ **Top, D.N. Jr., et al. (2016).** Atypical amygdala response to fear conditioning in autism spectrum disorder. *Biological Psychiatry: CNNI*, 1, 308–315.
    → 🟡 n=20 ASD + 19 controls. Anxiety related to impaired threat-vs-safe discrimination in the amygdala. Mechanistic support for your DAG.

48. ⚪ **"Unmasking Anxiety in Autism: Explicit and Implicit Threatening Face Stimuli Dissociate Amygdala-centered Functional Connectivity."** bioRxiv (2020).
    → 🟡 31 ASD + 30 controls, backward-masking paradigm. Proposes that anxiety in ASD may arise via a dissociation between explicit and implicit threat processing. Interesting mechanistic hypothesis for your discussion.

49. ⚪ **Banker, S.M., et al. (2025).** A unique neural and behavioral profile of social anxiety in autism. *Research Square* preprint / subsequent publication.
    → 🟡 Large online behavioural sample (ASD n=575, control n=357) plus neural profiling. Social anxiety and ASD are sometimes mutually misdiagnosed — directly relevant to your confound argument.

50. ⚪ **Green, S.A., Hernandez, L., Tottenham, N., et al.** Neurobiology of sensory overresponsivity in youth with ASD. *JAMA Psychiatry* / related.
    → 🟡 Sensorilimbic hyperresponsivity to aversive stimuli; amygdala activity correlated with sensory overresponsivity **after controlling for anxiety** — a good example of the analytic discipline you should imitate. Also documents failure of habituation.

51. ⚪ **Swartz, J.R., Wiggins, J.L., Carrasco, M., Lord, C., & Monk, C.S.** Amygdala habituation and prefrontal functional connectivity in youth with autism spectrum disorders. *JAACAP*.
    → 🟡 The amygdala habituation literature your proposal references. ⚪

52. ⚪ **Etkin, A., & Wager, T.D. (2007).** Functional neuroimaging of anxiety: A meta-analysis of emotional processing in PTSD, social anxiety disorder, and specific phobia. *American Journal of Psychiatry*, 164, 1476–1488.
    → 🟡 **The canonical anxiety-neuroimaging meta-analysis.** Cite this to establish the anxiety side of the circuit overlap — it's what lets you argue the two conditions share substrates.

53. ⚪ **Roy, A.K., et al. (2013).** Intrinsic functional connectivity of amygdala-based networks in adolescent generalized anxiety disorder. *JAACAP*. And: **"Aberrant amygdala functional connectivity at rest in pediatric anxiety disorders."** PMC4272798
    → 🟡 Pediatric anxiety rs-fMRI: amygdala–insula hyperconnectivity, amygdala–vmPFC and amygdala–PCC hypoconnectivity. **These are the exact edges you should put in your hypothesis-driven ROI subset.**

54. ⚪ **"Dysfunction of resting-state functional connectivity of amygdala subregions in drug-naïve patients with generalized anxiety disorder."** PMC8548605; and **"Illness severity moderated association between trait anxiety and amygdala-based functional connectivity in GAD."** PMC8044966
    → 🟡 Amygdala subregions (basolateral vs centromedial) and the state-vs-trait anxiety distinction. Note: your ABIDE anxiety measures are closer to **trait**; STAI-style state anxiety is not available. Worth naming as a limitation.

55. ⚪ **Kovacevic, M., et al. (2023).** Amygdala volumes in autism spectrum disorders: Meta-analysis of MRI studies. *Review Journal of Autism and Developmental Disorders*, 10, 169–183.
    → 🟡 Meta-analysis reporting bilateral amygdala and left hippocampal volume increases in ASD, but flagging high heterogeneity and evidence of publication bias (Egger's test). **Good honest-appraisal citation.**

---

## 8. THEME G — Predicting *severity*, not just diagnosis

Your project predicts severity. Most ABIDE ML papers only do binary diagnosis. **These are the papers that make your Task C credible, and they're missing from your proposal.**

56. ✅ **Jung, M., et al. (2022).** Sparse hierarchical representation learning on functional brain networks for prediction of autism severity levels. *Frontiers in Neuroscience*, 16, 935431. PMC9301472
    → 🔴 **The best severity-regression reference for you.** GCN with hierarchical self-attention graph pooling on ABIDE, predicting ADOS severity and ADI-R subscores. Best configuration (SHEN atlas, Tikhonov correlation, identity-adjacency embedding) reached **MAE 0.96, r = 0.61** for individual severity; beat the BrainNetCNN benchmark (MAE 1.30, r = 0.43). Also visualises the most contributive connections.
    → **Use their metrics as your Task C target.** MAE ~1.0 on a 1–10 severity scale is what good looks like — not R² of 0.8.

57. ⚪ **"Connectome-based prediction of the severity of autism spectrum disorder."** *Psychoradiology* (2023). doi:10.1093/psyrad/kkad027. PMC10917386
    → 🔴 **Read the methods — this is the simplest severity approach you could actually implement in a week.** Connectome-based Predictive Modelling (CPM): Dosenbach-160 atlas, control for age/sex/head motion, correlate each edge with ADOS, threshold at p<0.01, sum positive-network and negative-network edge strengths, feed those two scalars into a linear regression. n=151 ASD training, replication in 172, specificity check in 36 controls, leave-one-out CV.
    → CPM is far more robust at your sample size than a deep model. **Consider making this your primary Task C method.**

58. ⚪ **"The functional brain organization of an individual predicts measures of social abilities in autism spectrum disorder."** bioRxiv (2018) / subsequent journal version.
    → 🟡 CPM on **ABIDE** predicting both SRS and ADOS, with leave-one-subject-out and split-half CV; identifies anatomically distinct networks for the two scores. Almost exactly your Task C on your dataset.

59. ⚪ **"Neural biomarkers distinguish severe from mild autism spectrum disorder among high-functioning individuals."** *Frontiers in Human Neuroscience* (2021), 15, 657857. PMC8134539
    → 🟡 CCA + hierarchical clustering to find ASD subgroups, then SVM to predict ADOS: r = 0.30 (n=260), validated at r = 0.35 (n=29). **A sobering, realistic effect size.** Also notes that small effect sizes in ASD rs-fMRI are what block diagnostic use.

60. ⚪ **"Predicting individual autistic symptoms for patients with ASD using interregional morphological connectivity."** (2024)
    → 🟡 **Structural** analogue: morphological connectivity networks + SVR to predict ADOS social and behaviour subscores. Relevant to your sMRI branch.

61. ⚪ **Shen, X., Finn, E.S., Scheinost, D., et al. (2017).** Using connectome-based predictive modeling to predict individual behavior from brain connectivity. *Nature Protocols*, 12, 506–518.
    → 🔴 **The CPM protocol paper.** If you adopt CPM (recommended), this is your method citation and step-by-step guide.

---

## 9. THEME H — Site harmonisation (do not skip)

62. ✅ **Fortin, J.-P., Cullen, N., Sheline, Y.I., et al. (2018).** Harmonization of cortical thickness measurements across scanners and sites. *NeuroImage*, 167, 104–120. doi:10.1016/j.neuroimage.2017.11.024. PMID 29155184
    → 🔴 **Your ComBat citation for structural features.** Adapts ComBat from genomics; adds a site-specific scaling factor so it corrects both additive (mean) and multiplicative (variance) site effects, using empirical Bayes across features — which helps at small sample sizes. Shows it removes scanner variability while increasing statistical power and reproducibility.

63. ⚪ **Johnson, W.E., Li, C., & Rabinovic, A. (2007).** Adjusting batch effects in microarray expression data using empirical Bayes methods. *Biostatistics*, 8(1), 118–127.
    → 🔴 The original ComBat. Cite alongside Fortin. Short.

64. ⚪ **Fortin, J.-P., et al. (2017).** Harmonization of multi-site diffusion tensor imaging data. *NeuroImage*, 161, 149–170. ⚪

65. ✅ **"Multi-site harmonization of MRI data uncovers machine-learning discrimination capability in barely separable populations: An example from the ABIDE dataset."** *NeuroImage: Clinical* (2022), 35, 103042.
    → 🔴 **Read this — it is the most direct evidence for your expected-performance argument.** Analyses 2,226 ABIDE I+II subjects (1,060 ASD / 1,166 TD) from 26 institutions spanning **39 distinct site-samples**, all through FreeSurfer `recon-all`. Notes that earlier work (Haar et al., 2016) found case-control accuracy **below 60%** from anatomical measures, suggesting limited diagnostic utility, and attributes much of the wild variability in published ABIDE performance to site batch effects.
    → The phrase "barely separable populations" is the honest framing of this problem. Cite it in your Limitations.

66. ⚪ **Haar, S., Berman, S., Behrmann, M., & Dinstein, I. (2016).** Anatomical abnormalities in autism? *Cerebral Cortex*, 26(4), 1440–1452.
    → 🟡 Large ABIDE structural analysis finding minimal group differences. **The most important "negative result" paper for your sMRI branch.** Read it before promising structural findings.

67. ⚪ **Radua, J., et al. (2020).** Increased power by harmonizing structural MRI site differences with the ComBat batch adjustment method in ENIGMA. *NeuroImage*, 218, 116956. 🟡

68. ⚪ **Pomponio, R., et al. (2020).** ComBat-GAM — models non-linear age effects with a generalized additive model. *NeuroImage*.
    → 🟡 Relevant because ABIDE spans ages 5–64, where age effects are strongly non-linear. Consider ComBat-GAM over vanilla ComBat.

69. ⚪ **Beer, J.C., et al. (2020).** Longitudinal ComBat. *NeuroImage*, 220, 117129. ⚪
70. ⚪ **"DeepComBat: A statistically motivated, hyperparameter-robust, deep learning approach to harmonization of neuroimaging data."** *Human Brain Mapping*. PMC10168207. ⚪
71. ⚪ **"Detect and Correct Bias in Multi-Site Neuroimaging Datasets."** arXiv:2002.05049
    → 🟡 Good tutorial-style overview of harmonisation options, including the crucial warning that over-correcting for site can remove genuine biological variability.

---

## 10. THEME I — Causal inference and mediation

72. ✅ **Imai, K., Keele, L., & Tingley, D. (2010).** A general approach to causal mediation analysis. *Psychological Methods*, 15(4), 309–334.
    → 🔴 **The ACME/ADE framework that `pingouin.mediation_analysis` implements.** Read this so you can define your terms correctly. Also: Tingley, D., Yamamoto, T., Hirose, K., Keele, L., & Imai, K. (2014). mediation: R package for causal mediation analysis. *Journal of Statistical Software*, 59(5).

73. ⚪ **VanderWeele, T.J. (2015).** *Explanation in Causal Inference: Methods for Mediation and Interaction.* Oxford University Press. Also: VanderWeele, T.J. (2016). Mediation analysis: A practitioner's guide. *Annual Review of Public Health*, 37, 17–32.
    → 🔴 **Read the practitioner's guide (it's short) — this is what lets you write Flag 3 defensibly.** Sequential ignorability, sensitivity analysis for unmeasured confounding, and why cross-sectional mediation is fragile.

74. ⚪ **Pearl, J. (2009).** *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press. Or the shorter **Pearl & Mackenzie (2018), *The Book of Why***.
    → 🟡 For DAGs, confounders, mediators, colliders. The *Book of Why* is genuinely readable and enough for your purposes.

75. ✅ **"How causal inference tools can support debiasing of machine learning models for meaningful brain-based predictions."** medRxiv (2024/2025). doi:10.1101/2024.09.20.24314055
    → 🔴 **Written for exactly your situation.** Argues that neuroimaging ML commonly picks confounders heuristically ("age, sex") or by correlation, risking confusion with colliders and mediators — and proposes a three-step framework: build a domain-knowledge DAG, apply graph-theoretic rules to identify valid deconfounders, then handle unmeasured variables.
    → It explicitly names **comorbid anxiety** as an example of a variable that influences both imaging features and psychiatric diagnosis. That's your project in one sentence. Use this to justify your covariate set.

76. ⚪ **Richens, J.G., Lee, C.M., & Johri, S. (2020).** Improving the accuracy of medical diagnosis with causal machine learning. *Nature Communications*, 11, 3923. doi:10.1038/s41467-020-17419-7 ⚠️ from your proposal — plausible, verify
    → 🟡 The general "causal ML beats associative ML for diagnosis" argument. Good for your introduction's motivation.

77. ⚪ **Lindquist, M.A. (2012).** Functional causal mediation analysis with an application to brain connectivity. *Journal of the American Statistical Association*, 107(500), 1297–1309.
    → 🟡 Mediation specifically with neuroimaging mediators. ⚪

78. ⚪ **"Causal mediation analysis with one or multiple mediators: a comparative study."** arXiv:2505.07323
    → 🟡 Benchmarks mediation estimators and provides a Python package (`med_bench`). Useful if you want more than pingouin's coefficient-product approach; also notes DoWhy's `mediation.two_stage_regression`.

79. ⚪ **"Bayesian structured mediation analysis with unobserved confounders."** arXiv:2407.04142
    → ⚪ Advanced. Skim the introduction only — it states clearly why the no-unobserved-confounding assumption is the weak point in imaging mediation, which is useful phrasing for your limitations.

80. ⚪ **Sharma, A., & Kiciman, E. (2020).** DoWhy: An end-to-end library for causal inference. arXiv:2011.04216. And **DoWhy-GCM** (arXiv:2206.06821).
    → 🟡 Your software citation if you use DoWhy.

---

## 11. THEME J — Explainability

81. ✅ **Lundberg, S.M., & Lee, S.-I. (2017).** A unified approach to interpreting model predictions. *NeurIPS 30*.
    → 🔴 The SHAP paper. Your XAI method citation.

82. ✅ **"Identification of critical brain regions for autism diagnosis from fMRI data using explainable AI: an observational analysis of the ABIDE dataset."** *eClinicalMedicine* (2025). doi:10.1016/j.eclinm.2025.103384. PMC12573454
    → 🔴 **The best XAI-on-ABIDE paper, and it has public code.** 884 ABIDE I participants (408 ASD / 476 TD), 7–64 years, 17 sites, after mean-FD filtering at 0.2 mm — the same QC threshold your project uses. Stacked sparse autoencoder pipeline, and it **systematically benchmarks interpretability methods** for fMRI, which nobody had done. Found visual-processing regions consistently important and cross-checked biomarkers against independent genetic and neuroimaging evidence.
    → Code: **github.com/v1dya/XAI-for-ASD**. Read the repo — it will save you days on your SHAP module.

83. ⚪ **"From Predictions to Explanations: Explainable AI for Autism Diagnosis and Identification of Critical Brain Regions."** arXiv:2509.10523
    → 🟡 Compares three XAI techniques — saliency maps, Grad-CAM, and SHAP — on ABIDE plus Healthy Brain Network data. Useful if you want to justify choosing SHAP over alternatives.

84. ⚪ **"Explainable AI uncovers novel EEG microstate candidate neurophysiological markers for autism spectrum disorder."** PMC12913458
    → 🟡 Different modality, but a clean template for the workflow you're planning: XGBoost → SHAP top-20 features → retrain on SHAP-selected features → validate with Mann-Whitney U and effect sizes. It also articulates well why SHAP's global *and* local explanations suit biomarker work.

85. ⚪ **Ribeiro, M.T., Singh, S., & Guestrin, C. (2016).** "Why should I trust you?": Explaining the predictions of any classifier (LIME). *KDD*. ⚪

---

## 12. THEME K — Gut microbiome (Phase 2 continuity)

Include a short subsection citing these so your report reads as part of a coherent three-phase programme.

86. ✅ **Peralta-Marzal, L.N., Rojas-Velazquez, D., Rigters, D., Prince, N., Garssen, J., Kraneveld, A.D., Perez-Pardo, P., & Lopez-Rincon, A. (2024).** A robust microbiome signature for autism spectrum disorder across different studies using machine learning. *Scientific Reports*, 14, 814. doi:10.1038/s41598-023-50601-7. PMC10774349
    → 🟡 ✅ Correctly cited in your proposal. Sibling-controlled design, recursive ensemble feature selection on 16S rRNA data, 26 discriminating bacterial taxa, validated across two independent Chinese cohorts. AUC roughly 81.6% (US cohort) and ~74–75% (Chinese cohorts). Reported signals include reduced bifidobacteria, atypical *Clostridia* levels, and reduced *Butyricicoccus*.
    → **Note the AUCs: ~0.75–0.82 with a well-validated multi-cohort design.** Another data point that ~0.97 is not the norm in this field.

87. ⚪ **"Contributions of Artificial Intelligence to Analysis of Gut Microbiota in Autism Spectrum Disorder: A Systematic Review."** (2024). PMC11352523
    → 🟡 Your Phase 2 systematic-review citation.

88. ⚪ **Cryan, J.F., et al. (2019).** The microbiota-gut-brain axis. *Physiological Reviews*, 99(4), 1877–2013.
    → 🟡 The mechanistic bridge between Phase 2 and Phase 3. Cite in your future-work section — it's the theoretical justification for wanting trimodal fusion even though you can't do it yet.

89. ⚪ **Olaguez-Gonzalez, J.M., et al. (2024).** Assessment of machine learning strategies for simplified detection of ASD based on gut microbiome composition. *Neural Computing and Applications*, 36, 8163–8180. ⚪

90. ⚪ **Dan, Z., et al. (2020).** Altered gut microbial profile is associated with abnormal metabolism activity of autism spectrum disorder. *Gut Microbes*, 11, 1246–1267. ⚪

---

## 13. THEME L — Reviews, meta-analyses, and the skeptical literature

Cite at least two skeptical papers. It signals maturity.

91. ⚪ **"Accuracy of machine learning algorithms for the diagnosis of autism spectrum disorder: Systematic review and meta-analysis of brain MRI studies."** *JMIR* / PMC6942187
    → 🔴 **A meta-analysis of exactly what you're doing.** Read it to position your expected accuracy honestly and to see how heterogeneous published ML pipelines are.

92. ⚪ **Vargason, T., et al. (2020).** On the heterogeneity of reported ASD classification performance. (Referenced in the harmonisation literature.)
    → 🟡 Documents the wide spread of published ABIDE accuracies. ⚪

93. ⚪ **Button, K.S., Ioannidis, J.P.A., Mokrysz, C., et al. (2013).** Power failure: Why small sample size undermines the reliability of neuroscience. *Nature Reviews Neuroscience*, 14, 365–376.
    → 🔴 Short, essential, and directly applicable given your anxiety subsample will be small. Cite in Limitations.

94. ⚪ **Marek, S., Tervo-Clemmens, B., Calabro, F.J., et al. (2022).** Reproducible brain-wide association studies require thousands of individuals. *Nature*, 603, 654–660.
    → 🔴 **Read the abstract and main figure at minimum.** Brain–behaviour effect sizes are far smaller than the field assumed, and studies with hundreds of subjects are underpowered for them. This is the most important single piece of context for interpreting your results modestly — and citing it pre-empts the criticism.

95. ⚪ **Poldrack, R.A., Huckins, G., & Varoquaux, G. (2020).** Establishment of best practices for evidence for prediction: A review. *JAMA Psychiatry*, 77(5), 534–540.
    → 🔴 Short checklist-style paper on prediction methodology in psychiatry. **Read it and follow it.** It will directly improve your marks.

96. ⚪ **Kapur, S., Phillips, A.G., & Insel, T.R. (2012).** Why has it taken so long for biological psychiatry to develop clinical tests? *Molecular Psychiatry*, 17, 1174–1179.
    → 🟡 Good framing for your introduction and discussion. ⚪

97. ⚪ **Uddin, L.Q., Supekar, K., & Menon, V. (2013).** Reconceptualizing functional brain connectivity in autism from a developmental perspective. *Frontiers in Human Neuroscience*, 7, 458.
    → 🟡 The under/over-connectivity debate in ASD, and why age matters. Relevant because ABIDE spans 5–64 years. ⚪

98. ⚪ **Power, J.D., Barnes, K.A., Snyder, A.Z., Schlaggar, B.L., & Petersen, S.E. (2012).** Spurious but systematic correlations in functional connectivity MRI networks arise from subject motion. *NeuroImage*, 59(3), 2142–2154.
    → 🔴 **Read this.** The head-motion problem. Autistic children move more than controls, so motion can masquerade as a group difference. This paper is why you exclude mean FD > 0.2 and covary motion. Non-optional for your methods justification.

99. ⚪ **Ciric, R., Wolf, D.H., Power, J.D., et al. (2017).** Benchmarking of participant-level confound regression strategies for the control of motion artifact. *NeuroImage*, 154, 174–187.
    → 🟡 Where the "Power criteria" and "Ciric criteria" for motion exclusion come from. ⚪

100. ⚪ **Wang, M.L., et al. (2025).** The role of machine learning in ASD assessment and management. *Pediatric Research*. doi:10.1038/s41390-025-04566-0 ⚠️ from your proposal — verify
     → 🟡 Recent clinical-facing review. ⚪

---

## 14. What is *missing* from this literature = your gap statement

This is the most useful part of a literature review. Having read all of the above, here is the honest gap, which you can adapt almost verbatim:

> **Gap statement (draft).** Two literatures have developed largely in parallel. The first applies increasingly sophisticated machine learning — autoencoders, graph convolutional networks, attention-based multimodal fusion — to ABIDE for ASD classification (Abraham et al., 2017; Heinsfeld et al., 2018; Parisot et al., 2018; Li et al., 2021; Wang et al., 2025). Almost none of these models stratify by, adjust for, or even report psychiatric comorbidity. The second literature demonstrates that comorbid anxiety substantively alters the neural picture in autism: Herrington et al. (2016) showed that amygdala hypoactivation in ASD is detectable only in the low-anxiety subgroup and that anxiety and core autism symptoms relate to amygdala activity in *opposite* directions, while Andrews et al. (2022) showed distinct amygdala developmental trajectories for traditional versus ASD-specific anxiety. Yet this literature is almost entirely univariate and hypothesis-driven, using single-region analyses rather than multivariate prediction.
>
> Consequently, no study has, to our knowledge, (i) stratified a large multi-site ASD neuroimaging cohort by anxiety comorbidity within a predictive-modelling framework, (ii) tested whether anxiety-related features improve *severity* rather than diagnostic prediction, or (iii) applied a formal mediation framework to the anxiety–brain–severity path with explicit statement of identifying assumptions. Given that Yarger et al. (2021) report null structural findings in the largest sample examined to date, and that Marek et al. (2022) show brain–behaviour effects are smaller than previously assumed, this question warrants a well-powered, methodologically conservative treatment rather than a further increment in reported classification accuracy.

Adjust the confidence to match what your week-1 audit actually finds.

---

## 15. Finding more papers yourself

**Best single technique — backward and forward citation chaining.** Take Herrington et al. (2016) and Yarger et al. (2021). Read their reference lists (backward). Then look up who has cited them since (forward, via Google Scholar's "Cited by" or Semantic Scholar). Two rounds of this will surface nearly everything relevant in your niche. It's far more efficient than keyword searching.

**Useful search strings:**
- `autism anxiety comorbidity neuroimaging amygdala`
- `ABIDE anxiety stratification machine learning`
- `ASD severity prediction connectome ADOS`
- `mediation analysis neuroimaging comorbidity cross-sectional`
- `leave-one-site-out cross-validation ABIDE`

**Tools:** Google Scholar (alerts on your key terms), Semantic Scholar (best citation graph, free API), Connected Papers (visual neighbourhood map — start it from Herrington 2016), PubMed (clinical literature), and arXiv/bioRxiv/medRxiv for methods preprints.

**Reference manager: use Zotero.** Free, has a browser plugin that captures citations in one click, and syncs a shared library between the two of you. Set it up in week 1, add papers as you read them, and generate the bibliography automatically. Do **not** hand-type 60 references in week 8.

---

## 16. Your 12 CORE papers, in reading order

If you read nothing else, read these, in this sequence:

| # | Paper | Why | Week |
|---|---|---|---|
| 1 | Di Martino et al. (2014) — ABIDE I | Know your data | 1 |
| 2 | Di Martino et al. (2017) — ABIDE II | Where anxiety data lives | 1 |
| 3 | **Herrington et al. (2016)** | **Your entire premise** | 1 |
| 4 | **Yarger et al. (2021)** | Closest prior work; null result | 1 |
| 5 | Kerns & Kendall (2012) | How to define "anxiety" | 2 |
| 6 | **Abraham et al. (2017)** | **Your pipeline + benchmark** | 2 |
| 7 | Power et al. (2012) | Motion confound | 2 |
| 8 | Fortin et al. (2018) + Johnson et al. (2007) | ComBat | 3 |
| 9 | Poldrack et al. (2020) | Prediction best practice | 3 |
| 10 | Shen et al. (2017) — CPM protocol | Your severity method | 4 |
| 11 | Marek et al. (2022) | Effect-size realism | 5 |
| 12 | VanderWeele (2016) — practitioner's guide | Mediation, done honestly | 6 |

Plus, when you get to those weeks: Parisot et al. (2018) and Li et al. (2021) for the GNN; Wang et al. (2025) DAGMNet for the architecture; the eClinicalMedicine XAI paper (2025) for SHAP; the ABIDE harmonisation paper (2022) for your limitations.

---

*Compiled August 2026. Entries marked ✅ VERIFIED were checked against publisher/PubMed records. Entries marked ⚠️ UNVERIFIED come from your proposal or from secondary reference lists — confirm authors, journal, volume and DOI before citing. Always verify a DOI resolves before it goes in a submitted reference list.*
