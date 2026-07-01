---
title: "Bias-Corrected Cross-Validation for Genomic Prediction"
date: "2023-09"
tags:
  - Genomics
  - Machine learning
  - Cross-validation
image: "/projects/CVc/cvc-cover.png"
github: "https://github.com/theLongLab/CVc_in_bioinformatics"
publication: "https://doi.org/10.1080/02664763.2026.2646570"
backHref: "/#project"
---

**Journal paper:** [The bias of using cross-validation in genomic predictions and its correction](https://doi.org/10.1080/02664763.2026.2646570)  
**DOI:** [10.1080/02664763.2026.2646570](https://doi.org/10.1080/02664763.2026.2646570)

#### Background

Cross-validation (CV) is one of the most common tools for estimating prediction error, but genomic data break a quiet assumption behind standard CV: samples and markers are not fully independent. Human genomes contain population structure, relatedness, and linkage disequilibrium (LD), so training and validation folds can share correlated genetic information.

In genomic prediction, this matters because models are often trained to predict phenotypes from high-dimensional genotype data. Methods such as **OLS**, **GLS**, **ridge regression**, **Lasso**, **elastic net**, **linear mixed models (LMM/gBLUP)**, and **Bayesian sparse linear mixed models (BSLMM)** can appear more accurate under CV than they are when deployed to a target population with a different dependence structure.

<figure class="article-picture">
	<img src="/projects/CVc/cvc-correlation-matrices.png" alt="Correlation matrices comparing real genomic data and synthetic independent SNP data" />
	<figcaption>Real genomic data show LD among SNPs, while the synthetic data preserve allele frequencies but remove the correlation structure.</figcaption>
</figure>

#### Contribution

This project studies CV bias in genomic prediction and adapts the **CVc** framework for correlated data to a practical set of genomic models. The central idea is to adjust the CV estimate with a covariance-based correction term that compares the dependence structure seen during cross-validation with the dependence structure of the prediction target.

The paper derives explicit correction forms for models whose predictions can be represented as linear functions of the observed responses:

- OLS
- GLS
- Ridge regression
- LMM/gBLUP
- BSLMM, approximately, through its mixed-model equivalence

Lasso and elastic net are included as genomic prediction baselines, but they are not corrected in this framework because their fitted values do not admit the same closed-form linear projection representation.

| Method   | CV bias observed | CVc correction available |
| --- | --- | --- |
| OLS | Yes | Yes |
| GLS | Yes | Yes |
| LMM/gBLUP | Yes | Yes |
| Ridge | Yes | Yes |
| BSLMM | Yes | Yes, approximately |
| Elastic net | Yes | No |
| Lasso | Yes | No |


You can also read the original derivation note here: [Cross-Validation for Correlated Data](/projects/CVc/Cross-Validation%20for%20Correlated%20Data.html).

#### Simulation Design

The simulation uses genotype data from the **1000 Genomes Project**. After filtering rare variants, 10,000 chromosome 1 SNPs were sampled and phenotypes were simulated with a mixture of larger and smaller genetic effects.

Two evaluation targets are used:

- A held-out test set drawn from the same genomic data, preserving the training data's LD and relatedness structure.
- A synthetic genotype dataset that preserves minor allele frequencies while removing LD and relatedness, representing a more independent generalization target.

<figure class="article-picture">
	<img src="/projects/CVc/cvc-simulation-workflow.png" alt="Simulation workflow showing training data, test data, cross-validation correction, and synthetic data evaluation" />
	<figcaption>The workflow compares standard CV, corrected CV, held-out test error, and synthetic-data error across repeated simulations.</figcaption>
</figure>

#### Results

Across 100 independent replications, standard CV is close to the held-out test-set error because the test set shares the same LD and relatedness structure as the training data. However, CV substantially underestimates error on the LD-free synthetic genomes. This shows that CV can be optimistic when the target population is less correlated than the training data.

For models with a valid correction, CVc moves the estimate closer to the synthetic-data MSE. The correction is especially large for LMM/gBLUP and BSLMM, which explicitly model correlation structure.

<figure class="article-picture">
	<img src="/projects/CVc/cvc-bias-correction-results.png" alt="Bias and correction of cross-validation error across genomic prediction methods" />
	<figcaption>CV is nearly unbiased for the same-structure test set, but optimistic for the LD-free synthetic target. CVc increases as the target correlation structure diverges from training data.</figcaption>
</figure>

<figure class="article-picture">
	<img src="/projects/CVc/cvc-results-table.png" alt="Table summarizing CV, MSE, corrected CV, and correction percentage across genomic prediction methods" />
	<figcaption>Summary table comparing CV, test-set MSE, synthetic-set MSE, corrected CV, and the percentage of synthetic-target bias addressed by the correction.</figcaption>
</figure>

#### Takeaway

Standard cross-validation can be misleading in genomic prediction when the deployment target does not share the same correlation structure as the training data. CVc makes this dependence explicit: when the target data look like the training data, the correction is close to zero; when LD and relatedness are reduced or removed, the correction increases.

The result is a more bias-aware way to evaluate genomic prediction models under realistic sample dependence, especially when the goal is to predict on genomes that are less correlated with the training cohort.
