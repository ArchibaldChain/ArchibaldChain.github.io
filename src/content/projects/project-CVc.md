---
title: "Cross-validation Correction For Machine Learing in Genomics Datasets"
date: "Oct. 2021-Present"
tags:
  - Genomics
  - machine learning
  - cross-validation
image: "/projects/CVc/bioinformatics.jpg"
github: "https://github.com/ArchibaldChain/CVc_in_bio_informatics"
publication: "https://www.tandfonline.com/doi/full/10.1080/02664763.2026.2646570"
backHref: "/#project"
---

#### Background

In genomics dataset, the correlation between indivduals are very high, because most of the gene between indivduals are the same except few variants.
Some machine learning algorithms like **Linear Mixed Model (LMM), Sparse Bayesian Regression, Bayesian Sparse Linear Mixed Model (BSLMM)**, etc are wildely used in genomic analyzing and phynomics predictions.
Because the high correlation between individual, data leakage will happen in the process of calculating the **cross-validation (CV)** error, which causes the cross-validation underestimated the true prediction error.

#### Solution

We used a method with a covariance structure to correct the CV error.
We correct CV errors by adding a correction term.
You can find the definition and methods using this [link](/projects/CVc/Cross-Validation for Correlated Data.html).

#### Simulation and Result

We use the 1000-genomics project to simulate the cross-validation correction.
The simulation procedure can be found [here](#).
We simulated 1000 times.
The distribution of CV error corrected CV error, and the test error is shown below.
The below graph shows the mean of the corrected CV (green vertical line) is closer to the test error (black vertical line) than the CV error (orange vertical line).

<figure class="article-picture"><img src="/projects/CVc/GLS.png" alt="GLS simulation result" /></figure>
