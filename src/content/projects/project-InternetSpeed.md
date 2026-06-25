---
title: "Statistical Analysis of Ookla Internet Speeds for Rural/Urban Canadian Communities"
date: "May 2022"
tags:
  - Data Analysis
  - Logistic Regression
image: "/projects/Internet Speed/internet.jpg"
github: "https://github.com/HH197/Case-Study-Competition"
document: "/projects/Internet Speed/Interenet Speedposter.pdf"
backHref: "/#project"
---

#### Background

The Government of Canada has committed to helping 95% of Canadian households and businesses access high-speed internet at minimum speeds of 50 Mbps download and 10 Mbps upload by 2026, and 100% by 2030.
According to the CRTC, currently 45.6% of rural community households have access to the Commitment.

#### Methods

<figure class="article-picture"><img src="/projects/Internet Speed/avg download speed.svg" alt="Average download speed" /></figure>

We visualized average internet speed for each community shown below.
Next we splited the dataset into training set and test set.
And we fitted logistic regression to predict if a community can reach the commitment in the future.
And our model has 80% accuracy for the prediction.

<figure class="article-picture"><img src="/projects/Internet Speed/prediction.svg" alt="Internet speed prediction" /></figure>

#### Results and Conclusion

Our analyses show steady development in internet speed in most areas of Canada for fixed and mobile connection types.
However, underserved communities have large disparities in terms of internet access compared to rural and urban areas for both fixed and mobile connection types.
Specifically, mobile connection with current trends would not make any significant progress toward commitment.

<figure class="article-picture"><img src="/projects/Internet Speed/map.png" alt="Internet speed map" /></figure>
