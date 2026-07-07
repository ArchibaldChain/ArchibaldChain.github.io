---
title: "Statistical Analysis of Ookla Internet Speeds for Rural/Urban Canadian Communities"
zhTitle: "加拿大城乡社区 Ookla 网速数据统计分析"
description: "We visualized, processed, and analyzed the internet speed dataset provided by Ookla. We used logistic regression to predict future internet speed conditions and made recommendations based on the results."
zhDescription: "我们对 Ookla 提供的互联网速度数据进行了可视化、清洗和统计分析，并使用逻辑回归预测未来网速状况，基于结果提出相关建议。"
date: "May 2022"
zhDate: "2022 年 5 月"
tags:
  - Data Analysis
  - Logistic Regression
image: "/projects/Internet Speed/canada-internet-cover.png"
github: "https://github.com/HH197/Case-Study-Competition"
document: "/projects/Internet Speed/Interenet Speedposter.pdf"
backHref: "/#project"
featured: true
order: 2
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
