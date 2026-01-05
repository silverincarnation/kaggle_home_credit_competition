# kaggle home credit competition
<img src="img/bronze_medal.png" alt="Bronze Medal">
<i>I (Kaixuan Chen) won a bronze medal!</i>

## Overview

This project is based on the **[Home Credit Default Risk – Stability Competition](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/overview)** on Kaggle, which focuses on predicting loan repayment capabilities for clients with limited credit history. The main challenge of the competition is not only achieving strong predictive performance (AUC), but also ensuring model stability over time. 

Submissions are evaluated using a **gini stability metric**, where weekly Gini scores are computed and a linear regression is fitted across time to penalize performance degradation and excessive variability. This design explicitly enforces a trade-off between predictive power and long-term stability, which is critical for real-world consumer finance applications.

By balancing model performance and robustness, this competition aims to help consumer finance providers expand responsible lending and improve financial inclusion for underserved populations.  
**In this competition, I (Chen Kaixuan) achieved a Bronze Medal**, ranking among the top-performing solutions.

# Data

# Model

# Evaluation

The model is evaluated using a Gini score. For each `WEEK_NUM`, the AUC is computed based on the model predictions and converted to a Gini score using the formula:  
**Gini = 2 × AUC − 1**.

