# kaggle home credit competition
<img src="img/bronze_medal.png" alt="Bronze Medal">
<i>I (Kaixuan Chen) won a bronze medal!</i>

## Overview

This project is based on the **[Home Credit Default Risk – Stability Competition](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/overview)** on Kaggle, which focuses on predicting loan repayment capabilities for clients with limited credit history. The main challenge of the competition is not only achieving strong predictive performance (AUC), but also ensuring model stability over time. 

Submissions are evaluated using a **gini stability metric**, where weekly Gini scores are computed and a linear regression is fitted across time to penalize performance degradation and excessive variability. This design explicitly enforces a trade-off between predictive power and long-term stability, which is critical for real-world consumer finance applications.

By balancing model performance and robustness, this competition aims to help consumer finance providers expand responsible lending and improve financial inclusion for underserved populations.  
**In this competition, I (Chen Kaixuan) achieved a Bronze Medal**, ranking among the top-performing solutions.

# Data

For detailed information about the dataset, please refer to  
[Home Credit – Credit Risk Model Stability (Kaggle)](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/data)

Our data processing pipeline has two key highlights:

### 1. Memory Optimization
To reduce memory usage, we downcast data types to the smallest possible formats without losing precision.  
For example, numerical columns are converted from `int64` to `int32` wherever applicable.  
This strategy lowers the overall memory footprint of the dataset.

### 2. Aggregation for Historical Records (depth > 0)
For features with `depth > 0`, historical records associated with each `case_id` need to be condensed into a single representation.  
We apply aggregation functions—**maximum (max)**, **mean**, and **variance (var)**—to summarize these historical features effectively.

### Results
Through the above optimizations and aggregations:

- **Memory usage of dataframe `df_train`** was reduced from **4711.2195 MB** to **2665.6302 MB**
- **Final training data shape**: (1,526,659, 472)

# Model

# Evaluation

The model is evaluated using a Gini score. For each `WEEK_NUM`, the AUC is computed based on the model predictions and converted to a Gini score using the formula:  **Gini = 2 × AUC − 1**.

