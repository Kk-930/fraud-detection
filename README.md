## Credit Card Fraud Detection using Machine Learning

This project aims to identify fraudulent credit card transactions with high accuracy, even in the presence of extreme class imbalance. By using well-established ML techniques and proper evaluation metrics, the solution helps minimize the risk of false charges for customers.


## Objective

The goal is to create a machine learning model that can flag fraudulent transactions from a dataset where only about **0.17%** of entries are frauds. The challenge lies in dealing with the highly unbalanced nature of the data.

 ## Dataset Overview
  Note: The dataset `creditcard.csv` is not included due to file size. Please download it directly from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) .
  the file is placed under data folder.

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- The dataset consists of:
  - 28 anonymized principal components (`V1` to `V28`) from a PCA transformation
  - `Amount` – the transaction amount
  - `Time` – time difference from the first transaction
  - `Class` – the label (1 = fraud, 0 = legit)

## Tools & Libraries Used

- **Language:** Python 3
- **Notebook Environment:** Jupyter
- **Libraries:**
  - Data: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - ML: `scikit-learn`, `xgboost`, `imbalanced-learn`

