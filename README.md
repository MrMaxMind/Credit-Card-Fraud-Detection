# Credit Card Fraud Detection

Welcome to the Credit Card Fraud Detection project repository. This project aims to detect fraudulent credit card transactions using machine learning techniques, specifically using the XGBoost classifier. Below is an overview of the project, including features, code snippets, and instructions for running the code.

---

<div align="center">
  <img src="./card.jpeg" alt="Credit Card Fraud" style="border:none;">
</div>

---

## Overview

This project focuses on identifying fraudulent credit card transactions by analyzing various features in the dataset. The dataset contains information about transactions made by credit cards in September 2013 by European cardholders. It presents transactions that occurred over two days, with 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, with the positive class (frauds) accounting for 0.172% of all transactions.

---

## Features

- **Data Exploration**: Understanding the data distribution, statistical information, and data types.
- **Data Cleaning**: Handling missing values, data type conversion, and ensuring data consistency.
- **Data Visualization**: Visualizing the distribution of fraud vs. non-fraud transactions.
- **Feature Engineering**: Separating features from the target variable for model training.
- **Model Building**: Implementing the XGBoost classifier to detect fraudulent transactions.
- **Model Evaluation**: Evaluating model performance using metrics like Confusion Matrix, Classification Report, Accuracy, and ROC-AUC score.

---

## Contents

- `credit_card_fraud_detection.ipynb`: Jupyter notebook containing the code implementation and analysis.
- `README.md`: This file, providing an overview of the project.
- `creditcard.csv`: Dataset used for training and testing the models.

---

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Credit-Card-Fraud-Detection.git
   cd Credit-Card-Fraud-Detection
2. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
3. **Run the Jupyter Notebook**:
   ```bash
   credit_card_fraud_detection.ipynb

---

## Data Exploration and Cleaning
- The dataset is read and the first few rows are displayed.
- The number of rows and columns are checked.
- Null or missing values in the dataset are verified.
- The variable types in each column are identified.
- Statistical information about the variables is displayed.
- The percentage of non-fraud and fraud transactions is calculated.
- The amount values of normal transactions versus fraud transactions are compared using visualizations.

---

## Feature Engineering
- Input variables are separated from the target variable.
- The dataset is split into training and testing sets using train_test_split.

---

## Model Building and Evaluation
- The XGBoost classifier is built and trained using the training data.
- Model performance is measured using Confusion Matrix and Classification Report.
- Key metrics include accuracy score, ROC-AUC score, and various classification metrics such as precision, recall, and F1-score

---

## Key Insights
- Identified significant factors affecting fraudulent transactions.
- Trained the XGBoost classifier to detect fraudulent transactions accurately.
- Evaluated model performance using various metrics.

---

## Tools and Libraries
- `Pandas`: For data manipulation and analysis.
- `Matplotlib`: For creating static, animated, and interactive visualizations.
- `Seaborn`: For statistical data visualization.
- `scikit-learn`: For implementing machine learning models and evaluation metrics.
- `XGBoost`: For gradient boosting algorithms.
- `Scikit-plot`: For plotting confusion matrix and other model evaluation plots.

---

## Contributing
- If you have suggestions or improvements, feel free to open an issue or create a pull request.

---

## Thank you for visiting! If you find this project useful, please consider starring the repository. Happy coding!

---
