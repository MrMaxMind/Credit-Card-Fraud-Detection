# **Credit Card Fraud Detection**

Welcome to the Credit Card Fraud Detection project repository. This project aims to detect fraudulent credit card transactions using machine learning techniques, specifically using the XGBoost classifier. Below is an overview of the project, including features, code snippets, and instructions for running the code.

---

<div align="center">
  <img src="./card.jpeg" alt="Credit Card Fraud Detection" style="border:none;">
</div>

---

## 🚀 **Overview**

This project focuses on identifying fraudulent credit card transactions by analyzing various features in the dataset. The dataset contains information about transactions made by credit cards in September 2013 by European cardholders. It presents transactions that occurred over two days, with 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, with the positive class (frauds) accounting for 0.172% of all transactions.

---

## ✨ **Features**

- **Data Exploration**: Understanding the data distribution, statistical information, and data types.
- **Data Cleaning**: Handling missing values, data type conversion, and ensuring data consistency.
- **Data Visualization**: Visualizing the distribution of fraud vs. non-fraud transactions.
- **Feature Engineering**: Separating features from the target variable for model training.
- **Model Building**: Implementing the XGBoost classifier to detect fraudulent transactions.
- **Model Evaluation**: Evaluating model performance using metrics like Confusion Matrix, Classification Report, Accuracy, and ROC-AUC score.

---

## 📂 **Contents**

- `credit_card_fraud_detection.ipynb`: Jupyter notebook containing the code implementation and analysis.
- `requirements.txt`: Python dependencies required to run the project.

---

## 🛠️  **Getting Started**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MrMaxMind/Credit-Card-Fraud-Detection.git
   cd Credit-Card-Fraud-Detection
2. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
3. **Run the Jupyter Notebook**:
   ```bash
   credit_card_fraud_detection.ipynb

---

## 🔍 **Data Exploration & Cleaning**

- **📥 Dataset Loading**: The dataset is loaded, and the first few rows are displayed to give an initial overview.
- **📊 Shape Check**: The number of rows and columns is verified to understand the dataset size.
- **🔍 Null Value Detection**: The dataset is checked for missing or null values, and appropriate handling techniques are applied.
- **🔢 Data Type Identification**: Each column's variable type is identified and adjusted if necessary.
- **📈 Statistical Summary**: Key statistical information (mean, median, standard deviation, etc.) for numerical features is displayed.
- **📊 Class Distribution**: The percentage of non-fraud and fraud transactions is calculated to detect any class imbalance.
- **💸 Transaction Amount Visualization**: Visualizations are created to compare the amount values in normal versus fraud transactions for deeper insights.

---

## 🛠️ **Feature Engineering**

- **🧮 Variable Separation**: Input features are separated from the target variable (fraud vs. non-fraud).
- **📤 Data Splitting**: The dataset is split into training and testing sets using the **train_test_split** function to evaluate model performance.

---

## 🤖 **Model Building & Evaluation**

- **🏗️ XGBoost Classifier**: The **XGBoost** classifier is built and trained using the training data to predict fraud.
- **📊 Model Evaluation**:
  - **Confusion Matrix**: To understand the classification performance and misclassifications.
  - **Classification Report**: A detailed report that includes metrics like precision, recall, and F1-score for fraud detection.
- **📈 Key Metrics**:
  - **Accuracy Score**: Overall accuracy of the model.
  - **ROC-AUC Score**: To evaluate the model's ability to distinguish between fraud and non-fraud transactions.
  - **Precision, Recall, F1-score**: These metrics provide insight into the trade-offs between false positives and false negatives.

---


## 🔍 **Key Insights**

- Identified significant factors affecting fraudulent transactions.
- Trained the XGBoost classifier to detect fraudulent transactions accurately.
- Evaluated model performance using various metrics.

---

## 🛠️ **Tools and Libraries**

- `Pandas`: For data manipulation and analysis.
- `Matplotlib`: For creating static, animated, and interactive visualizations.
- `Seaborn`: For statistical data visualization.
- `scikit-learn`: For implementing machine learning models and evaluation metrics.
- `XGBoost`: For gradient boosting algorithms.
- `Scikit-plot`: For plotting confusion matrix and other model evaluation plots.

---

## 🤝 **Contributing**

If you have suggestions or improvements, feel free to open an issue or create a pull request.

---

## ⭐ **Thank You!**

Thank you for visiting! If you find this project useful, please consider starring the repository. Happy coding!
