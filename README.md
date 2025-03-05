### credit-risk-classification
Module 20 Challenge

## Overview of the Analysis
The purpose of this analysis is to leverage historical lending data from a peer-to-peer lending services company to build and evaluate a machine learning model that predicts loan risk. The dataset includes financial variables such as loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, derogatory marks, and total debt, with the target variable (loan_status) indicating whether a loan is healthy (0) or high-risk (1).

To achieve my goal, I followed several important stages in the machine learning process:

**Data Preparation**: I loaded the data, separated the features (X) from the target (y), and examined the imbalance in the loan_status variable.

**Data Splitting**: The dataset was divided into training and testing sets using train_test_split to ensure consistency across multiple runs.

**Model Training**: I built an initial logistic regression model using the original imbalanced data. Recognizing the imbalance (with many more healthy loans than high-risk ones), I then applied RandomOverSampler to create a balanced training set and trained a second logistic regression model.

**Model Evaluation**: Both models were evaluated using accuracy, precision, recall, and F1-score, with particular emphasis on the model’s ability to correctly identify both healthy and high-risk loans.

# Results
Machine Learning Model 1 (Original Data):

**Healthy Loans (0)**:
Precision: 100%
Recall: 99%
F1-Score: 1.00

**High-Risk Loans (1)**:
Precision: 85%
Recall: 94%
F1-Score: 0.90

**Overall Accuracy**: ~99%
(Note: Due to the imbalanced nature of the original data, the balanced accuracy score was lower at around 95%.)

Machine Learning Model 2 (Oversampled Data):

**Healthy Loans (0)**:
Precision: 100%
Recall: 99%
F1-Score: 1.00

**High-Risk Loans (1)**:
Precision: 99%
Recall: 99%
F1-Score: 0.99

Overall Accuracy & Balanced Accuracy: 99%

# Summary
Both models demonstrate strong overall accuracy; however, their performance differs when focusing on the prediction of high-risk loans. The first model, trained on the original imbalanced data, performs exceptionally well for healthy loans but shows moderate performance for high-risk loans, as shown by a precision of 85% and an F1-score of 0.90 for class 1. On the other hand, the second model, which was trained on sampled (balanced) data, improves the prediction of high-risk loans significantly, achieving precision and recall of around 99% for both classes.

Given the cost for both false positives (misclassifying a healthy loan as high-risk) and false negatives (overlooking a high-risk loan), accurate prediction of high-risk loans is critical. Therefore, based on these results, the logistic regression model built on oversampled data is recommended as the best model for predicting loan risk.
