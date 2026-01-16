# Diabetes Prediction Model Evaluation

## Problem Statement
This project aims to develop and evaluate several machine learning models for the prediction of diabetes based on a comprehensive dataset (`diabetes_dataset00.csv`). The objective is to identify the most effective model(s) that can accurately classify different types of diabetes, contributing to early diagnosis and better management strategies.

## Dataset Description
The `diabetes_dataset00.csv` dataset contains 109,175 entries across 34 features, with 'Target' being the dependent variable representing different diabetes types. During initial data exploration, several key observations were made:

*   **Initial Data Types**: Many columns that should have been numerical (e.g., 'Insulin Levels', 'Age', 'BMI', 'Blood Pressure', 'Cholesterol Levels', etc.) were initially imported as `object` type due to presence of non-numeric characters.
*   **Data Type Conversion**: These 10 numerical-but-object columns, along with 'Birth Weight', were successfully converted to `float64` to facilitate numerical analysis.
*   **Missing Values**: Missing values were present across multiple columns. For numerical features, missing values were imputed using the median. For categorical features, missing values were imputed using the mode.
*   **Feature Encoding**: Categorical features were transformed into a numerical format using one-hot encoding, resulting in an expanded feature set of 1115 columns.
*   **Feature Scaling**: The 'Birth Weight' feature, which remained as a continuous numerical column after encoding, was scaled using `StandardScaler`.

## Model Evaluation Metrics Comparison
The following table summarizes the performance of six different classification models evaluated on the test set, including Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC). The table is sorted by F1 Score in descending order.

```
                 Model  Accuracy  AUC Score  Precision  Recall  F1 Score  MCC Score
0  Logistic Regression    0.7131     0.9700     0.7110  0.7131    0.7107     0.6894
3          Naive Bayes    0.8258     0.9896     0.8264  0.8258    0.8255     0.8113
1        Decision Tree    0.8652     0.9270     0.8656  0.8652    0.8653     0.8540
5              XGBoost    0.8983     0.9962     0.9000  0.8983    0.8972     0.8901
4        Random Forest    0.8992     0.9951     0.9039  0.8992    0.8980     0.8914
2   K-Nearest Neighbor    0.6164     0.9252     0.6205  0.6164    0.6153     0.5849
```

## Observations on Model Performance

*   **Random Forest**: This model demonstrated the highest overall performance across most metrics, achieving an F1 Score of 0.8980 and an MCC Score of 0.8914. Its high AUC score (0.9951) suggests excellent discriminative power, making it a strong candidate for deployment.

*   **XGBoost**: The XGBoost model also performed very well, particularly in AUC score (0.9962), and achieved an F1 Score of 0.8972 and an MCC Score of 0.8901. Its performance is very close to Random Forest, indicating its robustness and efficiency.

*   **Decision Tree**: As a single tree model, Decision Tree delivered good results with an F1 Score of 0.8653 and an MCC Score of 0.8540. Its AUC score (0.9270) is also high, reflecting its ability to capture complex relationships within the data.

*   **Naive Bayes**: The Naive Bayes Classifier showed strong performance with an F1 Score of 0.8255 and an MCC Score of 0.8113. It is notable for its simplicity and efficiency, performing remarkably well for a probabilistic model.

*   **Logistic Regression**: This model performed moderately, with an F1 Score of 0.7107 and an MCC Score of 0.6894. Its AUC score (0.9700) is high, suggesting good separation capabilities despite lower overall classification metrics compared to ensemble methods.

*   **K-Nearest Neighbor (KNN)**: KNN showed significantly lower performance across all metrics (F1 Score 0.6153, MCC Score 0.5849, AUC 0.9252). This indicates that a simple distance-based classifier may not be well-suited for this dataset, possibly due to the high dimensionality after one-hot encoding or the nature of the feature space.

**Conclusion**: Random Forest and XGBoost models are the top performers, demonstrating excellent predictive capabilities for this diabetes classification task. Further steps could involve hyperparameter tuning for these models to potentially eke out even better performance or considering the interpretability of these models depending on specific project requirements.
