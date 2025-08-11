# Titanic Survival Prediction - Machine Learning Project

## Table of Contents
1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Project Steps](#project-steps)
4. [Data Preprocessing](#data-preprocessing)
5. [Machine Learning Models Used](#machine-learning-models-used)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Results](#results)
8. [Acknowledgements](#acknowledgements)
9. [Understanding the Models Used in Titanic Survival Prediction](#understanding-the-models-used-in-titanic-survival-prediction)
    1. [Logistic Regression](#logistic-regression)
    2. [Random Forest](#random-forest)
    3. [Support Vector Classifier (SVC)](#support-vector-classifier-svc)
10. [Model Comparison and Applicability](#model-comparison-and-applicability)
11. [Result Interpretation and Final Output](#result-interpretation-and-final-output)
12. [Conclusion](#conclusion)

---

## Project Overview

The Titanic dataset is a famous dataset used in data science and machine learning competitions. This project uses various algorithms to predict survival on the Titanic. The goal is to compare the performance of different models and understand the importance of data preprocessing.

## Technologies Used

- Python 3
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

## Project Steps

1. **Data Loading**: The Titanic dataset is loaded from a CSV file.
2. **Data Preprocessing**: Missing values are handled, categorical variables are encoded, and continuous variables are scaled.
3. **Model Training**: We train multiple models, including Logistic Regression, Random Forest, and Support Vector Classifier (SVC).
4. **Model Evaluation**: We evaluate the models using metrics such as accuracy, precision, recall, and confusion matrix.
5. **Model Comparison**: The models are compared to determine the best-performing one.

## Data Preprocessing

- **Missing Value Imputation**: The missing `Age` values are filled with the median age, and `Embarked` missing values are filled with the most frequent value (mode).
- **Feature Encoding**: The `Sex` feature is converted into binary values (0 for male, 1 for female). The `Embarked` feature is one-hot encoded.
- **Feature Scaling**: Continuous features like `Age` and `Fare` are scaled using the StandardScaler.

## Machine Learning Models Used

- **Logistic Regression**: A simple linear model for binary classification.
- **Random Forest**: An ensemble model that uses multiple decision trees.
- **Support Vector Classifier (SVC)**: A powerful classifier that separates classes using hyperplanes.

## Evaluation Metrics

- **Accuracy**: The ratio of correctly predicted observations to the total observations.
- **Precision, Recall, F1-Score**: Evaluates the model's ability to predict true positives.
- **Confusion Matrix**: A matrix to visualize classification performance.

## Results

- **Logistic Regression**: Accuracy = 80%
- **Random Forest**: Accuracy = 83%
- **SVC**: Accuracy = 78%

The **Random Forest** model performed the best with the highest accuracy.


---

# Understanding the Models Used in Titanic Survival Prediction

In this project, three different machine learning models were applied to predict the survival of passengers on the Titanic. Each model has unique characteristics and mechanisms of operation. Below, we'll explain how **Logistic Regression**, **Random Forest**, and **Support Vector Classifier (SVC)** work, along with their advantages and disadvantages in solving this classification problem.

## Logistic Regression

### Concept:
Logistic Regression is a linear model used for binary classification tasks, meaning it predicts one of two possible outcomes. In this case, it predicts whether a passenger survived or not on the Titanic.

- **Linear Model**: It models the relationship between the independent variables (features like Age, Pclass, Sex) and the target variable (Survived) as a linear function.
- **Sigmoid Function**: Logistic Regression uses a Sigmoid function (or logistic function) to transform the linear combination of features into a probability value between 0 and 1. This probability represents the likelihood that the passenger survived.

### Mathematical Equation:
The logistic function is:

$$
P(\text{Survived} = 1) = \frac{1}{1 + e^{-(w_1 x_1 + w_2 x_2 + \dots + w_n x_n)}}
$$

Where:
- \( w_1, w_2, \dots, w_n \) are the coefficients of the model (weights),
- \( x_1, x_2, \dots, x_n \) are the features (independent variables),
- The output is the probability of survival.

### How it works:
1. The algorithm calculates the weighted sum of the input features.
2. It applies the sigmoid function to obtain a probability value.
3. The model predicts a passenger's survival if the probability is above 0.5, and no survival if it is below 0.5.

### Advantages:
- Simple and easy to interpret.
- Fast to train and suitable for linearly separable data.

### Disadvantages:
- Struggles with complex, non-linear data.
- Assumes a linear relationship between features and target.

## Random Forest

### Concept:
Random Forest is an ensemble method that builds multiple decision trees and combines their predictions. It is a powerful model that works well on both regression and classification tasks.

- **Ensemble Method**: Random Forest creates a collection (or "forest") of decision trees. Each tree is trained on a random subset of the data with bootstrapping (sampling with replacement).
- **Voting Mechanism**: Once the trees are built, the Random Forest algorithm uses the majority vote (for classification) from all the individual trees to make the final prediction.

### How it works:
1. **Random Sampling**: The data is randomly sampled, and multiple decision trees are trained on these samples.
2. **Decision Trees**: Each decision tree makes a prediction based on feature splitting (similar to a flowchart).
3. **Voting**: In the case of classification, the model takes the majority vote of all the trees to make the final decision.
4. **Out-of-Bag Error**: Random Forest has a feature where it can evaluate its performance on data that was not used for training (out-of-bag data), which helps prevent overfitting.

### Advantages:
- Handles non-linear data well.
- Robust to overfitting due to the averaging effect.
- Works well with high-dimensional datasets and datasets with many features.

### Disadvantages:
- Can be computationally expensive.
- Less interpretable than simpler models like Logistic Regression.

## Support Vector Classifier (SVC)

### Concept:
Support Vector Classifier (SVC) is a supervised machine learning model that finds the optimal hyperplane which separates the data into two classes. It is a powerful model for classification, especially for non-linear data.

- **Hyperplane**: A hyperplane is a decision boundary that separates different classes. In a 2D space, this is a line, but in higher dimensions, it's a plane or hyperplane.
- **Support Vectors**: These are the data points that are closest to the hyperplane. They are the critical points that define the position of the hyperplane.

### How it works:
1. **Finding the Optimal Hyperplane**: SVC aims to find a hyperplane that maximizes the margin between the two classes (survived and not survived).
2. **Support Vectors**: The hyperplane is determined by the "support vectors," which are the data points closest to the hyperplane.
3. **Kernel Trick**: If the data is not linearly separable, SVC uses a kernel function (like the radial basis function or RBF kernel) to project the data into higher dimensions where it becomes linearly separable.

### Advantages:
- Works well for both linear and non-linear data.
- Efficient in high-dimensional spaces.
- Robust to overfitting, especially in high-dimensional spaces.

### Disadvantages:
- Can be computationally expensive, especially with large datasets.
- Requires careful selection of the kernel and regularization parameters.

---

## Model Comparison and Applicability

| **Model**                | **Key Strengths**                                | **Best For**                         |
|--------------------------|--------------------------------------------------|--------------------------------------|
| **Logistic Regression**   | Simple, fast, easy to interpret, efficient for linear relationships | Linearly separable problems         |
| **Random Forest**         | Robust to overfitting, handles complex relationships, works well on large datasets | Non-linear problems, large datasets |
| **SVC**                   | Effective for high-dimensional data, works well for both linear and non-linear problems | Complex, non-linear problems        |

---

## Result Interpretation and Final Output

### 1. Logistic Regression Result Interpretation
After training the Logistic Regression model on the Titanic dataset, the accuracy achieved on the test set was approximately **80%**.

- **Interpretation**: The Logistic Regression model uses a linear decision boundary, which might not fully capture the complex relationships in the data. However, an accuracy of 80% indicates that the model is able to correctly predict survival for the majority of passengers.

### 2. Random Forest Result Interpretation
The **Random Forest** model performed the best, with an accuracy of approximately **83%** on the test set.

- **Interpretation**: Random Forest captures the non-linear relationships in the data effectively. It works well by averaging predictions from multiple decision trees, which reduces overfitting and increases accuracy.

### 3. Support Vector Classifier (SVC) Result Interpretation
The **SVC** model achieved an accuracy of **78%** on the test set.

- **Interpretation**: While SVC performs well on complex problems, it did not outperform Random Forest in this case. The non-linear nature of the data might have hindered SVC’s ability to model the relationships effectively.

---

## Final Output / Model Performance Comparison

After evaluating all three models, the **Random Forest model** emerged as the best-performing model for predicting Titanic survival, achieving an accuracy of **83%**. Below is a summary of the final model performances:

| **Model**                | **Accuracy** | **Precision (Survived)** | **Recall (Survived)** | **F1-Score (Survived)** |
|--------------------------|--------------|--------------------------|-----------------------|-------------------------|
| **Logistic Regression**   | 80%          | 0.78                     | 0.70                  | 0.74                    |
| **Random Forest**         | 83%          | 0.82                     | 0.78                  | 0.80                    |
| **Support Vector Classifier (SVC)** | 78%  | 0.75                     | 0.65                  | 0.70                    |

---

## Conclusion
The Titanic survival prediction project demonstrated how different machine learning algorithms perform on the same dataset. While Logistic Regression offers simplicity and speed, **Random Forest** is more robust and accurate for this specific problem, achieving the highest accuracy of **83%**. This project not only helped in understanding model selection but also showcased how hyperparameters and algorithm types affect performance. The **Random Forest model** would be the recommended model for predicting survival on the Titanic given the data at hand.

---
## Acknowledgements

- The Titanic dataset is publicly available on Kaggle (https://www.kaggle.com/c/titanic/data).
- Special thanks to the Kaggle community for providing valuable insights and tutorials.
