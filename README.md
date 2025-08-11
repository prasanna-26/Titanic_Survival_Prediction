# Titanic Survival Prediction - Machine Learning Project

This project aims to predict whether a passenger survived or not on the Titanic using machine learning models. The dataset contains various features about the passengers, such as their age, gender, class, and more. We will be using multiple machine learning algorithms to predict survival and evaluate their performance.

## Project Overview

The Titanic dataset is a famous dataset used in data science and machine learning competitions. This project uses various algorithms to predict survival on the Titanic. The goal is to compare the performance of different models and understand the importance of data preprocessing.

### Key Features:
- **Survived**: Whether the passenger survived (1) or not (0).
- **Pclass**: Passenger class (1, 2, 3).
- **Name**: Name of the passenger.
- **Sex**: Gender of the passenger (male/female).
- **Age**: Age of the passenger.
- **SibSp**: Number of siblings or spouses aboard.
- **Parch**: Number of parents or children aboard.
- **Fare**: Amount of money the passenger paid for the ticket.
- **Embarked**: Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton).

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
