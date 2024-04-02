# Fraud Detection using Machine Learning

This project aims to develop a machine learning model for detecting fraudulent transactions based on a given dataset. By leveraging  techniques in data exploration, feature engineering, and model selection, we strive to create a robust fraud detection system that can help businesses mitigate financial risks and protect their customers.

## Dataset

The dataset used in this project consists of a large number of transactional records, each characterized by various attributes such as: (too large to upload but can be given if wanted!)

- Transaction amount
- Merchant details
- Customer information
- Time-related factors
- ...

The dataset includes a binary target variable indicating whether a transaction is fraudulent or not. It is important to note that the dataset is highly imbalanced, with fraudulent transactions constituting only a small percentage of the total records.

## Project Structure

The project is organized into the following main components:


**File: Data Exploration - For interesting Features Pt.1**

1. **Data Exploration Pt1**: In this phase, we perform an initial examination of the dataset, identify the class imbalance, analyze transaction dates and times, and count transactions per person/credit card. We also discuss potential challenges in building models, such as scalability, data sparsity, and generalization.

2. **Feature Engineering**: We create various features to capture relevant information from the dataset. This includes time-based features (hour of day, day of week), analysis of transaction     categories and amounts, creation of binary features for high-risk categories and transaction amount outliers, calculation of distances between customer and merchant locations, and combination    of distance and transaction amount features to identify atypical patterns.

**File: Fitting the features to Models Pt.1**

3. **Initial Modeling**: We select and justify initial features, scale numeric features, and one-hot encode categorical features. We train and evaluate a KNN model, apply PCA to handle multicollinearity, and test Decision Tree, Naive Bayes (after PCA), and a Stacking Ensemble with Gradient Boosting. We discuss model performance and challenges, such as class imbalance, and identify the need for more diverse features.

**File: Data Exploration - For interesting Features Pt.2**

4. **Advanced Feature Engineering**: We identify "at-risk" states and cities based on fraud rates, set minimum population thresholds for city-level analysis, analyze fraud rates by job title and merchant, and examine the composition of fraudulent transactions by category and hour of the day. We create risk score features for states, cities, jobs, and merchants based on adjusted fraud rates. We also convert the date of birth to an age feature, consolidate the day of week into a binary weekend/weekday feature, and one-hot encode the gender feature.

**File: Fitting the features to Models Pt.2**

5. **Final Modeling**: We apply feature engineering to both training and testing datasets, train and evaluate a Decision Tree Classifier (optimizing max depth using F1 score), Logistic Regression (with feature scaling), and Gradient Boosting (XGBoost) models. We compare model performance and use the best model (Gradient Boosting) for final predictions on the test dataset.

**File: Kaggle Fraudulent Transactions Competition PDF**

6. **Results and Discussion**: We summarize the best performing model and its performance metrics, discuss the effectiveness of the engineered features, and identify potential areas for further improvement.

## Installation and Usage

1. Run the Jupyter notebooks in the specified order to reproduce the analysis and model development process.
