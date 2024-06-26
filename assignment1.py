# -*- coding: utf-8 -*-
"""Assignment1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1wfZt34PuWDaaysx7a1Nx4BppoQjafC4o
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImPipeline
from scipy import stats
import matplotlib.pyplot as plt

# Load data
train_data = pd.read_csv("/content/train.csv")

# Extract target
target = train_data['target']
train_data.drop(['target'], axis=1, inplace=True)

# Define columns with missing values
numerical_columns_with_na = [col for col in ['Dropped_calls_ratio', 'call_cost_per_min'] if col in train_data.columns]
categorical_columns_with_na = [col for col in ['Usage_Band'] if col in train_data.columns]

# Handle missing values using KNN imputation for numerical columns and mode imputation for categorical columns
if numerical_columns_with_na:
    knn_imputer = KNNImputer(n_neighbors=5)
    train_data[numerical_columns_with_na] = knn_imputer.fit_transform(train_data[numerical_columns_with_na])
if categorical_columns_with_na:
    mode_imputer = SimpleImputer(strategy='most_frequent')
    train_data[categorical_columns_with_na] = mode_imputer.fit_transform(train_data[categorical_columns_with_na])

# Define the function to handle outliers and skewness
def handle_outliers_and_skewness(data):
    if 'Age' in data.columns:
        data['Age'] = np.clip(data['Age'], data['Age'].quantile(0.01), data['Age'].quantile(0.99))
    if 'Total_Cost' in data.columns:
        pt = PowerTransformer()
        data['Total_Cost'] = pt.fit_transform(data[['Total_Cost']])
    if 'Dropped_Calls' in data.columns:
        data['Dropped_Calls'] = np.log1p(data['Dropped_Calls'] + 1e-6)

    skewed_features = data.select_dtypes(include=['int64', 'float64']).columns
    skewness = data[skewed_features].apply(lambda x: x.skew()).sort_values(ascending=False)
    high_skew = skewness[abs(skewness) > 0.75]
    skewed_features = high_skew.index

    for feature in skewed_features:
        if feature in data.columns:
            pt = PowerTransformer()
            data[feature] = pt.fit_transform(data[[feature]])

    return data

train_data = handle_outliers_and_skewness(train_data)

# Identify and correct categorical features
categorical_columns = ['Gender', 'tariff', 'Handset', 'Usage_Band', 'Tariff_OK', 'high Dropped calls', 'No Usage']
categorical_features = [col for col in categorical_columns if col in train_data.columns]

# Apply one-hot encoding
train_data_encoded = pd.get_dummies(train_data, columns=categorical_features, drop_first=True)

# Save the columns of the training data for later use
train_columns = train_data_encoded.columns
joblib.dump(train_columns, 'train_columns.pkl')

# Define interaction terms
interaction_terms = [
    ('Total_Cost', 'L_O_S'), ('Total_Cost', 'Dropped_Calls'), ('L_O_S', 'Dropped_Calls'), ('Usage_Band', 'Dropped_Calls'),
    ('Total_Cost', 'Usage_Band'), ('AvePeak', 'Total_Cost'), ('Peak_calls_Sum', 'OffPeak_calls_Sum'),
    ('International_mins_Sum', 'Total_Cost'), ('Dropped_Calls', 'call_cost_per_min'), ('National_calls', 'International_mins_Sum')
]

for feature_a, feature_b in interaction_terms:
    interaction_column = f'{feature_a}_x_{feature_b}'
    if feature_a in train_data.columns and feature_b in train_data.columns:
        if not pd.api.types.is_numeric_dtype(train_data[feature_a]):
            train_data[feature_a] = pd.to_numeric(train_data[feature_a], errors='coerce')
        if not pd.api.types.is_numeric_dtype(train_data[feature_b]):
            train_data[feature_b] = pd.to_numeric(train_data[feature_b], errors='coerce')

        median_a = train_data[feature_a].median()
        median_b = train_data[feature_b].median()
        train_data[feature_a].fillna(median_a, inplace=True)
        train_data[feature_b].fillna(median_b, inplace=True)

        train_data[interaction_column] = train_data[feature_a] * train_data[feature_b]

# Convert 'Connect_Date' to datetime and extract components
if 'Connect_Date' in train_data.columns:
    train_data['Connect_Date'] = pd.to_datetime(train_data['Connect_Date'])
    train_data['Connect_Year'] = train_data['Connect_Date'].dt.year
    train_data['Connect_Month'] = train_data['Connect_Date'].dt.month
    train_data['Connect_Day'] = train_data['Connect_Date'].dt.day
    train_data.drop(columns=['Connect_Date'], inplace=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(train_data_encoded, target, test_size=0.2, random_state=42)

# Assuming you might want to scale numeric features (if not already scaled)
scaler = StandardScaler()
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
X_train_scaled = scaler.fit_transform(X_train[numeric_features])
X_test_scaled = scaler.transform(X_test[numeric_features])

X_train_scaled = pd.DataFrame(X_train_scaled, columns=numeric_features, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=numeric_features, index=X_test.index)

# Define transformers based on the training data
numeric_features = [column_name for column_name in X_train.columns if X_train[column_name].dtype in ['int64', 'float64']]
categorical_features = [column_name for column_name in X_train.columns if X_train[column_name].dtype == 'object']

transformers = [
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
]

preprocessor = ColumnTransformer(transformers=transformers)

# Setup models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

# Evaluate multiple models
model_performance = {}
for name, model in models.items():
    print(name)
    model_pipeline = ImPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])

    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    roc_auc = roc_auc_score(y_test, model_pipeline.predict_proba(X_test)[:, 1])
    print(f"Model: {name}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc}")

    cv_scores = cross_val_score(model_pipeline, train_data_encoded, target, cv=5, scoring='roc_auc')
    avg_cv_score = np.mean(cv_scores)
    print("Average 5-Fold ROC AUC Score:", avg_cv_score)
    print("-" * 80)

    model_performance[name] = (roc_auc, avg_cv_score, model_pipeline)

best_model_name = max(model_performance, key=lambda k: model_performance[k][0])
best_model_pipeline = model_performance[best_model_name][2]
print(f"Best Model: {best_model_name}")

joblib.dump(best_model_pipeline, 'best_model_pipeline.pkl')

# Load test data
test_data = pd.read_csv('/content/test.csv')

# Handle missing values
def handle_missing_values(data):
    numerical_columns_with_na = [col for col in data.columns if data[col].dtype in ['int64', 'float64'] and data[col].isnull().any()]
    categorical_columns_with_na = [col for col in data.columns if data[col].dtype == 'object' and data[col].isnull().any()]

    if numerical_columns_with_na:
        knn_imputer = KNNImputer(n_neighbors=5)
        data[numerical_columns_with_na] = knn_imputer.fit_transform(data[numerical_columns_with_na])
    if categorical_columns_with_na:
        mode_imputer = SimpleImputer(strategy='most_frequent')
        data[categorical_columns_with_na] = mode_imputer.fit_transform(data[categorical_columns_with_na])

    return data

test_data = handle_missing_values(test_data)
test_data = handle_outliers_and_skewness(test_data)

# Create interaction terms for test data
for feature_a, feature_b in interaction_terms:
    interaction_column = f'{feature_a}_x_{feature_b}'
    if feature_a in test_data.columns and feature_b in test_data.columns:
        if not pd.api.types.is_numeric_dtype(test_data[feature_a]):
            test_data[feature_a] = pd.to_numeric(test_data[feature_a], errors='coerce')
        if not pd.api.types.is_numeric_dtype(test_data[feature_b]):
            test_data[feature_b] = pd.to_numeric(test_data[feature_b], errors='coerce')

        median_a = test_data[feature_a].median()
        median_b = test_data[feature_b].median()
        test_data[feature_a].fillna(median_a, inplace=True)
        test_data[feature_b].fillna(median_b, inplace=True)

        test_data[interaction_column] = test_data[feature_a] * test_data[feature_b]

# Convert 'Connect_Date' to datetime and extract components
if 'Connect_Date' in test_data.columns:
    test_data['Connect_Date'] = pd.to_datetime(test_data['Connect_Date'])
    test_data['Connect_Year'] = test_data['Connect_Date'].dt.year
    test_data['Connect_Month'] = test_data['Connect_Date'].dt.month
    test_data['Connect_Day'] = test_data['Connect_Date'].dt.day
    test_data.drop(columns=['Connect_Date'], inplace=True)

# Identify categorical features in the test data
categorical_features = ['Gender', 'tariff', 'Handset', 'Usage_Band', 'Tariff_OK', 'high Dropped calls', 'No Usage']

# Perform one-hot encoding on the test data
test_data_encoded = pd.get_dummies(test_data, columns=categorical_features, drop_first=True)

# Align the test data with the training data to ensure consistency in the features
train_columns = joblib.load('train_columns.pkl')
test_data_encoded = test_data_encoded.reindex(columns=train_columns, fill_value=0)

# Verify preprocessing steps
def verify_preprocessing(data):
    print("Missing values after preprocessing:", data.isnull().sum().sum())
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_columns:
        skewness = data[col].skew()
        kurtosis = data[col].kurt()
        print(f"{col}: Skewness = {skewness:.2f}, Kurtosis = {kurtosis:.2f}")
    encoded_categorical_features = [col for col in data.columns if '_' in col and col.split('_')[0] in categorical_features]
    print(data[encoded_categorical_features].head())

verify_preprocessing(test_data_encoded)

# Generate predictions on the test set
test_predictions = best_model_pipeline.predict_proba(test_data_encoded)[:, 1]

# Create a DataFrame for the results
results_df = pd.DataFrame({'ID': test_data['ID'], 'PRED': test_predictions})

# Save the results to a CSV file
results_df.to_csv('test_predictions.csv', index=False, header=False)
results_df.to_csv('submission.csv', index=False, header=False)

print(results_df.head())
print("Prediction value counts:")
print(results_df['PRED'].value_counts())