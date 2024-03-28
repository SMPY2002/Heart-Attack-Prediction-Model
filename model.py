import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
from xgboost import XGBClassifier
import optuna
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    precision_score,
    roc_auc_score,
    mean_squared_error,
    make_scorer,
)
from sklearn.model_selection import cross_val_score, GridSearchCV
import pickle
import json
import requests

# Load the Dataset
df = pd.read_csv("heart_attack_prediction_dataset.csv")

# Printing the dataset
# df.head()

"""Setting column 'Blood Pressure' 
Splitting Between Diastolic and Systolic Blood Pressure"""

df["BP_Systolic"] = df["Blood Pressure"].apply(lambda x: x.split("/")[0])
df["BP_Diastolic"] = df["Blood Pressure"].apply(lambda x: x.split("/")[1])

# Converting object(here category present) into integer
mapping = {"Unhealthy": -1, "Average": 0, "Healthy": 1}
df["Diet"] = df["Diet"].map(mapping)

# Converting object(here category present) into integer
mapping = {"Male": -1, "Female": 0}
df["Sex"] = df["Sex"].map(mapping)

"""Converting 'Object' and 'Boolean' Datatype into int"""
cat_columns = ["BP_Systolic", "BP_Diastolic"]
df[cat_columns] = df[cat_columns].astype(int)

X = df.drop(
    [
        "Patient ID",
        "Blood Pressure",
        "Country",
        "Continent",
        "Hemisphere",
        "Heart Attack Risk",
    ],
    axis=1,
)
y = df["Heart Attack Risk"]

sm = SMOTE(random_state=42)
X_oversampled, y_oversampled = sm.fit_resample(X, y)

# Splitting Data into train set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X_oversampled, y_oversampled, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

selector = SelectFromModel(xgb.XGBClassifier()).fit(X_train, y_train)
selected_features = selector.get_support()
X_train_selected = X_train[:, selected_features]
X_test_selected = X_test[:, selected_features]


# Creating New feature lists according to top 10 features only (provided in the last of the notebook)[For XGboost/RandomForest only]
X_new = df.drop(
    [
        "Patient ID",
        "Blood Pressure",
        "Country",
        "Continent",
        "Hemisphere",
        "Heart Attack Risk",
        "Sex",
        "Cholesterol",
        "Obesity",
        "Previous Heart Problems",
        "Medication Use",
        "Stress Level",
        "Sedentary Hours Per Day",
        "Income",
        "BMI",
        "Triglycerides",
        "Physical Activity Days Per Week",
        "Sleep Hours Per Day",
        "BP_Systolic",
        "BP_Diastolic",
    ],
    axis=1,
)
y_new = df["Heart Attack Risk"]
print(X_new.info())

# Oversample the minority class using SMOTE
sm = SMOTE(random_state=42)
X_new_oversampled, y_new_oversampled = sm.fit_resample(X_new, y_new)

# Again check the distribution of class
check_distribution = y_new_oversampled.value_counts()
print("Class-Distribution\n", check_distribution)

# Splitting Data into train set and test set
X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(
    X_new_oversampled, y_new_oversampled, test_size=0.2, random_state=42
)

# Feature Scaling to standarize the features(such that mean is zero and variance is one.)
scaler = StandardScaler()
X_new_train = scaler.fit_transform(X_new_train)
X_new_test = scaler.transform(X_new_test)


# Making the XGboost model for our heart attack risk prediction (Model-1) [Train with selected features]
def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5),
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_new_train, y_new_train)
    y_pred = model.predict(X_new_test)
    accuracy = accuracy_score(y_new_test, y_pred)
    return accuracy


# Run Optuna to find the best hyperparameters
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# Get the best hyperparameters
best_params = study.best_params

# Train the model with the best hyperparameters
model = xgb.XGBClassifier(**best_params)
model.fit(X_new_train, y_new_train)
y_pred = model.predict(X_new_test)

# Evaluate model performance
# accuracy = accuracy_score(y_test, y_pred)
# print('Accuracy with tuned hyperparameters:', accuracy)

# Perform 5-fold cross-validation for XGBOOST model
scores = cross_val_score(model, X_new_train, y_new_train, cv=5)
print("Cross-validation scores:", scores)

# Calculate the mean cross-validation score
mean_score = np.mean(scores)
print("Mean cross-validation score:", mean_score)

# Make predictions
y_pred = model.predict(X_new_test)
y_pred_binary = np.where(
    y_pred > 0.5, 1, 0
)  # Convert predicted probabilities to binary labels

# Calculate metrics
accuracy = accuracy_score(y_new_test, y_pred_binary)
precision = precision_score(y_new_test, y_pred_binary)
recall = recall_score(y_new_test, y_pred_binary)
f1 = f1_score(y_new_test, y_pred_binary, average="macro")

# Print the evaluation results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)

pickle.dump(model, open("model.pkl", "wb"))

# model = pickle.load(open('model.pkl','rb'))
# print(model.predict([[1.8]]))
