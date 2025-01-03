# -*- coding: utf-8 -*-
"""DrugClassi.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Hhb-tYQ_vKxkpK_YxJVh88x7j6c2mXNi
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("drug200.csv")

# Encode categorical features
le_sex = LabelEncoder()
data['Sex'] = le_sex.fit_transform(data['Sex'])  # M -> 1, F -> 0

le_bp = LabelEncoder()
data['BP'] = le_bp.fit_transform(data['BP'])  # LOW -> 1, NORMAL -> 2, HIGH -> 0

le_cholesterol = LabelEncoder()
data['Cholesterol'] = le_cholesterol.fit_transform(data['Cholesterol'])  # NORMAL -> 1, HIGH -> 0

le_drug = LabelEncoder()
data['Drug'] = le_drug.fit_transform(data['Drug'])

# Define features and target
X = data.drop('Drug', axis=1)
y = data['Drug']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train[['Age', 'Na_to_K']] = scaler.fit_transform(X_train[['Age', 'Na_to_K']])
X_test[['Age', 'Na_to_K']] = scaler.transform(X_test[['Age', 'Na_to_K']])

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=le_drug.classes_, yticklabels=le_drug.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save model and encoders for deployment
import pickle
with open("drug_classifier_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("label_encoders.pkl", "wb") as encoders_file:
    pickle.dump({"Sex": le_sex, "BP": le_bp, "Cholesterol": le_cholesterol, "Drug": le_drug}, encoders_file)

# Load and display saved model and encoders for visibility
print("\nLoading saved model and encoders...")
with open("drug_classifier_model.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)
print("Loaded Model:\n", loaded_model)

with open("label_encoders.pkl", "rb") as encoders_file:
    loaded_encoders = pickle.load(encoders_file)
print("Loaded Encoders:\n", loaded_encoders)