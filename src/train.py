# src/train.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

def main():
    df = pd.read_csv("data/diabetes.csv")
    cols_with_zero = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    for col in cols_with_zero:
        df[col] = df[col].replace(0, np.nan)
        df[col].fillna(df[col].median(), inplace=True)

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {'n_estimators':[50,100], 'max_depth':[None,5,10]}
    gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    gs.fit(X_train_scaled, y_train)
    best = gs.best_estimator_
    preds = best.predict(X_test_scaled)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("ROC AUC:", roc_auc_score(y_test, best.predict_proba(X_test_scaled)[:,1]))

    os.makedirs("model", exist_ok=True)
    joblib.dump({'model': best, 'scaler': scaler}, "model/diabetes_model.pkl")
    print("Saved model to model/diabetes_model.pkl")

if __name__ == "__main__":
    main()
