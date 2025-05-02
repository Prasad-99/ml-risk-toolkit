import pandas as pd
import streamlit as st
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_dataset(file_path, target_column):
    st.caption("Loading dataset...")
    time.sleep(2)  # Simulate loading time
    df = pd.read_csv(file_path)
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    st.caption("Dataset loaded successfully!")
    return X, y

def split_data(X, y, test_size, random_state):
    st.caption("Splitting data into training and testing sets...")
    time.sleep(2)  # Simulate processing time
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def standardize_features(X_train, X_test):
    st.caption("Standardizing features...")
    time.sleep(2)  # Simulate processing time
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_logistic_regression(X_train, y_train, solver, C, penalty, class_weight, max_iter=1000, random_state=42):
    st.caption("Training Logistic Regression model...")
    time.sleep(2)  # Simulate processing time
    model = LogisticRegression(max_iter=max_iter, random_state=random_state, solver=solver, C=C, penalty=penalty, class_weight=class_weight)
    model.fit(X_train, y_train)
    st.caption("Model trained successfully!")
    return model

def evaluate_model(model, X_test, y_test):
    st.caption("Evaluating model...")
    time.sleep(2)  # Simulate processing time
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    st.caption("Model evaluation completed!")
    return accuracy, confusion, report

# Main execution

def run_logistic_regression_pipeline(file_path, target_column, test_size, random_state, solver, C, penalty, class_weight):
    X, y = load_dataset(file_path, target_column)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)
    model = train_logistic_regression(X_train_scaled, y_train, solver, C, penalty, class_weight)
    accuracy, confusion, report = evaluate_model(model, X_test_scaled, y_test)
    return accuracy, confusion, report

