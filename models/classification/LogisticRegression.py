from matplotlib import pyplot as plt
import seaborn as sns
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
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.caption("Model evaluation completed!")
    return accuracy, confusion, report_df

def plot_confusion_matrix(confusion):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    return fig

def plot_roc_auc(model, X_test, y_test):
    from sklearn.metrics import roc_curve, auc
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(roc_auc), color='blue')
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    return fig

def plot_feature_importance(model, feature_names):
    importance = model.coef_[0]
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
    ax.set_title('Feature Importance')
    return fig

# Main execution

def run_logistic_regression_pipeline(file_path, target_column, test_size, random_state, solver, C, penalty, class_weight):
    X, y = load_dataset(file_path, target_column)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)
    model = train_logistic_regression(X_train_scaled, y_train, solver, C, penalty, class_weight)
    accuracy, confusion, report = evaluate_model(model, X_test_scaled, y_test)
    fig = plot_confusion_matrix(confusion)
    roc_fig = plot_roc_auc(model, X_test_scaled, y_test)
    feature_importance_fig = plot_feature_importance(model, X.columns)
    return accuracy, confusion, report, fig, roc_fig, feature_importance_fig

