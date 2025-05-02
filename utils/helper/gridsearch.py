import time
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def perform_grid_search(model, param_grid, X_train, y_train, X_test, y_test, cv=5, scoring="accuracy", verbose=1, n_jobs=-1):
    """
    Perform grid search to find the best hyperparameters for a given model.

    Parameters:
        model: The machine learning model to tune.
        param_grid: Dictionary with parameters names as keys and lists of parameter settings to try as values.
        X_train: Training feature set.
        y_train: Training target set.
        X_test: Testing feature set.
        y_test: Testing target set.
        cv: Number of cross-validation folds (default is 5).
        scoring: Scoring metric to evaluate the model (default is "accuracy").
        verbose: Verbosity level (default is 1).
        n_jobs: Number of jobs to run in parallel (default is -1 for all processors).

    Returns:
        best_params: Best hyperparameters found during grid search.
        accuracy: Accuracy of the model with the best hyperparameters on the test set.
    """
    st.caption("Reading the parameter grid...")
    time.sleep(2)  # Simulate loading time
    st.caption("Performing grid search...")
    time.sleep(2)  # Simulate processing time
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, verbose=verbose, n_jobs=n_jobs)
    st.caption("Fitting the model...This may take a while...")
    time.sleep(2)  # Simulate processing time
    grid_search.fit(X_train, y_train)

    st.caption("Grid search completed!")
    time.sleep(2)  # Simulate processing time
    st.caption("Best parameters found:")
    time.sleep(2)  # Simulate processing time
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    st.caption("Evaluating the best model...")
    time.sleep(2)  # Simulate processing time
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.caption("Grid search and evaluation completed!")
    
    return best_params, accuracy