import pandas as pd
import streamlit as st
import time
from utils.config.Configuration import set_page_config
from utils.helper import eda
from models.classification import LogisticRegression
from utils.helper import gridsearch
set_page_config()

st.title("Classification Models in Financial Risk Analysis 📈")
st.caption("Understand how basic ML classification models work using different risk datasets")

tabs = st.tabs(["1. Load Data", "2. Explore", "3. Train Model", "4. Evaluate Model", "5. Tuning", "6. Predictions", "7. Insights"])
raw_paths = {"Credit Risk (Loan Default - Binary Classification)": "data\\raw\\UCI_Credit_Card.csv",
         "Operational Risk (Fraud Detection - Binary Classification)": "data\\raw\\Fraud_Detection.csv",
        "Credit Risk (Credit Rating Classification - Multi-Class Classification)": "data\\raw\\Credit_Rating.csv"}

processed_paths = {"Credit Risk (Loan Default - Binary Classification)": "data\\processed\\UCI_Credit_Card_Cleaned.csv",
            "Operational Risk (Fraud Detection - Binary Classification)": "data\\processed\\Fraud_Detection_Cleaned.csv",
            "Credit Risk (Credit Rating Classification - Multi-Class Classification)": "data\\processed\\Credit_Rating_Cleaned.csv"}

sample_paths = {"Credit Risk (Loan Default - Binary Classification)": "data\\sample\\UCI_Credit_Card_Sample.csv",
            "Operational Risk (Fraud Detection - Binary Classification)": "data\\sample\\Fraud_Detection_Sample.csv",
            "Credit Risk (Credit Rating Classification - Multi-Class Classification)": "data\\sample\\Credit_Rating_Sample.csv"}

models = {"Credit Risk (Loan Default - Binary Classification)": ["Logistic Regression","Random Forest", "KNN"],
          "Operational Risk (Fraud Detection - Binary Classification)": ["XGBoost", "SVM"],
          "Credit Risk (Credit Rating Classification - Multi-Class Classification)": ["Random Forest", "SVM"]}

target_columns = {"Credit Risk (Loan Default - Binary Classification)": "default.payment.next.month",
                     "Operational Risk (Fraud Detection - Binary Classification)": "is_fraud",
                     "Credit Risk (Credit Rating Classification - Multi-Class Classification)": "credit_rating"}

with tabs[0]:
    st.subheader("Select Dataset")
    selected_dataset = st.selectbox("--Select from dropdown--", ["Credit Risk (Loan Default - Binary Classification)", 
                                    "Operational Risk (Fraud Detection - Binary Classification)",
                                    "Credit Risk (Credit Rating Classification - Multi-Class Classification)"])


    if selected_dataset == "Credit Risk (Loan Default - Binary Classification)":
        st.subheader("About the Dataset")
        st.markdown("**Dataset Information**")
        st.write("This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.")
        st.caption("**Source**: UCI Machine Learning Repository")

        with st.expander("**Learn more about the dataset**"):
            st.subheader("Dataset Content")
            st.write("There are 25 variables in this dataset:")
            st.markdown("""
            - **ID**: ID of each client  
            - **LIMIT_BAL**: Amount of given credit in NT dollars (includes individual and family/supplementary credit)  
            - **SEX**: Gender (1=male, 2=female)  
            - **EDUCATION**: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)  
            - **MARRIAGE**: Marital status (1=married, 2=single, 3=others)  
            - **AGE**: Age in years  
            - **PAY_0**: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above)  
            - **PAY_2**: Repayment status in August, 2005 (scale same as above)  
            - **PAY_3**: Repayment status in July, 2005 (scale same as above)  
            - **PAY_4**: Repayment status in June, 2005 (scale same as above)  
            - **PAY_5**: Repayment status in May, 2005 (scale same as above)  
            - **PAY_6**: Repayment status in April, 2005 (scale same as above)  
            - **BILL_AMT1**: Amount of bill statement in September, 2005 (NT dollar)  
            - **BILL_AMT2**: Amount of bill statement in August, 2005 (NT dollar)  
            - **BILL_AMT3**: Amount of bill statement in July, 2005 (NT dollar)  
            - **BILL_AMT4**: Amount of bill statement in June, 2005 (NT dollar)  
            - **BILL_AMT5**: Amount of bill statement in May, 2005 (NT dollar)  
            - **BILL_AMT6**: Amount of bill statement in April, 2005 (NT dollar)  
            - **PAY_AMT1**: Amount of previous payment in September, 2005 (NT dollar)  
            - **PAY_AMT2**: Amount of previous payment in August, 2005 (NT dollar)  
            - **PAY_AMT3**: Amount of previous payment in July, 2005 (NT dollar)  
            - **PAY_AMT4**: Amount of previous payment in June, 2005 (NT dollar)  
            - **PAY_AMT5**: Amount of previous payment in May, 2005 (NT dollar)  
            - **PAY_AMT6**: Amount of previous payment in April, 2005 (NT dollar)  
            - **default.payment.next.month**: Default payment (1=yes, 0=no)  
            """)

        explore_data = st.button("View Data", use_container_width=True)
        if explore_data:
            file_path = raw_paths[selected_dataset]
            raw_data, data_description = eda.explore_data(file_path)
            st.caption("Raw Data")
            st.dataframe(raw_data.head())
            st.caption("Data Description")
            st.dataframe(data_description)

            st.info("**Note**: For simplicity, the dataset is already encoded. In Machine Learning classification, " \
            "converting text data or numerical values into categories is known as 'Encoding' or 'Categorization'." \
            " e.g. Gender(1 - male, 2 - female)")


    elif selected_dataset == "Operational Risk (Fraud Detection - Binary Classification)":
        st.write("This dataset contains information about transactions and whether they are fraudulent.")
        st.write("The target variable is binary: 0 (not fraud) or 1 (fraud).")
    elif selected_dataset == "Credit Risk (Credit Rating Classification - Multi-Class Classification)":
        st.write("This dataset contains information about loan applicants and their credit ratings.")
        st.write("The target variable is multi-class: 0 (low risk), 1 (medium risk), or 2 (high risk).")

with tabs[1]:
    st.subheader("Exploratory Data Analysis")
    st.write("This is where you will explore the data.")

    with st.expander("**Why is Exploratory Data Analysis (EDA) Important in ML?**"):

        st.markdown("""
        **1. Data Quality Check**
        📌 Identifies missing values, duplicates, and inconsistencies in the dataset.

        **2. Feature Understanding**
        🔍 Helps determine which features are relevant for model building.

        **3. Outlier Detection**
        🚨 Finds extreme values that could negatively impact model performance.

        **4. Distribution Analysis**
        📊 Understands how data is spread to apply proper preprocessing techniques.

        **5. Correlation & Relationships**
        🔗 Reveals dependencies between variables to improve feature selection.

        **6. Improves Model Accuracy**
        🚀 Ensures clean, well-structured data, leading to better predictions.
        """)

        st.info("Without proper EDA, even the best ML models may fail due to poor data quality.")

    st.markdown("**1. Data Cleaning**: Fill missing values using mean, median, or mode, or drop them if necessary.")

    if st.button("Clean the selected dataset", use_container_width=True):
        file_path = raw_paths[selected_dataset]
        missing_row_df = eda.check_missing_rows(file_path)
        st.caption("Missing Rows Data")
        st.dataframe(missing_row_df.head())
        if missing_row_df.shape[0] > 0:
            st.warning(f"{missing_row_df.shape[0]} Missing values found in the dataset! Please clean the data.")

            with st.status("Cleaning... Please wait"):
                eda.remove_missing_rows(file_path, missing_row_df)
                time.sleep(3)
                st.success("Data cleaned successfully!")
        
            st.markdown("### Data Cleaning Steps")
            st.write("1. Identify missing values.")
            st.write("2. Decide how to handle them (e.g., imputation, removal). We will remove them for simplicity.")
            st.write("3. Apply the chosen method to clean the data.")
            st.write("4. Verify that the data is clean.")

        else:
            st.success("No missing values found in the dataset! Proceed with EDA")
    
    st.markdown("**2. Class Distribution Analysis**: Visualize class distribution to check if the data is balanced or imbalanced.")

    if st.button("View Class Distribution", use_container_width=True):
        file_path = processed_paths[selected_dataset]
        target_column = target_columns[selected_dataset]
        with st.status("Plotting... Please wait"):
            fig, ratio = eda.plot_class_distribution(file_path, target_column)
            time.sleep(2)
        st.pyplot(fig)

        st.markdown(f"""
        <div style="text-align: center;">
            <p><strong>Class 0 (No Default):</strong> {ratio[0]:.2f}</p>
            <p><strong>Class 1 (Default):</strong> {ratio[1]:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

        st.info("**Note**: Typically, if the ratio of the majority class to the minority class exceeds 80-20 (or is even more extreme, like 90-10), it’s considered imbalanced.")

        if ratio[0] > 0.9:
            st.warning("The dataset is imbalanced. Consider using techniques like SMOTE or ADASYN for balancing.")
        else:
            st.success("The dataset is balanced. No further action needed.")
        
        with st.expander("When is a dataset considered imbalanced?"):
            markdown_text = """
            A dataset is **imbalanced** when one class appears significantly more frequently than others.  
            Typically, if the ratio of the majority class to the minority class exceeds **80-20** (or is even more extreme, like **90-10**),  
            it’s considered imbalanced. This is a problem because models tend to favor the dominant class,  
            leading to poor generalization and weak predictive power for the minority class.

            ### **How to fix class imbalance?**
            Here are some techniques to handle imbalance:

            **1. Resampling (Oversampling & Undersampling)**  
            - **Oversampling the minority class**: Create synthetic samples of the minority class using methods like  
            **SMOTE (Synthetic Minority Over-sampling Technique)**.  
            - **Undersampling the majority class**: Remove random samples from the dominant class to balance the dataset.

            **2. Adjust Class Weights**  
            - Many machine learning models allow you to set **class weights** so that errors in predicting the minority class  
            are penalized more, forcing the model to pay attention to them.

            **3. Use Different Evaluation Metrics**  
            - Instead of accuracy (which is misleading in imbalanced datasets), use:  
            - **Precision & Recall**  
            - **F1-score**  
            - **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**  

            **4. Try Advanced Algorithms**  
            Some algorithms handle imbalance better than others:  
            - Tree-based methods like **XGBoost** allow you to set parameters to focus on rare events.  
            - **Anomaly detection methods** (useful if minority cases are rare but crucial).  
            """
            st.markdown(markdown_text)

    st.markdown("**3. Data Visualization**: Use visualizations to understand relationships between features and the target variable.")
    
    if st.button("View Data Visualization", use_container_width=True):
        file_path = processed_paths[selected_dataset]
        target_column = target_columns[selected_dataset]
        with st.status("Plotting... Please wait"):
            plot = eda.plot_feature_target_correlation(file_path, target_column)
            time.sleep(2)
        st.pyplot(plot)

        st.markdown("### Insights from Feature-Target Correlation")
        st.write("The bar chart above shows the correlation of each feature with the target variable. Here are some key insights:")
        
        st.markdown("""
        - **PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6**: These features have the highest positive correlation with the target variable, indicating that repayment status is a strong predictor of default.
        - **LIMIT_BAL**: Shows a slight negative correlation, suggesting that higher credit limits may reduce the likelihood of default.
        - **BILL_AMT and PAY_AMT features**: These have weak correlations with the target, indicating they may not be as significant for predicting default.
        - **ID**: Has almost no correlation with the target, as expected, since it is just an identifier.
        """)

        st.info("**Note**: Features with high correlation (positive or negative) are more likely to be important for the model. However, feature selection should also consider multicollinearity and domain knowledge.")

        st.success("EDA completed! You can explore further by adding techniques like feature scaling, correlation heatmaps, or PCA. Once satisfied, proceed to the Model Training section.")
with tabs[2]:
    st.subheader("Choose Model & Set Parameters")
    st.caption(f"This is where you will train the model. For our selected dataset, we will use the following models: {models[selected_dataset]}")

    model_name = st.selectbox("Select Model", models[selected_dataset])

    st.markdown("### Model Parameters")
    if model_name == "Random Forest":
        n_estimators = st.slider("Number of Trees", min_value=10, max_value=100, value=50, step=10)
        max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=5, step=1)
    elif model_name == "XGBoost":
        n_estimators = st.slider("Number of Trees", min_value=10, max_value=100, value=50, step=10)
        learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.1, step=0.01)
    elif model_name == "Logistic Regression":
        with st.expander("**First learn about the important hyperparameters**"):
            # Markdown content for parameters
            markdown_text = """
            **1. Penalty (`penalty`)**
            - Adds regularization to prevent overfitting. 
                - `l1` (Lasso): Shrinks some coefficients to zero (feature selection).
                - `l2` (Ridge): Shrinks coefficients without eliminating features.
                - `elasticnet`: Combines Lasso and Ridge.

            **2. Regularization Strength (`C`)**
            - Controls regularization strength. 
                - Lower values: Stronger regularization (less overfitting).
                - Higher values: Weaker regularization (more flexibility).

            **3. Solver (`solver`)**
            - Optimization algorithm.
                - `liblinear`: Small datasets, L1 penalty.
                - `lbfgs`, `newton-cg`: Large datasets, L2 penalty.
                - `sag`, `saga`: Very large datasets, supports both penalties.

            **4. Maximum Iterations (`max_iter`)**
            - Number of optimization iterations.
                - Increase if convergence isn’t reached.
                - Too high values may slow training.

            **5. Class Weight (`class_weight`)**
            - Adjusts weights for imbalanced classes.
                - `'balanced'`: Auto-balances weights based on class frequencies.
                - Custom weights: Improve predictions for imbalanced datasets.
            """
            # Display content using Streamlit markdown
            st.markdown(markdown_text)

        penalty = st.selectbox("**Penalty**", ["l1", "l2", "none"])
        C = st.slider("**Inverse Regularization Strength**", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
        solver = st.selectbox("**Solver**", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"])
        max_iter = st.number_input("**Max Iterations**", min_value=100, max_value=1000, value=200, step=100)
        class_weight = st.selectbox("**Class Weight**", ["balanced", None])

        test_size = st.slider("**Test Size**", min_value=0.1, max_value=0.5, value=0.2, step=0.01)
        random_state = st.number_input("**Random State**", min_value=0, max_value=100, value=42, step=1)

        if st.button("Train Model", use_container_width=True):
            with st.status("Training... Please wait", expanded=True):
                file_path = processed_paths[selected_dataset]
                target_column = target_columns[selected_dataset]
                try:
                    accuracy, confusion, report, fig, roc_fig, feature_importance_fig, trained_model, X_train_scaled, X_test_scaled, y_train, y_test, scaler \
                    = LogisticRegression.run_logistic_regression_pipeline(
                        file_path, target_column, test_size, random_state, solver, C, penalty, class_weight
                    )
                    st.success("Model trained successfully!")

                    # Store the variables in session state
                    st.session_state.accuracy = accuracy
                    st.session_state.confusion = confusion
                    st.session_state.report = report
                    st.session_state.fig = fig
                    st.session_state.roc_fig = roc_fig
                    st.session_state.feature_importance_fig = feature_importance_fig
                    st.session_state.trained_model = trained_model
                    st.session_state.X_train_scaled = X_train_scaled
                    st.session_state.X_test_scaled = X_test_scaled
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.scaler = scaler

                except Exception as e:
                    st.error(f"Error: {e}")
                    st.warning("Please check the parameters and try again.")
                
            st.info("Interpreting model performance metrics is essential for understanding how well your classification model is performing. Let’s break down key metrics in the next section.")    

with tabs[3]:
    st.subheader("Model Evaluation")
    st.write("This is where you will evaluate the trained model.")

    with st.expander("**How to Interpret Metrics**"):
        content = """
        **🔹 Accuracy**
        **Formula**: Accuracy = Correct Predictions / Total Predictions

        **Interpretation**:
        - High accuracy means mostly correct predictions.
        - Low accuracy suggests underfitting or class imbalance.
        - Accuracy alone may not be reliable for imbalanced datasets.

        **🔹 Confusion Matrix**
        **Actual vs. Predicted classifications**:

        True Negative (TN) | False Positive (FP)
        -------------------|------------------
        False Negative (FN) | True Positive (TP)

        **Interpretation**:
        - TP: Correctly predicted positives.
        - TN: Correctly predicted negatives.
        - FP: Incorrectly classified negatives (Type I Error).
        - FN: Incorrectly classified positives (Type II Error).

        **🔹 Precision**
        **Formula**: Precision = TP / (TP + FP)

        - Measures correctly predicted positives.
        - High precision means fewer false positives.
        - Important for fraud detection.

        **🔹 Recall (Sensitivity)**
        **Formula**: Recall = TP / (TP + FN)

        - Measures correctly identified actual positives.
        - High recall means fewer false negatives.
        - Critical for medical tests.

        **🔹 F1-score**
        **Formula**: F1 = 2 * (Precision * Recall) / (Precision + Recall)

        - Balances precision and recall.
        - Useful for imbalanced datasets.
        - Higher F1-score indicates better model performance.

        **🔹 Support**
        - Represents the number of occurrences of each class in the dataset.

        **🎯 Key Takeaways**:
        - If accuracy is high but precision/recall are poor, consider:
        - Rebalancing dataset.
        - Adjusting decision thresholds.
        - Using alternative metrics like AUC-ROC.
        """

        # Display content using Streamlit write
        st.write(content)
    
    st.markdown("**Model Performance Metrics**")
    if st.button("View Model Performance Metrics", use_container_width=True):
        try:
            metrics = {
                "Accuracy": st.session_state.accuracy,
                "Precision": st.session_state.report.loc["1", "precision"],
                "Recall": st.session_state.report.loc["1", "recall"],
                "F1 Score": st.session_state.report.loc["1", "f1-score"],
                "Support": st.session_state.report.loc["1", "support"]
            }
        except NameError:
            st.warning("Model not trained yet!")
            metrics = {
                "Accuracy": 0.0,
                "Precision": 0.0,
                "Recall": 0.0,
                "F1 Score": 0.0,
                "Support": 0.0
            }
        metrics_df = pd.DataFrame(metrics, index=[0])
        st.dataframe(metrics_df)

    col1, col2, col3 = st.columns(3)
    if st.button("View Model Evaluation Graphs", use_container_width=True):
        try:
            with col1:
                st.markdown("**Confusion Matrix**")
                st.pyplot(st.session_state.fig, use_container_width=True)
                st.caption("The confusion matrix shows the counts of true positives, true negatives, false positives, and false negatives for the model's predictions.")
            with col2:
                st.markdown("**ROC Curve**")
                st.pyplot(st.session_state.roc_fig, use_container_width=True)
                st.caption("The ROC curve shows the trade-off between true positive rate and false positive rate at various thresholds.")
            with col3:
                st.markdown("**Feature Importance**")
                st.pyplot(st.session_state.feature_importance_fig, use_container_width=True)
                st.caption("Feature importance indicates the contribution of each feature to the model's predictions.")
        except NameError:
            st.warning("Graphs are not available. Please ensure the model is trained successfully.")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, 'Placeholder Graph', horizontalalignment='center', verticalalignment='center', fontsize=12)
            with col1:
                st.markdown("**Confusion Matrix**")
                st.pyplot(fig, use_container_width=True)
            with col2:
                st.markdown("**ROC Curve**")
                st.pyplot(fig, use_container_width=True)
            with col3:
                st.markdown("**Feature Importance**")
                st.pyplot(fig, use_container_width=True)

        st.info("**Note**: This is not the end. We can further improve the model's performance by tuning hyperparameters, using ensemble methods, or trying different algorithms. Lets explore the parameter tuning in the next section.")

with tabs[4]:
    st.subheader("Performance Tuning")
    st.write("This is where you will tune the model's performance.")

    st.markdown("### What is Hyperparameter Tuning?")
    st.write("Hyperparameter tuning is the process of finding the optimal values for parameters that control a machine learning model's learning process but are not learned from data. These parameters affect model complexity, performance, and generalization")
    st.write("- GridSearchCV systematically tests hyperparameters to find the best combination")
    st.write("- RandomizedSearchCV randomly samples hyperparameters, making it faster for large search spaces")
    st.write("- Finds the best combination of hyperparameters using cross-validation")
    st.write("- Again assesses performance using accuracy, precision, recall, and F1-score")

    st.markdown("**We will use following parameter grid**")
    param_grid = {
        "penalty": ["l1", "l2", "elasticnet", "none"],
        "class_weight": ["None", "balanced"],
        "C": [0.01, 0.1, 1, 10],
        "solver": ["liblinear", "lbfgs", "saga", "newton-cg", "sag"],
        "max_iter": [100, 200, 300, 400, 500],
    }
    st.dataframe(pd.DataFrame.from_dict(param_grid, orient="index").transpose())

    if st.button("Lets Tune The Hyperparameters", use_container_width=True):
        with st.status("Tuning... Please wait", expanded=True):
            if "trained_model" not in st.session_state:
                st.warning("Please train the model first.")
            else:
                file_path = processed_paths[selected_dataset]
                target_column = target_columns[selected_dataset]
                try:
                    best_params, best_score = gridsearch.perform_grid_search(
                        st.session_state.trained_model,
                        param_grid,
                        st.session_state.X_train_scaled,
                        st.session_state.y_train,
                        st.session_state.X_test_scaled,
                        st.session_state.y_test,
                        )
                    st.write(f"Best Parameters:")
                    st.dataframe(best_params)
                    st.write(f"Best Accuracy Score: {best_score:.4f}")

                except Exception as e:
                    st.error(f"Error: {e}")
                    st.warning("Please check the if the model is trained and try again.")
        
        st.info("**Note**: GridSearchCV does not always guarantee the best possible score because it searches over a predefined set of hyperparameters provided by you.\n" \
        "You can also use RandomizedSearchCV, Bayesian Optimization or ested cross-validation for a more efficient search over a larger hyperparameter space.")

with tabs[5]:
    st.subheader("Predictions")
    st.write("This is where you will make predictions using the trained model.")

    st.markdown("### Make Predictions")
    st.write("You can input new data to make predictions using the trained model.")

    # Input fields for new data
    limit_bal = st.number_input("Limit Balance", min_value=0, max_value=1000000, value=50000, step=1000)

    sex = st.selectbox("Sex", ["Male", "Female"])
    sex = 1 if sex == "Male" else 2

    education = st.selectbox("Education", ["Graduate School", "University", "High School", "Others"])
    if education == "Graduate School":
        education = 1
    elif education == "University":
        education = 2
    elif education == "High School":
        education = 3
    else:
        education = 4

    marriage = st.selectbox("Marriage", ["Married", "Single", "Others"])
    if marriage == "Married":
        marriage = 1
    elif marriage == "Single":
        marriage = 2
    else:
        marriage = 3
    
    age = st.number_input("Age", min_value=0, max_value=100, value=30, step=1)

    input_data = pd.read_csv(sample_paths[selected_dataset])
    input_data.at[0, 'LIMIT_BAL'] = int(limit_bal)
    input_data.at[0, 'SEX'] = int(sex)
    input_data.at[0, 'EDUCATION'] = int(education)
    input_data.at[0, 'MARRIAGE'] = int(marriage)
    input_data.at[0, 'AGE'] = int(age)
    X = input_data.drop(target_columns[selected_dataset], axis=1)

    if st.button("Make Prediction", use_container_width=True):
        with st.status("Making prediction... Please wait", expanded=True):
            try:
                # Ensure input data is preprocessed using the same scaler as training data
                X_scaled = st.session_state.scaler.transform(X)
                prediction = st.session_state.trained_model.predict(X_scaled)
                prediction_proba = st.session_state.trained_model.predict_proba(X_scaled)
                st.success(f"Prediction: {prediction[0]}. The client will {'default' if prediction[0] == 1 else 'not default'}")
                st.success(f"Prediction Probability: {prediction_proba[0]}")
            except Exception as e:
                st.error(f"Error: {e}")
                st.warning("Please check if the model is trained and try again.")
        st.info("**Note**: The prediction is based on the trained model and the input data. Ensure that the input data is preprocessed in the same way as the training data.")

with tabs[6]:
    st.subheader("Model Insights")
    st.write("This is where you will provide insights about the model.")

    with st.expander("**Model Insights**"):
        st.markdown("""
        - **Feature Importance**: The most important features for predicting loan default are PAY_0, PAY_2, and PAY_3.
        - **Model Performance**: The model achieved an accuracy of 0.80, indicating it correctly predicted 80% of the cases.
        - **Class Imbalance**: The dataset was imbalanced, but techniques like SMOTE can help balance it.
        - **ROC Curve**: The ROC curve shows a good trade-off between true positive and false positive rates.
        """)

