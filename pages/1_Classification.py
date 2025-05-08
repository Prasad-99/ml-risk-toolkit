import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Digraph
from sklearn.calibration import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import streamlit as st
import time
from utils.config.Configuration import set_page_config
from utils.helper import eda
from models.classification import LogisticRegression
from utils.helper import gridsearch
set_page_config()

st.title("Classification Models in Financial Risk Analysis 📈")
st.caption("Compare, Learn, and Evaluate Machine Learning Classifiers")

tabs = st.tabs(["0. Home", "1. Explore Data", "2. EDA", "3. Select Model", "4. Learn Basics", "5. Evaluate Model", "6. Tuning", "7. Predictions", "8. Insights"])

raw_path = "data\\raw\\credit_default_data.csv"

processed_path = "data\\processed\\credit_default_data_processed.csv"

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
    st.markdown("""
        Welcome to the Classification Model Explorer!  
        This tool lets you explore and compare 9 commonly used classification models using a real-world inspired synthetic dataset (Credit Default Prediction).  
        Dive in to see how different models perform under the same conditions.
        """)
    
    st.markdown("### 🧰 What You Can Do")
    st.markdown("""
    - 🧠 **Run Models**: Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes, Gradient Boosting, LDA, Neural Network  
    - 📊 **Visualize Results**: Confusion matrix, ROC curves, and feature importance  
    - 🎯 **Tune Thresholds**: See how precision/recall trade-offs change with thresholds  
    - ⚖️ **Compare Models**: Evaluate side-by-side with standard metrics
    """)

    st.markdown("### 👤 Use Cases")
    st.markdown("""
    - Understand how classifiers behave on real data  
    - Evaluate accuracy vs interpretability trade-offs  
    - Visualize the impact of imbalanced classes, outliers, and preprocessing  
    - Perfect for students, analysts, and ML practitioners
    """)

    st.markdown("---", unsafe_allow_html=True)

    st.markdown(
        """
        <div style='text-align: center'>
            Built with ❤️ using Streamlit |  
            <a href='www.linkedin.com/in/prasad-dhamane' target='_blank'>
                <img src='https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg' width='20' style='vertical-align: middle; margin-bottom: 3px;'/>
                Connect on LinkedIn
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

with tabs[1]:

    st.subheader("📂 About the Dataset")
    st.markdown("""
        This dataset has been synthetically generated to simulate a real-world credit risk scenario where the goal is to **predict whether a customer will default** on their loan.

        It is designed specifically for learning and comparing machine learning classification algorithms.
        """)
    
    st.markdown("### 🧾 Dataset Features")

    st.dataframe({"Feature": ["age", "income", "loan_amount", "education", "marital_status", "employment_type", "credit_score", "default"],
                    "Type": ["Numerical", "Numerical", "Numerical", "Categorical", "Categorical", "Categorical", "Numerical", "Binary"],
                    "Description": ["Age of the client", "Monthly Income of the client", "Outstanding Loan amount", "HighSchool, Graduate, PostGraduate", "Single, Married, Divorced", "Salaried, Self-Employed, Unemloyed", "Credit score(350-850)", "Target variable (1=default, 0=no default)"]}, use_container_width=True)

    explore_data = st.button("View Data", use_container_width=True)
    
    if explore_data:
        with st.status("Loading... Please wait"):
            time.sleep(2)
        file_path = raw_path
        raw_data, data_description = eda.explore_data(file_path)
        st.caption("Raw Data")
        st.dataframe(raw_data, use_container_width=True)
        st.caption("Data Description")
        st.dataframe(data_description, use_container_width=True)
        st.info("**Note**: The Data looks messy and unstructured. We will need to perform Exploratory Data Analysis (EDA) to clean and preprocess it before training the model.")

with tabs[2]:
    st.markdown("### 🧪 Impurities in Synthetic Credit Default Dataset")

    # Data for the table
    data = {
        "🔍 Impurity Type": [
            "❌ Missing Values",
            "🎲 Skewed Distribution",
            "🤯 Outliers",
            "🔁 Multicollinearity",
            "📛 Categorical Typos & Imbalance",
            "⚖️ Imbalanced Target",
            "🔊 Noise"
        ],
        "🧩 Column(s)": [
            "income, loan_amount, employment_type",
            "income, credit_score",
            "age",
            "income, income_2",
            "marital_status",
            "default",
            "loan_amount"
        ],
        "📖 Explanation": [
            "Some rows contain NaN to simulate incomplete real-world records",
            "Income and credit score distributions are heavily right-skewed",
            "Included values like -5 or 150 to simulate data entry errors",
            "income_2 is a linear function of income with noise",
            "Used values like 'Singl', 'Marrid', 'married', 'DIVORCED' in mixed case",
            "Made 90% of the values 0, and only 10% as 1 (defaults are rare)",
            "Added Gaussian noise to loan amount values"
        ],
        "🎯 Purpose": [
            "Test imputation or row dropping strategies",
            "Demonstrate normalization/log scaling",
            "Test outlier detection (IQR, Z-score)",
            "Show impact of correlated features",
            "Demonstrate cleaning + label encoding",
            "Handle class imbalance using SMOTE/weights",
            "Test model robustness against noise"
        ]
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Display DataFrame
    st.markdown("### 📋 Impurity Summary Table")
    st.dataframe(df, use_container_width=True)

    # For consistent plot style
    sns.set(style="whitegrid")

    @st.cache_data
    def load_data():
        return pd.DataFrame({
            'age': [25, 45, -5, 50, 150, 40, 60, 30],
            'income': [40000, 80000, np.nan, 100000, 35000, 75000, 1e7, 'fifty-thousand'],
            'loan_amount': [2000, 10000, 5000, None, 1500, -5000, 15000, 4000],
            'employment_type': ['Salaried', np.nan, 'Self-employed', 'Salaried', 'Other', 'Salaried', np.nan, 'Self-employed'],
            'marital_status': ['Single', 'Marrid', 'Singl', 'DIVORCED', 'married', 'single', 'Yes', 'NO'],
            'default': [0, 1, 0, 1, 0, 0, 1, 0],
            'income_2': [44000, 82000, np.nan, 101000, 36000, 76000, 10000000, 51000],
            'credit_score': [300, 600, 750, 800, 400, 700, 900, 550]
        })

    df = load_data()

    st.subheader("🔍 Raw Data Snapshot")
    st.dataframe(df)

    # Step 1: Missing Values
    if st.checkbox("❌ Fill Missing Values"):
        df['income'] = pd.to_numeric(df['income'], errors='coerce')
        df['income'].fillna(df['income'].median(), inplace=True)
        df['loan_amount'].fillna(df['loan_amount'].median(), inplace=True)
        df['employment_type'].fillna("Unknown", inplace=True)
        st.success("✅ Missing values filled.")

    # Step 2: Fix Types
    if st.checkbox("🔁 Convert and Sanitize Data Types"):
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df['income'] = pd.to_numeric(df['income'], errors='coerce')
        df['income'].fillna(df['income'].median(), inplace=True)
        st.success("✅ Data types fixed.")

    # Step 3: Categorical Cleanup
    if st.checkbox("📛 Normalize Categorical Typos"):
        df['marital_status'] = df['marital_status'].str.lower().map({
            'single': 'single', 'singl': 'single', 'marrid': 'married',
            'married': 'married', 'divorced': 'divorced', 'yes': 'married', 'no': 'single'
        })
        st.success("✅ Cleaned marital status.")

    # Step 4: Outlier Handling + Plot
    if st.checkbox("🤯 Clip Outliers in Age"):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.boxplot(y=df['age'], ax=ax[0]).set_title("Before Clipping")

        q1, q3 = df['age'].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df['age'] = df['age'].clip(lower, upper)

        sns.boxplot(y=df['age'], ax=ax[1]).set_title("After Clipping")
        st.pyplot(fig)

    # Step 5: Normalize Skewed Income + Plot
    if st.checkbox("🎲 Apply Log Transform to Income"):
        df['income_log'] = np.log1p(df['income'])

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df['income'], bins=20, ax=ax[0], kde=True).set_title("Original Income")
        sns.histplot(df['income_log'], bins=20, ax=ax[1], kde=True).set_title("Log-Transformed Income")
        st.pyplot(fig)

    # Step 6: Drop Multicollinearity Column
    if st.checkbox("🔁 Drop Correlated Column (income_2)"):
        df.drop(columns=['income_2'], inplace=True)
        st.success("✅ Dropped income_2")

    # Step 7: Encode Categorical Columns
    if st.checkbox("🔤 Label Encode Categorical Features"):
        le = LabelEncoder()
        for col in ['employment_type', 'marital_status']:
            df[col] = le.fit_transform(df[col].astype(str))
        st.success("✅ Label encoded.")
        st.dataframe(df)

    # Step 8: Class Imbalance Check
    if st.checkbox("⚖️ Show Class Imbalance (Target Variable)"):
        st.subheader("Class Distribution")
        st.bar_chart(df['default'].value_counts())

    # Step 9: Final Cleanup
    if st.checkbox("🧽 Drop Duplicates and Reset Index"):
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        st.success("✅ Final cleanup done.")
        st.dataframe(df)

    st.markdown("---")
    st.success("🎉 Data cleaning and visualization completed!")

with tabs[3]:

    st.subheader("Select Classification Model")
    st.session_state.selected_model = st.selectbox("--Select from dropdown--", ["Logistic Regression", 
                                    "Decision Tree",
                                    "Random Forest",
                                    "Support Vector Machine",
                                    "K-Nearest Neighbors",
                                    "Naive Bayes",
                                    "XGBoost",
                                    "Neural Network",
                                    "Linear Discriminant Analysis"])

    if st.session_state.selected_model == "Logistic Regression":
        st.subheader("Logistic Regression 🧮")
        st.markdown("**🔍 Type**: Linear, Binary/Multi-Class Classification")
        st.markdown("**📊 Use Case**: Credit Default Prediction, Email Spam Detection")
        st.markdown("**✨ Key Point**: Outputs Probabilities using a Sigmoid function")
        st.markdown("**✅ Pros**: Simple, interpretable, fast")
        st.markdown("**⚠️ Cons**: Assumes linearity, sensitive to outliers")

    elif st.session_state.selected_model == "Decision Tree":
        st.subheader("Decision Tree 🌳")
        st.markdown("**🔍 Type**: Non-Linear, Binary/Multi-Class Classification")
        st.markdown("**📊 Use Case**: Medical Diagnosis, Customer Segmentation")
        st.markdown("**✨ Key Point**: Splits data based on feature values")
        st.markdown("**✅ Pros**: Easy to interpret, handles non-linear data")
        st.markdown("**⚠️ Cons**: Prone to overfitting, sensitive to noise")

    elif st.session_state.selected_model == "Random Forest":
        st.subheader("Random Forest 🌲")
        st.markdown("**🔍 Type**: Ensemble, Non-Linear, Binary/Multi-Class Classification")
        st.markdown("**📊 Use Case**: Credit Scoring, Fraud Detection")
        st.markdown("**✨ Key Point**: Combines multiple decision trees for better accuracy")
        st.markdown("**✅ Pros**: Reduces overfitting, handles large datasets")
        st.markdown("**⚠️ Cons**: Less interpretable, slower to train")

    elif st.session_state.selected_model == "Support Vector Machine":
        st.subheader("Support Vector Machine (SVM) 🛡️")
        st.markdown("**🔍 Type**: Non-Linear, Binary/Multi-Class Classification")
        st.markdown("**📊 Use Case**: Image Classification, Text Categorization")
        st.markdown("**✨ Key Point**: Finds the optimal hyperplane to separate classes")
        st.markdown("**✅ Pros**: Effective in high-dimensional spaces, robust to overfitting")
        st.markdown("**⚠️ Cons**: Memory-intensive, less effective on large datasets")

    elif st.session_state.selected_model == "K-Nearest Neighbors":
        st.subheader("K-Nearest Neighbors (KNN) 👥")
        st.markdown("**🔍 Type**: Non-Linear, Binary/Multi-Class Classification")
        st.markdown("**📊 Use Case**: Recommender Systems, Anomaly Detection")
        st.markdown("**✨ Key Point**: Classifies based on the majority class of nearest neighbors")
        st.markdown("**✅ Pros**: Simple, effective for small datasets")
        st.markdown("**⚠️ Cons**: Computationally expensive, sensitive to irrelevant features")

    elif st.session_state.selected_model == "Naive Bayes":
        st.subheader("Naive Bayes 🐦")
        st.markdown("**🔍 Type**: Probabilistic, Binary/Multi-Class Classification")
        st.markdown("**📊 Use Case**: Text Classification, Spam Detection")
        st.markdown("**✨ Key Point**: Assumes independence between features")
        st.markdown("**✅ Pros**: Fast, works well with high-dimensional data")
        st.markdown("**⚠️ Cons**: Assumes feature independence, less effective with correlated features")

    elif st.session_state.selected_model == "XGBoost":
        st.subheader("XGBoost 🚀")
        st.markdown("**🔍 Type**: Ensemble, Non-Linear, Binary/Multi-Class Classification")
        st.markdown("**📊 Use Case**: Kaggle Competitions, Credit Scoring")
        st.markdown("**✨ Key Point**: Boosting algorithm that builds trees sequentially")
        st.markdown("**✅ Pros**: High performance, handles missing values")
        st.markdown("**⚠️ Cons**: Complex, requires tuning")

    elif st.session_state.selected_model == "Neural Network":
        st.subheader("Neural Network 🧠")
        st.markdown("**🔍 Type**: Non-Linear, Binary/Multi-Class Classification")
        st.markdown("**📊 Use Case**: Image Recognition, Natural Language Processing")
        st.markdown("**✨ Key Point**: Mimics human brain structure with layers of neurons")
        st.markdown("**✅ Pros**: Handles complex patterns, scalable")
        st.markdown("**⚠️ Cons**: Requires large datasets, less interpretable")

    elif st.session_state.selected_model == "Linear Discriminant Analysis":
        st.subheader("Linear Discriminant Analysis (LDA) 📊")
        st.markdown("**🔍 Type**: Linear, Binary/Multi-Class Classification")
        st.markdown("**📊 Use Case**: Face Recognition, Medical Diagnosis")
        st.markdown("**✨ Key Point**: Projects data onto a lower-dimensional space for classification")
        st.markdown("**✅ Pros**: Works well with small datasets, interpretable")
        st.markdown("**⚠️ Cons**: Assumes normal distribution, sensitive to outliers")

    # st.subheader("Exploratory Data Analysis")
    # st.write("This is where you will explore the data.")

    # with st.expander("**Why is Exploratory Data Analysis (EDA) Important in ML?**"):

    #     st.markdown("""
    #     **1. Data Quality Check**
    #     📌 Identifies missing values, duplicates, and inconsistencies in the dataset.

    #     **2. Feature Understanding**
    #     🔍 Helps determine which features are relevant for model building.

    #     **3. Outlier Detection**
    #     🚨 Finds extreme values that could negatively impact model performance.

    #     **4. Distribution Analysis**
    #     📊 Understands how data is spread to apply proper preprocessing techniques.

    #     **5. Correlation & Relationships**
    #     🔗 Reveals dependencies between variables to improve feature selection.

    #     **6. Improves Model Accuracy**
    #     🚀 Ensures clean, well-structured data, leading to better predictions.
    #     """)

    #     st.info("Without proper EDA, even the best ML models may fail due to poor data quality.")

    # st.markdown("**1. Data Cleaning**: Fill missing values using mean, median, or mode, or drop them if necessary.")

    # if st.button("Clean the selected dataset", use_container_width=True):
    #     file_path = raw_paths[selected_model]
    #     missing_row_df = eda.check_missing_rows(file_path)
    #     st.caption("Missing Rows Data")
    #     st.dataframe(missing_row_df.head())
    #     if missing_row_df.shape[0] > 0:
    #         st.warning(f"{missing_row_df.shape[0]} Missing values found in the dataset! Please clean the data.")

    #         with st.status("Cleaning... Please wait"):
    #             eda.remove_missing_rows(file_path, missing_row_df)
    #             time.sleep(3)
    #             st.success("Data cleaned successfully!")
        
    #         st.markdown("### Data Cleaning Steps")
    #         st.write("1. Identify missing values.")
    #         st.write("2. Decide how to handle them (e.g., imputation, removal). We will remove them for simplicity.")
    #         st.write("3. Apply the chosen method to clean the data.")
    #         st.write("4. Verify that the data is clean.")

    #     else:
    #         st.success("No missing values found in the dataset! Proceed with EDA")
    
    # st.markdown("**2. Class Distribution Analysis**: Visualize class distribution to check if the data is balanced or imbalanced.")

    # if st.button("View Class Distribution", use_container_width=True):
    #     file_path = processed_paths[selected_model]
    #     target_column = target_columns[selected_model]
    #     with st.status("Plotting... Please wait"):
    #         fig, ratio = eda.plot_class_distribution(file_path, target_column)
    #         time.sleep(2)
    #     st.pyplot(fig)

    #     st.markdown(f"""
    #     <div style="text-align: center;">
    #         <p><strong>Class 0 (No Default):</strong> {ratio[0]:.2f}</p>
    #         <p><strong>Class 1 (Default):</strong> {ratio[1]:.2f}</p>
    #     </div>
    #     """, unsafe_allow_html=True)

    #     st.info("**Note**: Typically, if the ratio of the majority class to the minority class exceeds 80-20 (or is even more extreme, like 90-10), it’s considered imbalanced.")

    #     if ratio[0] > 0.9:
    #         st.warning("The dataset is imbalanced. Consider using techniques like SMOTE or ADASYN for balancing.")
    #     else:
    #         st.success("The dataset is balanced. No further action needed.")
        
    #     with st.expander("When is a dataset considered imbalanced?"):
    #         markdown_text = """
    #         A dataset is **imbalanced** when one class appears significantly more frequently than others.  
    #         Typically, if the ratio of the majority class to the minority class exceeds **80-20** (or is even more extreme, like **90-10**),  
    #         it’s considered imbalanced. This is a problem because models tend to favor the dominant class,  
    #         leading to poor generalization and weak predictive power for the minority class.

    #         ### **How to fix class imbalance?**
    #         Here are some techniques to handle imbalance:

    #         **1. Resampling (Oversampling & Undersampling)**  
    #         - **Oversampling the minority class**: Create synthetic samples of the minority class using methods like  
    #         **SMOTE (Synthetic Minority Over-sampling Technique)**.  
    #         - **Undersampling the majority class**: Remove random samples from the dominant class to balance the dataset.

    #         **2. Adjust Class Weights**  
    #         - Many machine learning models allow you to set **class weights** so that errors in predicting the minority class  
    #         are penalized more, forcing the model to pay attention to them.

    #         **3. Use Different Evaluation Metrics**  
    #         - Instead of accuracy (which is misleading in imbalanced datasets), use:  
    #         - **Precision & Recall**  
    #         - **F1-score**  
    #         - **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**  

    #         **4. Try Advanced Algorithms**  
    #         Some algorithms handle imbalance better than others:  
    #         - Tree-based methods like **XGBoost** allow you to set parameters to focus on rare events.  
    #         - **Anomaly detection methods** (useful if minority cases are rare but crucial).  
    #         """
    #         st.markdown(markdown_text)

    # st.markdown("**3. Data Visualization**: Use visualizations to understand relationships between features and the target variable.")
    
    # if st.button("View Data Visualization", use_container_width=True):
    #     file_path = processed_paths[selected_model]
    #     target_column = target_columns[selected_model]
    #     with st.status("Plotting... Please wait"):
    #         plot = eda.plot_feature_target_correlation(file_path, target_column)
    #         time.sleep(2)
    #     st.pyplot(plot)

    #     st.markdown("### Insights from Feature-Target Correlation")
    #     st.write("The bar chart above shows the correlation of each feature with the target variable. Here are some key insights:")
        
    #     st.markdown("""
    #     - **PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6**: These features have the highest positive correlation with the target variable, indicating that repayment status is a strong predictor of default.
    #     - **LIMIT_BAL**: Shows a slight negative correlation, suggesting that higher credit limits may reduce the likelihood of default.
    #     - **BILL_AMT and PAY_AMT features**: These have weak correlations with the target, indicating they may not be as significant for predicting default.
    #     - **ID**: Has almost no correlation with the target, as expected, since it is just an identifier.
    #     """)

    #     st.info("**Note**: Features with high correlation (positive or negative) are more likely to be important for the model. However, feature selection should also consider multicollinearity and domain knowledge.")

    #     st.success("EDA completed! You can explore further by adding techniques like feature scaling, correlation heatmaps, or PCA. Once satisfied, proceed to the Model Training section.")

with tabs[4]:
    st.subheader(f"🧠 {st.session_state.selected_model} – Step-by-Step Guide")
    
    if st.session_state.selected_model == "Logistic Regression":
        st.markdown("""
        This interactive guide walks you through the complete logic of Logistic Regression – 
        from score calculation to sigmoid conversion, loss function, and gradient descent updates.
        """)

        st.markdown("---")

        st.markdown("### 👇 Example Setup")
        st.markdown("""
        - A single feature value `x = 4.0`
        - Initial weight `w = 0.8`
        - Initial bias `b = -0.5`
        - True class label `y = 0 or 1`
        - Learning rate `α` to control the update step size

        You can adjust these values below to see how they affect the prediction, loss, and updates.
        """)

        x_val = st.number_input("Feature value (x)", value=4.0)
        w_val = st.number_input("Initial weight (w)", value=0.8)
        b_val = st.number_input("Initial bias (b)", value=-0.5)
        actual_y = st.radio("Actual class (y)", options=[0, 1], horizontal=True)
        alpha = st.slider("Learning Rate (α)", 0.01, 1.0, 0.1)

        z = w_val * x_val + b_val

        st.markdown("---")

        st.markdown("### Step 1: Linear Combination")
        st.info("""
        **What:** Combine input feature and weight linearly to compute score $z$.
                
        **Why:** It generates the base value before transforming into probability.
        """)
        st.latex(r"\quad z = w \cdot x + b")

        st.latex(rf''' z = {w_val:.2f} \times {x_val:.2f} + {b_val:.2f} = {z:.2f} ''')

        # Visual hint
        st.markdown("#### 📈 Visual: Linear Relationship")
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.plot([0, x_val], [b_val, z], marker='o')
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        st.pyplot(fig)

        st.markdown("---")

        st.markdown("### Step 2: Sigmoid Function")
        st.info("""
        **What:** Transform raw score $z$ into a probability.
        **Why:** Logistic regression predicts the probability of class 1.
            """)
        st.latex(r''' \sigma(z) = \frac{1}{1 + e^{-z}} ''')

        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        sig_val = sigmoid(z)
        st.latex(rf'''\sigma({z:.2f}) = {sig_val:.4f} \rightarrow \text{{Probability of class {1 if sig_val > 0.5 else 0}}}''')

        x_plot = np.linspace(-10, 10, 100)
        y_plot = sigmoid(x_plot)
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.plot(x_plot, y_plot)
        ax2.axvline(x=z, color='red', linestyle='--', label=f'z = {z:.2f}')
        ax2.axhline(y=sig_val, color='blue', linestyle='--', label=f'ŷ = {sig_val:.2f}')
        ax2.set_title("Sigmoid Function")
        ax2.set_xlabel("z")
        ax2.set_ylabel("σ(z)")
        ax2.legend()
        st.pyplot(fig2)

        st.markdown("---")

        st.markdown("### Step 3: Loss Function (Binary Cross-Entropy)")
        st.info("""
        **What:** Evaluate how wrong the predicted probability is.
                
        **Why:** The loss function guides the learning by penalizing wrong predictions.
        """)
        st.latex(r'''\mathcal{L} = - \left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]''')
    
        loss = - (actual_y * np.log(sig_val + 1e-8) + (1 - actual_y) * np.log(1 - sig_val + 1e-8))
        st.latex(rf'''\text{{Loss}} = {loss:.4f}''')

        st.markdown("---")

        st.markdown("### Step 4: Gradient Descent Update")
        st.info("""
        **What:** Improve model by adjusting weights and bias.
                
        **Why:** Minimize the loss by learning from the error.
            """)
        st.latex(r'''
            \begin{aligned}
            w &:= w - \alpha (\hat{y} - y) x \\
            b &:= b - \alpha (\hat{y} - y)
            \end{aligned}
            ''')

        error = sig_val - actual_y
        new_w = w_val - alpha * error * x_val
        new_b = b_val - alpha * error
        st.latex(r'''\text{{Error}} = \hat{y} - y''')
        st.latex(rf'''\text{{Error}} = {error:.4f}''')
        st.latex(rf''' w: {w_val:.4f} \rightarrow {new_w:.4f} ''')
        st.latex(rf''' b: {b_val:.4f} \rightarrow {new_b:.4f} ''')


        st.markdown("---")

        st.markdown("### Step 5: What’s Next?")
        st.info("""
        **What:** Use updated weights and bias in the next forward pass.  
        **Why:** Gradient descent is an iterative optimization process.  
        **How:** Replace old `w` and `b` with updated values and repeat Steps 1–4.
        """)
        st.markdown("""
        **When to Stop:**  
        - When loss change is below a small threshold (e.g., 0.00001)  
        - Or after a maximum number of epochs  
        - Or when validation accuracy stops improving  

        This completes one training step. Repeat with more data samples or in batches to train a full model.
        """)

        st.markdown("---")

        st.markdown("### Step 6: Final Prediction")
        st.info(r"""
        **What:** Use the trained model to make predictions on new data.
                """)
        st.markdown(""" 
        **How:**  
        - Multiply input features by learned weights  
        - Apply the sigmoid function  
        - Convert the probability to 0 or 1 → prediction done!  
        """)

        st.success("✅ You've walked through every step of Logistic Regression using a concrete example!")

    elif st.session_state.selected_model == "Decision Tree":
        st.markdown("### 🤔 What is a Decision Tree?")
        st.markdown("""
        A decision tree is a flowchart-like structure used for classification or regression. It splits data into branches based on conditions (questions), until reaching a prediction.

        We’ll use a toy binary classification example to demonstrate how decision trees work step by step.
        """)

        # Sample dataset
        data = {
            "Weather": ["Sunny", "Sunny", "Overcast", "Rainy", "Rainy",
                        "Rainy", "Overcast", "Sunny", "Sunny", "Rainy"],
            "Play Tennis": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "Yes", "Yes", "Yes"]
        }

        df = pd.DataFrame(data)

        # Streamlit app layout
        st.markdown("### 🎾 Tennis Dataset - Weather vs Play Decision")
        st.markdown("This is the dataset we'll use to understand how a **Decision Tree Classifier** works.")

        st.markdown("### 📄 Sample Dataset")
        st.dataframe(df, use_container_width=True)

        # Title
        st.markdown("### 🎾 Step 1: Class Distribution & Root Node Impurity")

        # Explanation
        st.info("""
        **What:**  
        We are calculating how "impure" or "mixed" our target values are before making any split.

        **Why:**  
        This impurity tells us how useful a split will be. The goal of a decision tree is to reduce this impurity by splitting the dataset.

        **How:**  
        We use either: Gini Index or Entropy to measure impurity.
                """)
        st.latex(r"Gini = 1 - p_{yes}^2 - p_{no}^2")
        st.latex(r"Entropy = -p_{yes} \log_2(p_{yes}) - p_{no} \log_2(p_{no})")

        st.markdown("---")

        # Dropdown to select criterion
        st.session_state.criterion = st.selectbox("**🔧 Choose Split Criterion**", ["gini", "entropy"])

        # Calculate class counts
        st.session_state.yes_count = df["Play Tennis"].value_counts().get("Yes", 0)
        st.session_state.no_count = df["Play Tennis"].value_counts().get("No", 0)
        st.session_state.total = st.session_state.yes_count + st.session_state.no_count

        # Probabilities
        st.session_state.p_yes = st.session_state.yes_count / st.session_state.total
        st.session_state.p_no = st.session_state.no_count / st.session_state.total

        # Display class distribution
        st.markdown("#### 📊 Class Distribution")
        st.write(f"✔️ Yes: {st.session_state.yes_count}, ❌ No: {st.session_state.no_count} (Total: {st.session_state.total})")
        st.latex(r"p_{yes} = \frac{%d}{%d} = %.2f \quad p_{no} = \frac{%d}{%d} = %.2f"
                % (st.session_state.yes_count, st.session_state.total, st.session_state.p_yes, st.session_state.no_count, st.session_state.total, st.session_state.p_no))

        # Compute impurity
        if st.session_state.criterion == "gini":
            impurity = 1 - st.session_state.p_yes**2 - st.session_state.p_no**2
            st.markdown("#### 🔢 Gini Index at Root Node")
            st.latex(r"Gini = 1 - p_{yes}^2 - p_{no}^2")
            st.latex(r"= 1 - %.2f - %.2f = %.3f" % (st.session_state.p_yes**2, st.session_state.p_no**2, impurity))
        else:
            entropy = 0
            if st.session_state.p_yes > 0:
                entropy -= st.session_state.p_yes * np.log2(st.session_state.p_yes)
            if st.session_state.p_no > 0:
                entropy -= st.session_state.p_no * np.log2(st.session_state.p_no)
            impurity = entropy
            st.markdown("#### 🔢 Entropy at Root Node")
            st.latex(r"Entropy = -p_{yes} \log_2(p_{yes}) - p_{no} \log_2(p_{no})")
            st.latex(r"= -%.2f \log_2(%.2f) - %.2f \log_2(%.2f) = %.3f" % (st.session_state.p_yes, st.session_state.p_yes, st.session_state.p_no, st.session_state.p_no, entropy))
        
        st.session_state.root_impurity = round(impurity, 3)

        st.markdown("---")

        st.markdown("### 🌤️ Step 2: Splitting Based on 'Weather' Feature")

        # Explanation
        st.info("""
        **What:**  
        We'll evaluate each group within the 'Weather' feature (Sunny, Overcast, Rainy).

        **Why:**  
        To determine how much each split reduces impurity in the dataset.

        **How:**  
        For each subset:
        - Calculate the proportion of Yes/No
        - Compute Gini or Entropy
        - Weight each subset's impurity by its size
        - Add them for overall split impurity
        """)

        # Total size
        total = len(df)

        # Group by Weather
        split_data = []
        for weather_type in df["Weather"].unique():
            subset = df[df["Weather"] == weather_type]
            yes = sum(subset["Play Tennis"] == "Yes")
            no = sum(subset["Play Tennis"] == "No")
            subset_total = yes + no
            p_yes = yes / subset_total if subset_total else 0
            p_no = no / subset_total if subset_total else 0

            # Calculate impurity
            if st.session_state.criterion == "gini":
                impurity = 1 - p_yes**2 - p_no**2
            else:
                impurity = 0
                if p_yes > 0:
                    impurity -= p_yes * np.log2(p_yes)
                if p_no > 0:
                    impurity -= p_no * np.log2(p_no)

            weight = subset_total / total
            weighted_impurity = weight * impurity

            split_data.append({
                "Weather": weather_type,
                "Total": subset_total,
                "Yes": yes,
                "No": no,
                "Impurity": round(impurity, 3),
                "Weighted Impurity": round(weighted_impurity, 3)
            })

        # Show results in a table
        split_df = pd.DataFrame(split_data)
        st.markdown("### 📊 Impurity of Each Weather Subset")
        st.dataframe(split_df, use_container_width=True, column_config={"Total": st.column_config.TextColumn(width="small"),
                                                                    "Yes": st.column_config.TextColumn(width="small"),
                                                                    "No": st.column_config.TextColumn(width="small"),
                                                                    "Impurity": st.column_config.TextColumn(width="small"),
                                                                    "Weighted Impurity": st.column_config.TextColumn(width="medium")})

        # Show weighted total
        st.session_state.split_impurity = round(split_df["Weighted Impurity"].sum(), 3)
        st.markdown("**🧮 Total Weighted Impurity After Weather Split**")
        if st.session_state.criterion == "gini":
            st.latex(fr"Gini_{{split}} = {st.session_state.split_impurity}")
        else:
            st.latex(fr"Entropy_{{split}} = {st.session_state.split_impurity}")

        st.markdown("---")

        st.markdown("### ✅ Step 3: Evaluate 'Weather' Split Using Gain")

        st.info("""
        **What:**  
        Gain measures how much impurity is reduced by the split.

        **Why:**  
        Higher gain indicates a better split, leading to more accurate predictions.
                """)

        # Step 3: Gain calculation
        st.session_state.gain = st.session_state.root_impurity - st.session_state.split_impurity

        st.markdown("### 🧮 Gain Calculation")
        if st.session_state.criterion == "gini":
            st.latex(fr"Gini_{{gain}} = Gini_{{root}} - Gini_{{split}}")
            st.latex(fr"Gain = {st.session_state.root_impurity:.3f} - {st.session_state.split_impurity:.3f} = {st.session_state.gain:.3f}")
        else:
            st.latex(fr"Info\ Gain = Entropy_{{root}} - Entropy_{{split}}")
            st.latex(fr"Gain = {st.session_state.root_impurity:.3f} - {st.session_state.split_impurity:.3f} = {st.session_state.gain:.3f}")

        st.success(f"🎯 The gain from splitting on 'Weather' is **{st.session_state.gain:.3f}**. Higher gain means better split!")

        st.markdown("---")

        st.markdown("### 📈 Step 4: Visualize the Decision Tree Using 'Weather' as Feature")
        
        # Encode categorical features
        le_weather = LabelEncoder()
        le_label = LabelEncoder()
        df["Weather_Encoded"] = le_weather.fit_transform(df["Weather"])
        df["Play_Encoded"] = le_label.fit_transform(df["Play Tennis"])

        max_depth = st.slider("**Max Depth**", min_value=1, max_value=2, value=1, step=1)

        # Train model
        X = df[["Weather_Encoded"]]
        y = df["Play_Encoded"]
        model = DecisionTreeClassifier(criterion=st.session_state.criterion, max_depth=max_depth, random_state=42)
        model.fit(X, y)

        # Generate Graphviz DOT
        dot_data = export_graphviz(
            model,
            out_file=None,
            feature_names=["Weather"],
            class_names=le_label.classes_,
            filled=True,
            rounded=True,
            special_characters=True
        )

        # Render in Streamlit
        st.markdown("### 🌐 Graphviz Decision Tree")
        st.graphviz_chart(dot_data, use_container_width=True)

        # Legend
        st.markdown("### 🧾 Legend")
        legend = {
            "Sunny": le_weather.transform(["Sunny"])[0],
            "Overcast": le_weather.transform(["Overcast"])[0],
            "Rainy": le_weather.transform(["Rainy"])[0],
            "Yes": le_label.transform(["Yes"])[0],
            "No": le_label.transform(["No"])[0]
        }
        st.write("Encoded values used in the tree:")
        st.write(legend)

        # Short interpretation in Streamlit
        st.markdown("### 🧠 Interpretation (Quick Summary)")
        st.markdown("""
        - The **root node** splits on the `Weather` feature based on its encoded values.
        - Each **leaf node** shows:
        - `samples`: Number of records
        - `value`: Count of [No, Yes]
        - `class`: Predicted outcome (based on majority)
        - A **lower impurity** means more confident predictions.
        - **Darker colors** = purer nodes (stronger class dominance).
        """)

        st.success("✅ You've walked through every step of Logistic Regression using a concrete example!")
    # model_name = st.selectbox("Select Model", models[selected_model])

    # st.markdown("### Model Parameters")
    # if model_name == "Random Forest":
    #     n_estimators = st.slider("Number of Trees", min_value=10, max_value=100, value=50, step=10)
    #     max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=5, step=1)
    # elif model_name == "XGBoost":
    #     n_estimators = st.slider("Number of Trees", min_value=10, max_value=100, value=50, step=10)
    #     learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.1, step=0.01)
    # elif model_name == "Logistic Regression":
    #     with st.expander("**First learn about the important hyperparameters**"):
    #         # Markdown content for parameters
    #         markdown_text = """
    #         **1. Penalty (`penalty`)**
    #         - Adds regularization to prevent overfitting. 
    #             - `l1` (Lasso): Shrinks some coefficients to zero (feature selection).
    #             - `l2` (Ridge): Shrinks coefficients without eliminating features.
    #             - `elasticnet`: Combines Lasso and Ridge.

    #         **2. Regularization Strength (`C`)**
    #         - Controls regularization strength. 
    #             - Lower values: Stronger regularization (less overfitting).
    #             - Higher values: Weaker regularization (more flexibility).

    #         **3. Solver (`solver`)**
    #         - Optimization algorithm.
    #             - `liblinear`: Small datasets, L1 penalty.
    #             - `lbfgs`, `newton-cg`: Large datasets, L2 penalty.
    #             - `sag`, `saga`: Very large datasets, supports both penalties.

    #         **4. Maximum Iterations (`max_iter`)**
    #         - Number of optimization iterations.
    #             - Increase if convergence isn’t reached.
    #             - Too high values may slow training.

    #         **5. Class Weight (`class_weight`)**
    #         - Adjusts weights for imbalanced classes.
    #             - `'balanced'`: Auto-balances weights based on class frequencies.
    #             - Custom weights: Improve predictions for imbalanced datasets.
    #         """
    #         # Display content using Streamlit markdown
    #         st.markdown(markdown_text)

    #     penalty = st.selectbox("**Penalty**", ["l1", "l2", "none"])
    #     C = st.slider("**Inverse Regularization Strength**", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
    #     solver = st.selectbox("**Solver**", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"])
    #     max_iter = st.number_input("**Max Iterations**", min_value=100, max_value=1000, value=200, step=100)
    #     class_weight = st.selectbox("**Class Weight**", ["balanced", None])

    #     test_size = st.slider("**Test Size**", min_value=0.1, max_value=0.5, value=0.2, step=0.01)
    #     random_state = st.number_input("**Random State**", min_value=0, max_value=100, value=42, step=1)

    #     if st.button("Train Model", use_container_width=True):
    #         with st.status("Training... Please wait", expanded=True):
    #             file_path = processed_paths[selected_model]
    #             target_column = target_columns[selected_model]
    #             try:
    #                 accuracy, confusion, report, fig, roc_fig, feature_importance_fig, trained_model, X_train_scaled, X_test_scaled, y_train, y_test, scaler \
    #                 = LogisticRegression.run_logistic_regression_pipeline(
    #                     file_path, target_column, test_size, random_state, solver, C, penalty, class_weight
    #                 )
    #                 st.success("Model trained successfully!")

    #                 # Store the variables in session state
    #                 st.session_state.accuracy = accuracy
    #                 st.session_state.confusion = confusion
    #                 st.session_state.report = report
    #                 st.session_state.fig = fig
    #                 st.session_state.roc_fig = roc_fig
    #                 st.session_state.feature_importance_fig = feature_importance_fig
    #                 st.session_state.trained_model = trained_model
    #                 st.session_state.X_train_scaled = X_train_scaled
    #                 st.session_state.X_test_scaled = X_test_scaled
    #                 st.session_state.y_train = y_train
    #                 st.session_state.y_test = y_test
    #                 st.session_state.scaler = scaler

    #             except Exception as e:
    #                 st.error(f"Error: {e}")
    #                 st.warning("Please check the parameters and try again.")
                
    #         st.info("Interpreting model performance metrics is essential for understanding how well your classification model is performing. Let’s break down key metrics in the next section.")    

with tabs[5]:
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

with tabs[6]:
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
                file_path = processed_path
                target_column = target_columns[selected_model]
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

with tabs[7]:
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

    input_data = pd.read_csv(sample_paths[selected_model])
    input_data.at[0, 'LIMIT_BAL'] = int(limit_bal)
    input_data.at[0, 'SEX'] = int(sex)
    input_data.at[0, 'EDUCATION'] = int(education)
    input_data.at[0, 'MARRIAGE'] = int(marriage)
    input_data.at[0, 'AGE'] = int(age)
    X = input_data.drop(target_columns[selected_model], axis=1)

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

with tabs[8]:
    st.subheader("Model Insights")
    st.write("This is where you will provide insights about the model.")

    with st.expander("**Model Insights**"):
        st.markdown("""
        - **Feature Importance**: The most important features for predicting loan default are PAY_0, PAY_2, and PAY_3.
        - **Model Performance**: The model achieved an accuracy of 0.80, indicating it correctly predicted 80% of the cases.
        - **Class Imbalance**: The dataset was imbalanced, but techniques like SMOTE can help balance it.
        - **ROC Curve**: The ROC curve shows a good trade-off between true positive and false positive rates.
        """)

