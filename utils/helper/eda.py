import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(file_path):
    raw_data = pd.read_csv(file_path)
    data_description = raw_data.describe()
    return [raw_data, data_description]

def remove_missing_rows(file_path, missing_row_df):
    raw_data = pd.read_csv(file_path)
    cleaned_data = raw_data.drop(missing_row_df.index)
    cleaned_data.to_csv("data/processed/UCI_Credit_Card_Cleaned.csv", index=False)

def check_missing_rows(file_path):
    raw_data = pd.read_csv(file_path)
    missing_row_df = raw_data[raw_data.isnull().any(axis=1)]
    return missing_row_df

def plot_class_distribution(file_path, target_column):
    df = pd.read_csv(file_path)

    class_counts = df[target_column].value_counts()
    total_samples = len(df)
    class_ratio = class_counts / total_samples


    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=target_column, palette="coolwarm")

    plt.xlabel("Classes")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    plt.xticks(rotation=45)
    
    return plt, class_ratio

def plot_feature_target_correlation(file_path, target_column):
    df = pd.read_csv(file_path)

    # Calculate correlation of features with the target variable
    corr_with_target = df.corr()[target_column].drop(target_column)

    # Plot a simple bar plot for feature-target correlations
    plt.figure(figsize=(10, 6))
    plt.bar(corr_with_target.index, corr_with_target.values, color='skyblue')
    plt.title("Feature-Target Correlation")
    plt.xlabel("Features")
    plt.ylabel("Correlation with Target")
    plt.xticks(rotation=45, ha='right')

    return plt

