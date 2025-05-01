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
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=target_column, palette="coolwarm")

    plt.xlabel("Classes")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    plt.xticks(rotation=45)
    
    return plt

