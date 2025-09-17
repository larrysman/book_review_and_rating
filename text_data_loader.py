# STAGE 01: LOADING THE TEXT DATASET

import pandas as pd
import os

# LOAD THE DATASET
def load_text_data():
    print("Enter the path to your dataset (CSV format): ")
    path = input()
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"The columns in the dataset are: {df.columns.tolist()}")
    else:
        raise FileNotFoundError(f"The file at {path} was not found.")
    return df


if __name__ == "__main__":
    df = load_text_data()
    print("Data loaded successfully.")
    print(f"Dataset shape: {df.shape}")
    print(df.head(2))