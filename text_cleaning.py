# STAGE 02: TEXT CLEANING

import numpy as np
import pandas as pd
import re
import os

import text_data_loader


def text_cleaning_fn():
    df = text_data_loader.load_text_data()
    print("Enter the name of the text column to clean: ")
    text_col = input()
    if text_col not in df.columns:
        raise ValueError(f"The column {text_col} does not exist in the dataset.")
    else:
        df[text_col] = df[text_col].astype(str).str.lower()
        df[text_col] = df[text_col].str.replace(r"[^a-z\s]", "", regex=True)
        df[text_col] = df[text_col].str.replace(r"\s+", " ", regex=True).str.strip()

    return df, text_col
    

if __name__ == "__main__":
    df_cleaned, column_name = text_cleaning_fn()
    print(f"Text cleaning completed on column: {column_name}")
    print("Text cleaning completed.")
    print(df_cleaned.head(2))