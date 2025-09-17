# STAGE 03: TEXT TOKENIZATION, STOPWORD REMOVAL & LEMMATIZATION

import spacy
import subprocess
import pandas as pd

import text_cleaning

# ENSURE THE SPACY ENGLISH MODEL IS INSTALLED
subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])

# LOAD THE ENGLISH LANGUAGE MODEL
english_word = spacy.load("en_core_web_sm")


def token_stopwords_lemmatization_fns(text):

    # CONVERT THE TEXT TO A SPACY DOCUMENT INTO ENGLISH
    document_text = english_word(text)

    # PERFORM TOKENIZATION, STOP WORD REMOVAL AND LEMMATIZATION
    return " ".join([TOKENS.lemma_ for TOKENS in document_text if not TOKENS.is_stop])


def tokenize_stopwords_lemmatization_column():
    df, text_col = text_cleaning.text_cleaning_fn()
    if text_col not in df.columns:
        raise ValueError(f"The column {text_col} does not exist in the dataset.")
    else:
        df["full_cleaned"] = df[text_col].astype(str).apply(token_stopwords_lemmatization_fns)

    print(f"Select the full_cleaned and one other column as your target (review/score): {df.columns.tolist()}")
    feature = input("Select the feature column: ")
    target = input("Select the target column: ")
    if feature and target not in df.columns:
        raise ValueError(f"The columns {feature} and {target} does not exist in the dataset.")
    else:
        df = df[[feature, target]]
    return df


if __name__ == "__main__":
    df = tokenize_stopwords_lemmatization_column()
    print(df.head())