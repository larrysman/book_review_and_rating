# STAGE 04: TEXT PREPROCESSING


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import text_toks_lem_stw


def text_preprocessing(stop_words="english", min_df=0.1):

    TFIDFvector = TfidfVectorizer(stop_words=stop_words, min_df=min_df)

    df = text_toks_lem_stw.tokenize_stopwords_lemmatization_column()

    df.rename(columns={"review/score": "score"}, inplace=True)

    dtm = TFIDFvector.fit_transform(df["full_cleaned"])

    df_array = dtm.toarray()

    df_final = pd.DataFrame(df_array, columns=TFIDFvector.get_feature_names_out())

    df_final = pd.concat([df_final, df["score"].reset_index(drop=True)], axis=1)

    return TFIDFvector, df_final


if __name__ == "__main__":
    vectorizer, df = text_preprocessing()
    print(df.head())

    
