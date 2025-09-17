# STAGE 06: PREDICTION PIPELINE

import numpy as np
import pandas as pd
import text_toks_lem_stw
import joblib
import os


############################################################################################################################################

loaded_vectorizer_path = os.path.join(os.getcwd(), "model", "vectorizer.pkl")
loaded_trained_model_path = os.path.join(os.getcwd(), "model", "trained_model.pkl")

loaded_vectorizer = joblib.load(loaded_vectorizer_path)
loaded_trained_model = joblib.load(loaded_trained_model_path)

def inference_pipeline():

    new_text = input("Enter your review below: \n").split("\n")

    user_text_df = pd.DataFrame(new_text, columns=["user_input"])

    user_text_df["user_input"] = user_text_df["user_input"].astype(str).str.lower()
    user_text_df["user_input"] = user_text_df["user_input"].str.replace(r"[^a-z\s]", "", regex=True)
    user_text_df["user_input"] = user_text_df["user_input"].str.replace(r"\s+", " ", regex=True).str.strip()

    user_text_df["user_input"] = user_text_df["user_input"].astype(str).apply(text_toks_lem_stw.token_stopwords_lemmatization_fns)

    dtm = loaded_vectorizer.transform(user_text_df["user_input"])

    df_array = dtm.toarray()

    user_df_final = pd.DataFrame(df_array, columns=loaded_vectorizer.get_feature_names_out())

    prediction = np.round(loaded_trained_model.predict(user_df_final), 0)
    res_dict = {"review_text": new_text, "review_rating": prediction}

    return print(f"Your rating for your review is: \n{pd.DataFrame(res_dict)}")


if __name__ == "__main__":
    inference_pipeline()