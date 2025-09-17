# STAGE 05: MODEL TRAINING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR

import text_preprocess


def splittng_data_and_model_training(seed=42):
    TFIDFvector, df_final = text_preprocess.text_preprocessing()
    X = df_final.drop(columns=["score"])
    y = df_final["score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)

    # USER SELECT THEIR CHOICE MODEL
    print("Select a model of choice: ")
    print("1 - LINEAR REGRESSION")
    print("2 - RIDGE REGRESSION")
    print("3 - LASSO REGRESSION")
    print("4 - RANDOM FOREST REGRESSOR")
    print("5 - DECISION TREE REGRESSOR")
    print("6 - SUPPORT VECTOR REGRESSOR")
    print("7 - HIST GRADIENT REGRESSOR")
    print("8 - GRADIENT BOOSTING REGRESSOR")

    CHOICE = input("Select the number of the model you want to use: ")

    if CHOICE == "1":
        model = LinearRegression()
    elif CHOICE ==  "2":
        model = Ridge()
    elif CHOICE == "3":
        model = Lasso()
    elif CHOICE == "4":
        model = RandomForestRegressor(random_state=seed)
    elif CHOICE == "5":
        model = DecisionTreeRegressor(random_state=seed)
    elif CHOICE == "6":
        model = SVR()
    elif CHOICE == "7":
        model = HistGradientBoostingRegressor(random_state=seed)
    elif CHOICE == "8":
        model = GradientBoostingRegressor(random_state=seed)
    else:
        print("Invalid Selection. Training with the default Linear Regaression.")

    model.fit(X_train, y_train)

    y_pred = np.round(model.predict(X_test), 0)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    dict = {
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "R_SQUARED": f"{round(r2*100, 2)}%"
    }

    print(f"METRICS: \n{pd.DataFrame([dict])}")

    # print(f"Mean Absolute Error (MAE): {round(mae, 2)}")
    # print(f"Mean Squared Error (MSE): {round(mse, 2)}")
    # print(f"Root Mean Squared Error (RMSE): {round(rmse, 2)}")
    # print(f"R-squared (R2 ): {round(r2*100, 2)}")

    # SAVING THE VECTORIZER AND MODEL INTO THE MODEL FOLDER
    path = os.path.join(os.getcwd(), "model")
    os.makedirs(path, exist_ok=True)

    vectorizer_path = os.path.join(path, "vectorizer.pkl")
    model_path = os.path.join(path, "trained_model.pkl")

    joblib.dump(TFIDFvector, vectorizer_path)
    joblib.dump(model, model_path)

    return TFIDFvector, model


if __name__ == "__main__":
    preprossor_vectorizer, trained_model = splittng_data_and_model_training(seed=42)