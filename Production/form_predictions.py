import pandas as pd
import pickle
import xgboost as xgb

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

df = pd.read_csv("test-dataset-preprocessed.csv")

predictions = model.predict(df)

output = pd.DataFrame({"transaction_id": df["transaction_id"], "is_fraud": predictions})

output.to_csv("predictions.csv", index=False)