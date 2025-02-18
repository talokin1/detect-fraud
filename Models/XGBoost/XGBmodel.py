import pandas as pd
import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import numpy as np
import pickle


df = pd.read_csv(r'data\preprocessed_data.csv')
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)


model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=200,  
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)


model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("AUC-ROC Score:", roc_auc_score(y_test, y_pred_proba))
print("F1-score:", f1_score(y_test, y_pred))

model.save_model(r"C:\Users\Klucly\Desktop\competition\mods\\xgboost.cbm")
with open(r"C:\Users\Klucly\Desktop\competition\mods\xgboost.pkl", "wb") as f:
    pickle.dump(model, f)
