import pandas as pd
import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
import jolib

df = pd.read_csv(r'C:\Edu\detect-fraud\data\prepocessed_data-mini_v2.csv')
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

model = cb.CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=100,
    learning_rate=0.1,
    depth=6,
    subsample=0.8,
    colsample_bylevel=0.8,
    random_seed=42,
    auto_class_weights="Balanced"  
)

model.fit(X_train, y_train, verbose=10)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("AUC-ROC Score:", roc_auc_score(y_test, y_pred_proba))


model.save_model("C:\Edu\detect-fraud\models\catboost_model.cbm")
jolib.dump(model, "C:\Edu\detect-fraud\models\catboost_model.jolib")