import catboost as ctb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
import pickle

df = pd.read_csv(r'Models\CatBoost\preprocessed_data_mini.csv', low_memory = False, encoding = 'latin1')
 
X = df.drop(['is_fraud'], axis=1)
y = df['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sampler = RandomUnderSampler(sampling_strategy = 0.235)
X_train, y_train = sampler.fit_resample(X_train, y_train)

model = ctb.CatBoostClassifier(
    iterations = 1000,
    learning_rate = 0.03,
    max_depth = 3,
    bootstrap_type = 'Bernoulli',
    subsample = 0.6,
    verbose = 100,
    l2_leaf_reg = 3,
    random_strength = 1,
    model_size_reg = 0.5,
    eval_metric = 'F1',
    od_type = 'Iter',
    early_stopping_rounds = 100,
    grow_policy = 'Lossguide',
    loss_function = 'Logloss',
    class_weights = [0.5, 0.5],
    cat_features = [0, 1, 3] + [i for i in range(5, 25)] + [i for i in range(27, 34)],
    one_hot_max_size = 10,
    counter_calc_method = 'Full',
    nan_mode = 'Min'
)  

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

with open('mods\catboost.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("AUC-ROC Score:", roc_auc_score(y_test, y_pred_proba))
