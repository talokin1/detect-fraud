import catboost as ctb
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

df = pd.read_csv(r'Models\CatBoost\preprocessed_data_mini.csv', low_memory=False, encoding='latin1')

X = df.drop(['is_fraud'], axis=1)
y = df['is_fraud']

print("Missing values in dataset:", X.isnull().sum().sum())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sampler = RandomUnderSampler(sampling_strategy=0.235)
X_train, y_train = sampler.fit_resample(X_train, y_train)

class_counts = Counter(y_train)
scale_pos_weight = class_counts[0] / class_counts[1]

cat_features = [0, 1, 3] + [i for i in range(5, 25)] + [i for i in range(27, 34)]

for col in cat_features:
    X_train.iloc[:, col] = X_train.iloc[:, col].astype(str)
    X_test.iloc[:, col] = X_test.iloc[:, col].astype(str)

catboost_model = ctb.CatBoostClassifier(
    iterations=1000,
    learning_rate=0.03,
    max_depth=3,
    bootstrap_type='Bernoulli',
    subsample=0.6,
    verbose=100,
    l2_leaf_reg=3,
    random_strength=1,
    model_size_reg=0.5,
    eval_metric='F1',
    od_type='Iter',
    early_stopping_rounds=100,
    grow_policy='Lossguide',
    loss_function='Logloss',
    class_weights=[0.545, 0.455],
    cat_features=cat_features,
    one_hot_max_size=10,
    counter_calc_method='Full'
)

xgboost_model = xgb.XGBClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.6,
    colsample_bytree=0.8,
    reg_lambda=3,
    reg_alpha=1,
    gamma=0.1,
    eval_metric='logloss',
    objective='binary:logistic',
    scale_pos_weight=scale_pos_weight,
    tree_method='hist',
    early_stopping_rounds=100,
    verbosity=1
)

lightgbm_model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.6,
    colsample_bytree=0.8,
    reg_lambda=3,
    reg_alpha=1,
    min_child_samples=10,
    boosting_type='gbdt',
    objective='binary',
    metric='binary_logloss',
    early_stopping_rounds=100
)

catboost_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
xgboost_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
lightgbm_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)

stacking_model = StackingClassifier(
    estimators=[
        ('catboost', catboost_model),
        ('xgboost', xgboost_model),
        ('lightgbm', lightgbm_model)
    ],
    final_estimator=ctb.CatBoostClassifier(iterations=500, learning_rate=0.03, depth=4, verbose=0, early_stopping_rounds=None),
    cv=5
)

stacking_model.fit(X_train, y_train)

y_pred = stacking_model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
