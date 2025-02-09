import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RemoveMissingValues(BaseEstimator, TransformerMixin):
    def __init__(self, missing_threshold=0.22):
        self.missing_threshold = missing_threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logging.info('Removing missing values...')
        X = X.dropna(axis=1, thresh=self.missing_threshold * len(X))
        drop_cols = ["user_agent", "traffic_source", "card_holder_first_name", "card_holder_last_name"]
        X.drop(columns=[col for col in drop_cols if col in X.columns], errors="ignore", inplace=True)

        cat_features = X.select_dtypes(include=["object"]).columns
        X[cat_features] = X[cat_features].fillna("Missing")

        num_features = X.select_dtypes(include=["int64", "uint64", "float64"]).columns
        X.dropna(subset=num_features, inplace=True)

        return X


class HandleOutliers(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logging.info('Handling outliers...')
        num_cols = X.select_dtypes(include=["number"]).columns

        for col in num_cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.factor * IQR
            upper_bound = Q3 + self.factor * IQR
            X[col] = np.clip(X[col], lower_bound, upper_bound)

        return X


class ReplaceRareCategories(BaseEstimator, TransformerMixin):
    def __init__(self, threshold_ratio=0.0001):
        self.threshold_ratio = threshold_ratio

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logging.info('Replacing rare categories...')
        unbalanced_categories = ['merchant_country', 'merchant_language', 'ip_country', 'platform', 'cardbrand', 'cardcountry']
        length = len(X)

        for categ in unbalanced_categories:
            if categ in X.columns:
                category_counts = X[categ].value_counts()
                threshold = self.threshold_ratio * length
                X[categ] = X[categ].apply(lambda value: value if category_counts.get(value, 0) > threshold else 'other')

        return X


class ProcessDatetimeFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logging.info('Processing datetime features...')
        if 'created_at' in X.columns:
            X['created_at'] = pd.to_datetime(X['created_at'], errors='coerce')
            X['month_creating'] = X['created_at'].dt.month
            X['week_day_creating'] = X['created_at'].dt.dayofweek
            X.drop('created_at', axis=1, inplace=True)
        return X


class EncodeCategoricalFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.encoders = {}
        for col in X.select_dtypes(include=["object"]).columns:
            self.encoders[col] = LabelEncoder()
            self.encoders[col].fit(X[col].astype(str))
        return self

    def transform(self, X):
        logging.info('Encoding categorical features...')
        for col, encoder in self.encoders.items():
            X[col] = encoder.transform(X[col].astype(str))
        return X



class NormalizeNumerical(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.scaler = MinMaxScaler()
        self.num_features = X.select_dtypes(include=["number"]).columns.tolist()
        self.scaler.fit(X[self.num_features])
        return self

    def transform(self, X):
        logging.info('Normalizing numerical features...')
        X[self.num_features] = self.scaler.transform(X[self.num_features])
        return X
