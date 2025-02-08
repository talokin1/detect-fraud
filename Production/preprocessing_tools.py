import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.preprocessing import LabelEncoder

def remove_missing_values(df: pd.DataFrame, missing_theshold=0.22):
    df_cleaned = df.dropna(axis=1, thresh=missing_theshold*len(df))
    df_cleaned = df_cleaned.drop(columns=["user_agent", "traffic_source", "card_holder_first_name", "card_holder_last_name"], errors="ignore")
    
    categorical_features = df_cleaned.select_dtypes(include=["object"]).columns
    df_cleaned[categorical_features] = df_cleaned[categorical_features].fillna("Missing")

    numerical_features = df_cleaned.select_dtypes(include=["int64", "uint64", "float64"]).columns
    df_cleaned = df_cleaned.dropna(subset=numerical_features)
    
    return df_cleaned

def handle_outliers(df, factor=1.5):

    conditions = []
    total_outliers = 0
    for col in df.select_dtypes(include=['number']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        num_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        total_outliers += num_outliers

        print(f"{col}: {num_outliers} outliers removed")

        conditions.append((df[col] >= lower_bound) & (df[col] <= upper_bound))

    combined_condition = conditions[0]
    for cond in conditions[1:]:
        combined_condition &= cond

    return df.loc[combined_condition]  

def replace_rare_categories(df: pd.DataFrame, threshold_ratio: float = 0.0001) -> pd.DataFrame:
    length = len(df)
    unbalanced_categories = ['merchant_country', 'merchant_language', 'ip_country', 'platform', 'cardbrand', 'cardcountry']
    
    for categ in unbalanced_categories:
        category_counts = df[categ].value_counts()
        threshold = threshold_ratio * length 
        df[categ] = df[categ].apply(lambda value: value if category_counts[value] > threshold else 'other')
    
    return df

# Categorical features preprocessing

def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Видаляє зайві колонки."""
    df = df.copy()
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
    return df

def process_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df['month_creating'] = df['created_at'].dt.month
        df['week_day_creating'] = df['created_at'].dt.dayofweek
        df.drop('created_at', axis=1, inplace=True)
    return df

def balance_categorical_features(df: pd.DataFrame, threshold_ratio: float = 0.0001) -> pd.DataFrame:
    unbalanced_categories = ['merchant_country', 'merchant_language', 'ip_country', 'platform', 'cardbrand', 'cardcountry']
    length = len(df)
    threshold = threshold_ratio * length
    for categ in unbalanced_categories:
        if categ in df.columns:
            category_counts = df[categ].value_counts()
            df[categ] = df[categ].apply(lambda value: value if category_counts.get(value, 0) > threshold else 'other')
    return df

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    categ_features = df.select_dtypes(include=['object']).columns.to_list()
    for feat in categ_features:
        encoder = LabelEncoder()
        df[feat] = encoder.fit_transform(df[feat].astype(str))
    return df

def preprocess_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_unnecessary_columns(df)
    df = process_datetime_features(df)
    df = balance_categorical_features(df)
    df = encode_categorical_features(df)
    return df

    

def normalize_numerical(df):
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)
    return df

