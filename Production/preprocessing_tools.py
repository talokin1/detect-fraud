import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

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

def replace_rare_categories(df: pd.DataFrame, columns: list, threshold_ratio: float = 0.0001) -> pd.DataFrame:
    length = len(df)

    for categ in columns:
        category_counts = df[categ].value_counts()
        threshold = threshold_ratio * length 
        df[categ] = df[categ].apply(lambda value: value if category_counts[value] > threshold else 'other')
    
    return df

def normalize_numerical(df):
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)
    return df

def encode_categorical(df):
    encoder = OneHotEncoder()
    df = encoder.fit_transform(df)
    return df

