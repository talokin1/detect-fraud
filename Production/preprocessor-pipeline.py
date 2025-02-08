import numpy as np
import pandas as pd
import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from preprocessing_tools import remove_missing_values, handle_outliers, drop_unnecessary_columns, process_datetime_features, balance_categorical_features, encode_categorical_features, replace_rare_categories


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PreprocessorPipeline:
    def __init__(self, num_features, cat_features):
        self.num_features = num_features
        self.cat_features = cat_features

        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        logging.info('Building preprocessing pipeline...')

        num_pipeline = Pipeline([
            ("remove_missing_values", remove_missing_values()),
            ("handle_outliers", handle_outliers()),
            ('scaling', MinMaxScaler())
        ])

        cat_pipeline = Pipeline([
            ('drop_unnecessary', drop_unnecessary_columns),
            ('process_datetime', process_datetime_features),
            ('replace_rare', replace_rare_categories),
            ('balance_categories', balance_categorical_features),
            ('encode', encode_categorical_features)
        ])
    

        preprocessor = ColumnTransformer([
            ("num", num_pipeline, self.num_features),
            ("cat", cat_pipeline, self.cat_features)
        ])

        full_pipeline = Pipeline([
            ("preprocessor", preprocessor),
        ])

        logging.info('Preprocessing pipeline built.')
        
        return full_pipeline

    def transformation(self, df):
        logging.info('Transforming data...')
        prepocessed_data = self.pipeline.fit_transform(df)
        logging.info('Data transformation completed.')

        return pd.DataFrame(prepocessed_data)
    

if __name__ == '__main__':
    logging.info('Loading data...')
    df = pd.read_csv('data/dataset-mini.csv')
    num_features = df.select_dtypes(include=['int64', 'uint64', 'float64']).columns.tolist()
    cat_features = df.select_dtypes(include=['object']).columns.tolist()

    preprocessor = PreprocessorPipeline(num_features=num_features, cat_features=cat_features)
    preprocessed_data = preprocessor.transformation(df)
    logging.info('Preprocessing completed.')

    logging.info('Saving preprocessed data...')
    pd.DataFrame(preprocessed_data).to_csv('data/preprocessed_data-mini.csv', index=False)


    
                                        

       

    