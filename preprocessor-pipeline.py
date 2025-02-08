import numpy as np
import pandas as pd
import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from preprocessing-tools import remove_missing_values, handle_outliers, normalize_numerical, encode_categorical, feature_selection


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class PreprocessorPipeline:
    def __init__(self, num_features, cat_features):
        self.num_features = num_features
        self.cat_features = cat_features
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        logging.info('Building preprocessing pipeline...')

        num_pipeline = Pipeline([
            ("remove_missing_values", remove_missing_values()),
            ("handle_outliers", handle_outliers(method='iqr')),
            ("normalize_numerical", normalize_numerical(scaler=StandardScaler()))
        ])

        cat_pipeline = Pipeline([
            ("remove_missing_values", remove_missing_values()),
            ("encode_categorical", encode_categorical(encoder=OneHotEncoder()))])

        preprocessor = ColumnTransformer([
            ("num", num_pipeline, self.num_features),
            ("cat", cat_pipeline, self.cat_features)
        ])

        full_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("feature_selection", feature_selection())
        ])

    def transformation(self, df):
        logging.info('Transforming data...')
        prepocessed_data =  self.pipeline.fit_transform(df)
        logging.info('Data transformation completed.')

        return prepocessed_data
    

if __name__ == '__main__':
    logging.info('Loading data...')
    df = pd.read_csv('data/train.csv')
    num_features = [...]
    catf_eatures = [...]


    preprocessor = PreprocessorPipeline(num_features=num_features, cat_features=catfeatures
    
                                        

       

    