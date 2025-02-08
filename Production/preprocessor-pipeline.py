import numpy as np
import pandas as pd
import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from preprocessing_tools import remove_missing_values, handle_outliers, normalize_numerical, encode_categorical, feature_selection


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
            ("handle_outliers", handle_outliers()),
            ("normalize_numerical", normalize_numerical())
        ])

        cat_pipeline = Pipeline([
            ("remove_missing_values", remove_missing_values()),
            ("encode_categorical", encode_categorical())])
    

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
    df = pd.read_csv('data/dataset-mini.csv')
    num_features = [...]
    cat_features = [...]


    preprocessor = PreprocessorPipeline(num_features=num_features, cat_features=cat_features)
    preprocessed_data = preprocessor.transformation(df)
    logging.info('Preprocessing completed.')

    logging.info('Saving preprocessed data...')
    pd.DataFrame(preprocessed_data).to_csv('data/preprocessed_data-mini.csv', index=False)


    
                                        

       

    