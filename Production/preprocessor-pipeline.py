import numpy as np
import pandas as pd
import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from preprocessing_tools import RemoveMissingValues, HandleOutliers, ReplaceRareCategories, ProcessDatetimeFeatures, EncodeCategoricalFeatures, NormalizeNumerical


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PreprocessorPipeline:
    def __init__(self, num_features, cat_features):
        self.num_features = num_features
        self.cat_features = cat_features

        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        logging.info('Building preprocessing pipeline...')

        num_pipeline = Pipeline([
            ("remove_missing", RemoveMissingValues(self.num_features)),
            ("handle_outliers", HandleOutliers(self.num_features)),
            ('scaling', NormalizeNumerical())
        ])

        cat_pipeline = Pipeline([
            ("process_datetime", ProcessDatetimeFeatures()),
            ("replace_rare", ReplaceRareCategories()),
            ("encode", EncodeCategoricalFeatures())
        ])
    
        preprocessor = ColumnTransformer([
            ("num", num_pipeline, self.num_features),
            ("cat", cat_pipeline, self.cat_features)
        ])

        logging.info('Preprocessing pipeline built.')
        
        return Pipeline([("preprocessor", preprocessor)])
    
    def transformation(self, df):
        logging.info('Transforming data...')
        return pd.DataFrame(self.pipeline.fit_transform(df))
    

if __name__ == '__main__':
    logging.info('Loading data...')
    df = pd.read_csv('C:\Edu\detect-fraud(draft)\data\general_datasets\dataset-mini.csv')
    num_features = df.select_dtypes(include=['int64', 'uint64', 'float64']).columns.tolist()
    cat_features = df.select_dtypes(include=['object']).columns.tolist()

    preprocessor = PreprocessorPipeline(num_features=num_features, cat_features=cat_features)
    preprocessed_data = preprocessor.transformation(df)
    logging.info('Preprocessing completed.')

    logging.info('Saving preprocessed data...')
    pd.DataFrame(preprocessed_data).to_csv('data/preprocessed_data-mini.csv', index=False)


    
                                        

       

    