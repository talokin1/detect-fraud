import numpy as np
import pandas as pd
import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.under_sampling import RandomUnderSampler
from preprocessing_tools import (
    RemoveMissingValues, HandleOutliers, ReplaceRareCategories,
    ProcessDatetimeFeatures, EncodeCategoricalFeatures, NormalizeNumerical
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PreprocessorPipeline:
    def __init__(self, num_features, cat_features, undersample=True, chunk_size=1_000_000):
        self.num_features = num_features
        self.cat_features = cat_features
        self.undersample = undersample
        self.chunk_size = chunk_size
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        logging.info('Building preprocessing pipeline...')

        num_pipeline = Pipeline([ 
            ("remove_missing", RemoveMissingValues()),
            ("handle_outliers", HandleOutliers(factor=1.5)),
            ('scaling', NormalizeNumerical())
        ])

        cat_pipeline = Pipeline([ 
            ("replace_rare", ReplaceRareCategories()),
            ("process_datetime", ProcessDatetimeFeatures()),
            ("encode", EncodeCategoricalFeatures())
        ])

        preprocessor = ColumnTransformer([ 
            ("num", num_pipeline, self.num_features),
            ("cat", cat_pipeline, self.cat_features)
        ])

        logging.info('Preprocessing pipeline built.')
        return Pipeline([("preprocessor", preprocessor)])

    def transformation(self, df, target):
        logging.info('Transforming data...')
        processed_chunks = []

        num_chunks = len(df) // self.chunk_size + (1 if len(df) % self.chunk_size else 0)
        for i in range(num_chunks):
            chunk = df.iloc[i * self.chunk_size : (i + 1) * self.chunk_size]
            logging.info(f'Processing chunk {i + 1} of {num_chunks}...')
            
            chunk_transformed = pd.DataFrame(self.pipeline.fit_transform(chunk))
            chunk_transformed[target.name] = target.iloc[i * self.chunk_size : (i + 1) * self.chunk_size].values
            
            processed_chunks.append(chunk_transformed)

        transformed_data = pd.concat(processed_chunks, axis=0)

        if self.undersample:
            logging.info('Applying undersampling...')
            rus = RandomUnderSampler()
            transformed_data, target = rus.fit_resample(transformed_data.drop(columns=[target.name]), transformed_data[target.name])
            transformed_data[target.name] = target
            logging.info('Undersampling completed.')

        return transformed_data.reset_index(drop=True)

if __name__ == '__main__':
    logging.info('Loading data...')
    df = pd.read_csv(r"...\data\general_datasets\dataset-mini.csv")

    df.pop("Unnamed: 0")
    is_fraud = df.pop("is_fraud")

    num_features = df.select_dtypes(include=['int64', 'uint64', 'float64']).columns.tolist()
    cat_features = df.select_dtypes(include=['object']).columns.tolist()

    preprocessor = PreprocessorPipeline(num_features=num_features, cat_features=cat_features, undersample=True, chunk_size=1_000_000)
    preprocessed_data = preprocessor.transformation(df, is_fraud)
    
    logging.info('Preprocessing completed.')

    logging.info('Saving preprocessed data...')
    preprocessed_data.to_csv(r'..\data\prepocessed_data-mini_v3.csv', index=False)
