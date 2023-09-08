from src.constants import *
from src.config.configuration import *
import os, sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
import pickle
from src.utils import load_model
from sklearn.pipeline import Pipeline




class BatchPrediction:
    def __init__(self, input_file_path, model_file_path, transformer_file_path, feature_engineering_file_path):
        self.input_file_path = input_file_path
        self.model_file_path = model_file_path
        self.transformer_file_path = transformer_file_path
        self.feature_engineering_file_path = feature_engineering_file_path

    def start_batch_prediction(self):
        try:
            logging.info("Loading the saved pipeline")

            # Load the feature engineering pipeline
            with open(self.feature_engineering_file_path, 'rb') as f:
                feature_pipeline = pickle.load(f)

            logging.info(f"Feature Engineering Object Accessed: {self.feature_engineering_file_path}")

            # Load the data transformation pipeline
            with open(self.transformer_file_path, 'rb') as f:
                preprocessor = pickle.load(f)

            logging.info(f"Preprocessor Object Accessed: {self.transformer_file_path}")

            # Load the model separately
            model = load_model(file_path=self.model_file_path)

            logging.info(f"Model File Path: {self.model_file_path}")

            # Create the feature engineering pipeline
            feature_engineering_pipeline = Pipeline([
                ('feature_engineering', feature_pipeline)
            ])

            # Read the input file
            df = pd.read_csv(self.input_file_path)

            # Apply feature engineering
            df = feature_engineering_pipeline.transform(df)

            # Save the feature-engineered data as a CSV file
            feature_eng_path = os.path.join(FEATURE_ENG)
            os.makedirs(feature_eng_path, exist_ok=True)
            feature_eng_file_path = os.path.join(feature_eng_path, 'batch_feature_engineered.csv')
            df.to_csv(feature_eng_file_path, index=False)
            logging.info("Feature-engineered batch data saved as CSV.")

            # Dropping target column
            #df = df.drop('Item_Identifier',axis=1)

            logging.info(f"Columns before transformation: {', '.join(f'{col}: {df[col].dtype}' for col in df.columns)}")

            # Transform the feature-engineered data using the preprocessor
            transformed_data = preprocessor.transform(df)

            logging.info(f"Transformed Data Shape: {transformed_data.shape}")

            predictions = model.predict(transformed_data)

            # Create a DataFrame from the predictions array
            df_predictions = pd.DataFrame(predictions, columns=['prediction'])
            # Assuming 'prediction' column contains 0 or 1 values
            df_predictions['prediction'] = df_predictions['prediction'].map({0: '<=50k', 1: '>=50k'})


            # Save the predictions to a CSV file
            batch_prediction_path = os.path.join(BATCH_PREDICTION)
            os.makedirs(os.path.dirname(batch_prediction_path), exist_ok=True)
            df_predictions.to_csv(batch_prediction_path, index=False)
            logging.info(f"Batch predictions saved to '{batch_prediction_path}'.")

        except Exception as e:
            raise CustomException(e, sys)