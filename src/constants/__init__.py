import os
import shutil
from datetime import datetime
import pandas as pd

# Function to get the current timestamp
def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"

CURRENT_TIME_STAMP = get_current_time_stamp()

# Root Directory
ROOT_DIR_KEY = os.getcwd()

# Data file path
DATA_DIR = "Data"
DATA_DIR_KEY = 'Train.csv'  # Update with your train data file name
#TEST_DATA_KEY = 'Test.csv'    # Update with your test data file name

# Artifact
ARTIFACT_DIR_KEY = "Artifact"

# Data Ingestion constants
DATA_INGESTION_KEY = 'data_ingestion'
DATA_INGESTION_RAW_DATA_DIR_KEY = 'raw_data_dir'
DATA_INGESTION_INGESTED_DIR_NAME_KEY = 'ingested_dir'
RAW_DATA_DIR_KEY = 'raw.csv'
TRAIN_DATA_DIR_KEY = 'train.csv'
TEST_DATA_DIR_KEY = 'test.csv'

#Data Transformation Constants
DATA_TRANSFORMATION_ARTIFACT = 'data_transformation'
DATA_PREPROCESSED_DIR='preprocessed'
DATA_TRANSFORMATION_PREPROCESSING_OBJ = 'preprocessor.pkl'
DATA_TRANSFORMED_DIR = 'transformed_data'
TRANSFORMED_TRAIN_DIR_KEY = 'train.csv'
TRANSFORMED_TEST_DIR_KEY = 'test.csv'

# Model Training Constant
MODEL_TRAINER_KEY = 'model_trainer'
MODEL_OBJECT = 'model.pkl'

#Batch Prediction
PREDICTION_FOLDER='batch_Prediction'
PREDICTION_CSV='prediction_csv'
PREDICTION_FILE='prediction.csv'
FEATURE_ENG_FOLDER='feature_eng'
