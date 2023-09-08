import pymongo
import pandas as pd
import os, sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.constants import *
from src.config.configuration import *

# Define your MongoDB connection details
MONGO_DB_URL = "mongodb+srv://Amit8281:Amit8281@cluster0.jwsyedo.mongodb.net/?retryWrites=true&w=majority"
DATABASE_NAME = "ADULT"
COLLECTION_NAME = "ADULT_PROJECT"



@dataclass
class DataIngestionconfig:
    train_data_path: str = TRAIN_FILE_PATH
    test_data_path: str = TEST_FILE_PATH
    raw_data_path: str = RAW_FILE_PATH

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info("=" * 50)
        logging.info("Initiate Data Ingestion config")
        logging.info("=" * 50)
        
        try:
            # Connect to MongoDB
            client = pymongo.MongoClient(MONGO_DB_URL)
            db = client[DATABASE_NAME]
            collection = db[COLLECTION_NAME]
            logging.info("Making Connection with DataBase")
            # Query data from MongoDB collection
            data_from_mongodb = list(collection.find())

            if not data_from_mongodb:
                raise CustomException("No data found in MongoDB collection")

            logging.info("Convert data to pandas DataFrame")
            df = pd.DataFrame(data_from_mongodb)

            logging.info("deleting _id column cause it is unnecessary")
            df.drop('_id',axis=1,inplace=True)

            logging.info(f"Downloaded data from MongoDB collection {COLLECTION_NAME}")
            logging.info('Dataset read as pandas DataFrame')

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False)   

            logging.info("Train-test split")
            train_set, test_set = train_test_split(df, test_size=0.20, random_state=42)

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)

            logging.info(f"Train data saved at {self.data_ingestion_config.train_data_path}")

            os.makedirs(os.path.dirname(self.data_ingestion_config.test_data_path), exist_ok=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)

            logging.info(f"Test data saved at {self.data_ingestion_config.test_data_path}")

            logging.info("Data ingestion complete")

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
  
        except Exception as e:
            logging.error('Exception occurred at Data Ingestion stage')
            raise CustomException(e, sys)


