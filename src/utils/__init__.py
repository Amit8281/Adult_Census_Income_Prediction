import pandas as pd
import numpy as np
import os,sys
from src.exception import CustomException
from src.config import mongo_client
from src.logger import logging
import yaml
import dill



def get_collection_as_dataframe(database_name:str,collection_name:str)->pd.DataFrame:
    try:
        logging.info(f"Reading data from database: {database_name} and collection: {collection_name}")
        df = pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        logging.info(f"Found columns: {df.columns}")
        #if "_id" in df.columns:
            #logging.info(f"Dropping column: _id ")
            #df = df.drop("_id",axis=1)
        #logging.info(f"Row and columns in df: {df.shape}")
        return df
    except Exception as e:
        raise CustomException(e, sys)