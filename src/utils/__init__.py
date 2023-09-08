import pandas as pd
import numpy as np
#from src.config import mongo_client
import yaml
import dill
from src.logger import logging
from src.exception import CustomException
import os, sys
import pickle
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score



# def get_collection_as_dataframe(database_name:str,collection_name:str)->pd.DataFrame:
#     try:
#         logging.info(f"Reading data from database: {database_name} and collection: {collection_name}")
#         df = pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
#         logging.info(f"Found columns: {df.columns}")
#         #if "_id" in df.columns:
#             #logging.info(f"Dropping column: _id ")
#             #df = df.drop("_id",axis=1)
#         #logging.info(f"Row and columns in df: {df.shape}")
#         return df
#     except Exception as e:
#         raise CustomException(e, sys)



def save_obj(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
            
    except Exception as e:
        logging.info('Exception occured while saving an object')
        raise CustomException(e,sys)
    
def evaluate_model(X_train, y_train,X_test, y_test, models):
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]

            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)

            test_model_score = accuracy_score(y_test,y_test_pred )

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        logging.info('Exception occured while evaluating model')
        raise CustomException(e,sys)
    
def load_model(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        logging.info("Exception occured while loading a model")
        raise CustomException(e,sys)