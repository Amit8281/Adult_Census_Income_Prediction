import os 
import sys
import pandas as pd

from src.config.configuration import PREPROCESSING_OBJ_PATH,MODEL_FILE_PATH
from src.exception import CustomException
from src.logger import logging
from src.utils import load_model



class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            preprocessor_path = PREPROCESSING_OBJ_PATH
            model_path = MODEL_FILE_PATH
            
            preprocessor = load_model(preprocessor_path)
            model = load_model(model_path)
            
            data_scaled = preprocessor.transform(features)
            
            pred = model.predict(data_scaled)
            
            return pred 
        
        except Exception as e:
            logging.info("Error occured in prediction pipeline")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 Age:int,  
                 Education_num:int, 
                 Capital_gain:int, 
                 Hours_per_week:int,  
                 Marital_status:str,  
                 Relationship:str,
                 Race:str,
                 Sex:str,
                 Native_country:str,
                 New_occupation:str):
        
        self.Age = Age
        self.Education_num = Education_num
        self.Capital_gain = Capital_gain
        self.Hours_per_week = Hours_per_week
        self.Marital_status = Marital_status
        self.Relationship = Relationship
        self.Race = Race
        self.Sex=Sex
        self.Native_country=Native_country
        self.New_occupation=New_occupation
        

        
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Age':[self.Age],
                'Education_num':[self.Education_num],
                'Capital_gain':[self.Capital_gain],
                'Hours_per_week':[self.Hours_per_week],
                'Marital_status':[self.Marital_status],
                'Relationship':[self.Relationship],
                'Race':[self.Race],
                'Sex':[self.Sex],
                'Native_country':[self.Native_country],
                'New_occupation':[self.New_occupation]
                

            }
            
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("DataFrame gatherd")
            
            return df
        except Exception as e:
            logging.info("Exception occured in Custom data")
            raise CustomException(e,sys)