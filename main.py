import pandas as pd
import numpy as np
import sys
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer



if __name__ == "__main__":
    try:
        #Data Ingestion
        obj = DataIngestion()
        train_data_path,test_data_path=obj.initiate_data_ingestion()
        #Data transformation
        data_transformation = DataTransformation()
        train_arr,test_arr,_ = data_transformation.initaite_data_transformation(train_data_path,test_data_path)
        # #Model Training
        model_trainer = ModelTrainer()
        print(model_trainer.initate_model_training(train_arr,test_arr))
        
    except Exception as e:
        raise CustomException(e,sys)