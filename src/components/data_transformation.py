from src.constants import *
from src.logger import logging
from src.exception import CustomException
import os, sys
from src.config.configuration import *
from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder,OneHotEncoder,MinMaxScaler
from sklearn.pipeline import Pipeline
from src.utils import save_obj
from src.config.configuration import PREPROCESSING_OBJ_PATH,TRANSFORMED_TRAIN_FILE_PATH,TRANSFORMED_TEST_FILE_PATH,FEATURE_ENG_OBJ_PATH
# FE
# Data transformation


class Feature_Engineering(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        
        """
        This class applies necessary Feature Engneering 
        """
        logging.info(f"\n{'*'*20} Feature Engneering Started {'*'*20}\n\n")




    def transform_data(self,df):
        try:
            df.rename(columns={'age':'Age','workclass':'Workclass','fnlwgt':'Final_weight','education':'Education','education.num':'Education_num','marital.status':'Marital_status','occupation':'Occupation','relationship':'Relationship','race':'Race','sex':'Sex','capital.gain':'Capital_gain','capital.loss':'Capital_loss',
                               'hours.per.week':'Hours_per_week','native.country':'Native_country','income':'Income'}, inplace=True)

            logging.info("Replace income value more than 50k with 1 and less than 50k with 0.")
            df['Income'] =df['Income'].str.strip()
            df['Income'].replace({'<=50K': 0, '>50K': 1}, inplace=True)

            logging.info('Dropping Duplocate values')
            df.drop_duplicates(inplace=True)

            logging.info("replace charactors with nan.")
            replace_chars = ["\n", "\n?\n", "?","\n?"," ?","? "," ? "," ?\n"]
            if any(char in df.values for char in replace_chars):
                df.replace(replace_chars, np.nan, inplace=True)
                logging.info("Successfully replaced characters with NaN.")

            logging.info("Imputating Missing value with mode for categorical features")
            df['Occupation'].fillna(df['Occupation'].mode()[0],inplace=True)
            df['Workclass'].fillna(df['Workclass'].mode()[0],inplace=True)
            df['Native_country'].fillna(df['Native_country'].mode()[0],inplace=True)
            logging.info("filled null values with mode")

            logging.info('Clean up occupation column by removing any leading/trailing spaces')
            df['Occupation'] = df['Occupation'].str.strip()
            logging.info("Replace occupation categories with new categories")
            df['New_occupation'] = df['Occupation'].replace({
                'Prof-specialty': 'Professional_Managerial',
                'Craft-repair': 'Skilled_Technical',
                'Exec-managerial': 'Professional_Managerial',
                'Adm-clerical': 'Sales_Administrative',
                'Sales': 'Sales_Administrative',
                'Other-service': 'Service_Care',
                'Machine-op-inspct': 'Skilled_Technical',
                'Missing': 'Unclassified Occupations',
                'Transport-moving': 'Skilled_Technical',
                'Handlers-cleaners': 'Service_Care',
                'Farming-fishing': 'Service_Care',
                'Tech-support': 'Skilled_Technical',
                'Protective-serv': 'Professional_Managerial',
                'Priv-house-serv': 'Service_Care',
                'Armed-Forces': 'Unclassified Occupations',
            })

            df.drop(['Occupation'], axis=1,inplace=True)
            logging.info(f"New narrowed categories : \n{df['New_occupation'].value_counts()}")



            logging.info("deleting 'Education','Final_weight','Workclass','Capital_loss' because there not much usefull ")
            df.drop(['Education','Final_weight','Workclass','Capital_loss'],axis=1,inplace=True)
            
            logging.info("now we are good to go for Data Transformation")

            logging.info(f'Train Dataframe Head: \n{df.head().to_string()}')

            return df
     



        except Exception as e:
            logging.info(" error in transforming data")
            raise CustomException(e, sys) from e 
        


    def fit(self,X,y=None):
        return self
    
    
    def transform(self,X:pd.DataFrame,y=None):
        try:    
            transformed_df=self.transform_data(X)
                
            return transformed_df
        except Exception as e:
            raise CustomException(e,sys) from e



@dataclass
class DataTransformationConfig(): # first modofication
    preprocessor_obj_file_path=PREPROCESSING_OBJ_PATH
    transformed_train_path=TRANSFORMED_TRAIN_FILE_PATH
    transformed_test_path=TRANSFORMED_TEST_FILE_PATH
    feature_eng_obj_path=FEATURE_ENG_OBJ_PATH


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')


            logging.info('defining the ordinal data ranking')
            
            #Workclass = ['Private','Self-emp-not-inc','Local-gov','State-gov','Self-emp-inc','Federal-gov','Without-pay','Never-worked']
            Marital_status = ['Married-civ-spouse','Never-married','Divorced','Separated','Widowed','Married-spouse-absent','Married-AF-spouse']
            Relationship = ['Husband','Not-in-family','Own-child','Unmarried','Wife','Other-relative']
            Race = ['White','Black','Asian-Pac-Islander','Amer-Indian-Eskimo','Other']
            Sex = ['Male','Female']
            Native_country  = [
                                "United-States", "Mexico", "Philippines", "Germany", "Canada", "Puerto-Rico", 
                                "El-Salvador", "India", "Cuba", "England", "Jamaica", "South", "China", "Italy", 
                                "Dominican-Republic", "Vietnam", "Guatemala", "Japan", "Poland", "Columbia", "Taiwan", 
                                "Haiti", "Iran", "Portugal", "Nicaragua", "Peru", "Greece", "France", "Ecuador", 
                                "Ireland", "Hong", "Cambodia", "Trinadad&Tobago", "Laos", "Thailand", "Yugoslavia", 
                                "Outlying-US(Guam-USVI-etc)", "Hungary", "Honduras", "Scotland", "Holand-Netherlands"
                            ]
            New_occupation = ['Professional_Managerial','Skilled_Technical','Sales_Administrative','Service_Care','Unclassified Occupations']
            

            # defining the categorical and numerical column
            #categorical_column=['']
            ordinal_encod=['Marital_status','Relationship','Race','Sex','Native_country','New_occupation']
            numerical_column=['Age','Education_num','Capital_gain','Hours_per_week']
            
            #item_type = ['']
            # numerical pipeline

            numerical_pipeline=Pipeline(steps=[
                ('impute',SimpleImputer(strategy='constant',fill_value=0)),
                ('scaler',MinMaxScaler())
                ])



            # categorical pipeline

            # categorical_pipeline=Pipeline(steps=[
            #     ('impute',SimpleImputer(strategy='most_frequent')),
            #     ('onehot',OneHotEncoder(handle_unknown='ignore')),
            #     ('scaler',StandardScaler(with_mean=False))
            #     ])




            # ordinal pipeline
            ordianl_pipeline=Pipeline(steps=[
                ('impute',SimpleImputer(strategy='most_frequent')),
                ('ordinal', OrdinalEncoder(categories=[ Marital_status, Relationship, Race,Sex,Native_country,New_occupation], handle_unknown='use_encoded_value', unknown_value=-1)),
                ('scaler',MinMaxScaler())   
                ])
            
            # preprocessor

            preprocessor =ColumnTransformer([
                ('numerical_pipeline',numerical_pipeline,numerical_column),
                #('categorical_pipeline',categorical_pipeline,categorical_column),
                ('ordianl_pipeline',ordianl_pipeline,ordinal_encod)
                ],remainder='passthrough')


            # returning preprocessor

            return preprocessor
        
            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("error in data transformation")
            raise CustomException(e,sys)
    

    def get_feature_engineering_object(self):
        try:
            
            feature_engineering = Pipeline(steps = [("fe",Feature_Engineering())]) # 2nd modification
            return feature_engineering
        except Exception as e:
            raise CustomException(e,sys) from e


    def initaite_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            #logging.info('Read train and test data completed')
            #logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            #logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')



            logging.info('Obtaining preprocessing object')



            logging.info(f"Obtaining feature engineering object.")
            fe_obj = self.get_feature_engineering_object()

            logging.info(f"Applying feature engineering object on training dataframe and testing dataframe")
            logging.info(">>>" * 20 + " Training data " + "<<<" * 20)
            logging.info(f"Feature Enineering - Train Data ")

            train_df = fe_obj.fit_transform(train_df)
            logging.info(">>>" * 20 + " Test data " + "<<<" * 20)
            logging.info(f"Feature Enineering - Test Data ")
            test_df = fe_obj.transform(test_df)

            train_df.to_csv("train_data.csv")
            test_df.to_csv("test_data.csv")
            logging.info(f"Saving csv to train_data and test_data.csv")

# preprocessing object

            preprocessing_obj = self.get_data_transformation_object()


            

# target column

            target_column_name = 'Income'
            #drop_columns = [target_column_name,'id']

            X_train = train_df.drop(columns=target_column_name,axis=1)
            y_train=train_df[target_column_name]

            X_test=test_df.drop(columns=target_column_name,axis=1)
            y_test=test_df[target_column_name]

            logging.info(f"shape of {X_train.shape} and {X_test.shape}")
            logging.info(f"shape of {y_train.shape} and {y_test.shape}")

            # Transforming using preprocessor obj
            
            X_train=preprocessing_obj.fit_transform(X_train)            
            X_test=preprocessing_obj.transform(X_test)
            logging.info("Applying preprocessing object on training and testing datasets.")
            logging.info(f"shape of {X_train.shape} and {X_test.shape}")
            logging.info(f"shape of {y_train.shape} and {y_test.shape}")
            

            logging.info("transformation completed")
            


            train_arr = np.c_[X_train, np.array(y_train)]
            test_arr = np.c_[X_test, np.array(y_test)]

            logging.info("train_arr, test_arr completed")

            

            logging.info("train arr , test arr")


            df_train= pd.DataFrame(train_arr)
            df_test = pd.DataFrame(test_arr)

            logging.info("converting train_arr and test_arr to dataframe")
            #logging.info(f"Final Train Transformed Dataframe Head:\n{df_train.head().to_string()}")
            #logging.info(f"Final Test transformed Dataframe Head:\n{df_test.head().to_string()}")

            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_path),exist_ok=True)
            df_train.to_csv(self.data_transformation_config.transformed_train_path,index=False,header=True)

            logging.info("transformed_train_path")
            logging.info(f"transformed dataset columns : {df_train.columns}")

            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_test_path),exist_ok=True)
            df_test.to_csv(self.data_transformation_config.transformed_test_path,index=False,header=True)


            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj)
            
            logging.info("Preprocessor file saved")


            save_obj(
                file_path=self.data_transformation_config.feature_eng_obj_path,
                obj=fe_obj)
            logging.info("Feature eng file saved")

            
            return(train_arr,
                   test_arr,
                   self.data_transformation_config.preprocessor_obj_file_path)



        except Exception as e:
            logging.info("error in data transformation")
            raise CustomException(e,sys)