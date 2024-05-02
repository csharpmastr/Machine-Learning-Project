import sys
import os
from dataclasses import dataclass

# libraries for data cleaning and transformation
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# libraries for logging and exception handling
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

# create path for the data transformer
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


# create the data transformer
class DataTransformation:
    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_obj(self):
        """This function is responsible to clean and transform the data

        Raises:
            CustomException: Exception handling method to debug code easily

        Returns:
            preprocessor: The custom data precessor that will clean and transform
            the training data
        """
        try:
            # select all numerical / continuous columns
            num_cols = [
                'Urea', 
                'Cr', 
                'HbA1c', 
                'Chol', 
                'TG', 
                'HDL', 
                'LDL', 
                'VLDL', 
                'BMI', 
                'AGE'
                ]
            # select all categorical / discrete columns
            cat_cols = ['Gender', 'CLASS']
            
            # prepare pipeline for numerical columns
            num_pipeline = Pipeline(
                steps=[('scaler', StandardScaler(with_mean=False))]
            )
            
            # prepare pipeline for categorical columns
            cat_pipeline = Pipeline(
                steps=[
                    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))
                ]
            )
            
            # pass logging info
            logging.info("Scaling of Numerical Columns is completed")
            logging.info("Encoding of Categorical Columns is completed")
            
            # combine pipelines using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('numerical_pipeline', num_pipeline, num_cols),
                    ('categorical_pipeline', cat_pipeline, cat_cols)
                ]
            )
            
            # return the preprocessor data transformer
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    
    # initiate data transformation
    def initiate_data_transformation(self, train_path, test_path):
        
        try:
            # retrieve train and test data and store to pandas dataframe
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info('Train and Test data import completed')
            
            preprocessor_obj = self.get_data_transformer_obj()
            logging.info('Preprocessor object obtained')
            
            # apply preprocessor object
            input_feature_train_arr = preprocessor_obj.fit_transform(train_df)
            input_feature_test_arr = preprocessor_obj.transform(test_df) 
            
            logging.info(
                f"Preprocessor object has been applied on train and test data"
                )        
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )
            
            logging.info('Preprocessor object saved')   
            
            return (
                input_feature_train_arr,
                input_feature_test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)