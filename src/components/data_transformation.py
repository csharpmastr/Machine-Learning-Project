import sys
import os
import numpy as np
from dataclasses import dataclass

# libraries for data cleaning and transformation
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek, SMOTEENN

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
        """Function responsible in building the pipelines to preprocess and 
        transform the data

        Raises:
            CustomException: For exception handling purposes

        Returns:
            preprocessor: The custom data preprocessor that will clean and transform
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
            cat_cols = ['Gender_F','Gender_M']
            
            # prepare pipeline for numerical columns
            num_pipeline = Pipeline(
                steps=[('scaler', StandardScaler(with_mean=False))]
            )
            
            # pass logging info
            logging.info("Scaling of Numerical Columns is completed")
            logging.info("Encoding of Categorical Columns is completed")
            
            # combine pipelines using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('numerical_pipeline', num_pipeline, num_cols),
                    ('pass_through', 'passthrough', cat_cols)
                ]
            )
            
            # return the preprocessor data transformer
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    
    # initiate data transformation
    def initiate_data_transformation(self, train_path, test_path):
        """Function to initiate the data transformation and processing by 
        loading the preprocessor object

        Args:
            train_path (obj): Raw training data obtained from file path
            test_path (obj): Raw testing data obtained from file path

        Raises:
            CustomException: For exception handling purposes

        Returns:
            list: A list of object, training data of type array,
                  test data of type array, and the preprocessor
                  or type object.
        """
        
        try:
            # retrieve train and test data and store to pandas dataframe
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info('Train and Test data import completed')
            
            # apply oversampling to train_df
            x_features = train_df.iloc[:, 0:12]
            y_target = train_df.iloc[:, 12:]
            
            smote = SMOTETomek(random_state=42)
            x_resampled, y_resampled = smote.fit_resample(x_features.to_numpy(), y_target.to_numpy())
            
            x_data = pd.DataFrame(x_resampled, columns=x_features.columns)
            y_data = pd.DataFrame(y_resampled, columns=y_target.columns)
            train_df = pd.concat([x_data, y_data], axis=1)
            
            preprocessor_obj = self.get_data_transformer_obj()
            logging.info('Preprocessor object obtained')
            
            print("index 1 of train_df: ", train_df.iloc[0:1, :])
            print("Shape of the train_df: ", train_df.shape)
            
            # get target variable from train and test data
            train_target = train_df.iloc[:, -3:]
            test_target = test_df.iloc[:, -3:]
            
            # apply preprocessor object
            input_feature_train_arr = preprocessor_obj.fit_transform(train_df)
            input_feature_test_arr = preprocessor_obj.transform(test_df) 
            
            # concat target and pre-processed data
            input_feature_train_arr = np.c_[input_feature_train_arr, np.array(train_target)]
            input_feature_test_arr = np.c_[input_feature_test_arr, np.array(test_target)]
            
            print("index 1 of train data: ", input_feature_train_arr[1])
            print("Shape of the train data: ", input_feature_train_arr.shape)
            print("index 1 of test data: ", input_feature_test_arr[1])
            print("Shape of the test data: ", input_feature_test_arr.shape)
            
            
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