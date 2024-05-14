import os
import sys

from sklearn.preprocessing import OneHotEncoder

from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
from src.logger import logging

import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE



# direct exported data to a specific file path

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', 'train_data.csv')
    test_data_path: str=os.path.join('artifacts', 'test_data.csv')
    raw_data_path: str=os.path.join('artifacts', 'raw_data.csv')


# import data from sources
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    # perform data retrieval
    def initiate_data_ingestion(self):
        logging.info('Started Data Ingestion Method')
        
        try:
            data = pd.read_csv('notebook\data\cleaned_data.csv')
            logging.info('Imported .csv Data as a Pandas DataFrame')
            
            # create artifacts directory (if existing, just replace the folder)
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # apply one-hot encoding first to the data
            encoder = OneHotEncoder(categories='auto', handle_unknown='ignore', sparse_output=False)
            
            data_to_encode = data[['Gender', 'CLASS']]
            encoded_data = encoder.fit_transform(data_to_encode)
            
            encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['Gender', 'CLASS']))
            
            data = data.drop(['Gender', 'CLASS'], axis=1)
            encoded_data = pd.concat([data, encoded_df], axis=1)
            encoded_data
            
            # apply oversampling to data before splitting into train and test
            x_features = encoded_data.iloc[:, 0:12]
            y_target = encoded_data.iloc[:, 12:]
            
            # smote = SMOTE(random_state=42)
            # x_resampled, y_resampled = smote.fit_resample(x_features.to_numpy(), y_target.to_numpy())
            
            x_data = pd.DataFrame(x_features, columns=x_features.columns)
            y_data = pd.DataFrame(y_target, columns=y_target.columns)
            data = pd.concat([x_data, y_data], axis=1)
            
            # save raw pandas data to csv
            data.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            # print('Resampled data shape:', resampled_data.shape)
            
            train_data, test_data = train_test_split(data, test_size=0.2, random_state=62)
            logging.info('Train and Test data created')
            
            # save train and test data to csv
            train_data.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info('Data Ingestion completed')
            
            # return train and test data path for data transformation purposes
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__=='__main__':
    obj=DataIngestion()
    
    # obtain train and test data path
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    
    # initialize DataTransformation
    data_transformation = DataTransformation()
    train_data, test_data, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    # initialize ModelTrainer
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_data, test_data))
