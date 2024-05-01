import os
import sys

from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.logger import logging

import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split



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
            
            # save raw pandas data to csv
            data.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
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
    data_transformation.initiate_data_transformation(train_data_path, test_data_path)
