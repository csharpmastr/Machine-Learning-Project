import os
import sys
from dataclasses import dataclass

from numpy import hstack
from numpy import vstack
from numpy import asarray

from sklearn import metrics
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import StackingClassifier

import tensorflow as tf
from tensorflow import keras
from scikeras.wrappers import KerasClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_arr, test_arr, preprocessor_path):
        try:
            # split train and test data from data transformer
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1:],
                test_arr[:, :-1],
                test_arr[:, -1:]
            )
            
            logging.info('Data split has been completed')
            
            # instantiate models
            models = {
                "Logistic Regression": LogisticRegression(random_state=52, solver='liblinear'),
                "K-Neighbors Classifier": KNeighborsClassifier(n_neighbors=3, metric='euclidean'),
                "Support Vector Machine": SVC(gamma='scale'),
                "Random Forest": RandomForestClassifier(max_depth=5),
                "AdaBoost Classifier": AdaBoostClassifier(random_state=12),
                "Neural Network": get_neural_network_model()
            }
            
        except Exception as e:
            raise CustomException(e,sys)


def get_neural_network_model():
    # create neural network model
    nn_model = tf.keras.Sequential([
        # input layer
        keras.layers.Dense(24, activation='relu', input_dim=11),
        keras.layers.Dense(32, activation='relu'), # second layer
        keras.layers.Dropout(0.2), # for regularization
        keras.layers.Dense(3, activation='softmax')
    ])
    
    # compile the model
    nn_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', 'Recall']
    )
    
    return nn_model