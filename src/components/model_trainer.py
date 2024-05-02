import os
import sys
from dataclasses import dataclass

from matplotlib.pyplot import axis
from numpy import hstack
from numpy import vstack
from numpy import asarray

from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import average_precision_score 

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

import tensorflow as tf
from tensorflow import keras
from scikeras.wrappers import KerasClassifier
from mlens.ensemble import SuperLearner

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_base_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            # split train and test data from data transformer
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-3],
                train_arr[:, -3:],
                test_arr[:, :-3],
                test_arr[:, -3:]
            )
            
            logging.info('Data split has been completed')
            
            print(f"Train Data: {x_train.shape, y_train.shape}, \nTest: {x_test.shape, y_test.shape}")
            
            # get base models for evaluation
            base_model = get_base_models()
            
            logging.info('Base models obtained')
            
            # evaluate base models alone
            # cv_score = evaluate_base_model(base_model, x_train, y_train)
            
            # fit and evaluate base-model and meta-model
            ensemble = get_meta_learner(x_train)
            
            print(x_train.shape)
            print(y_train.shape)
            
            ensemble.fit(x_train, y_train.ravel())
            
            logging.info('Base-Model and Meta-Learner has been trained')
            
            # make predictions on test data
            y_hat = ensemble.predict(x_test)
            
            # summarize base-model
            print(ensemble.data)
            
            # calculate confusion matrix
            
            confusion = confusion_matrix(y_test, y_hat)
            
            logging.info('Base-Model and Meta-Learner has been tested')
            print('Meta Learner Accuracy: %.3f' % (accuracy_score(y_test, y_hat) * 100))
            
            # save ensemble model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=ensemble
            )
            
            logging.info('Ensemble Model has been saved')
            
            return confusion
            
        except Exception as e:
            raise CustomException(e,sys)


# create list of models
def get_base_models():
    try:
        models = list()
        models.append(LogisticRegression(random_state=52, solver='liblinear'))
        models.append(KNeighborsClassifier(n_neighbors=5, metric='euclidean'))
        models.append(SVC(gamma='scale'))
        models.append(RandomForestClassifier(max_depth=5))
        models.append(AdaBoostClassifier(random_state=12))
        models.append(BaggingClassifier(n_estimators=10))
        models.append(get_neural_network_model())
        return models
    
    except Exception as e:
        raise CustomException(e, sys)

# create meta learner
def get_meta_learner(x):
    try:
        ensemble = SuperLearner(scorer=[accuracy_score, average_precision_score ], 
                            folds=7)
    
        # add base models
        models = get_base_models()
        ensemble.add(models)
        
        logging.info('Base models added to ensemble')
        
        # add the meta model
        ensemble.add_meta(LogisticRegression(solver='lbfgs'))
        logging.info('Used Logistic Regression as an experiment for the meta learner')
        
        return ensemble
    
    except Exception as e:
        raise CustomException(e,sys)

def build_neural_network_model():
    # Define your neural network
    nn_model = tf.keras.Sequential([
        keras.layers.Input(shape=(12,)),  # Explicit input layer with 12 features
        keras.layers.Dense(24, activation='relu'),  # First hidden layer
        keras.layers.Dense(32, activation='relu'),  # Second hidden layer
        keras.layers.Dropout(0.2),  # Regularization
        keras.layers.Dense(3, activation='sigmoid')  # Output layer (multi-class classification)
    ])
    
    # Compile the model
    nn_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', 'Recall']
    )
    
    return nn_model  # Returns the model itself

def get_neural_network_model():
    try:
        # Wrap the Keras model with KerasClassifier
        keras_clf = KerasClassifier(model=build_neural_network_model, epochs=5, batch_size=32, verbose=0)
        
        logging.info('Neural Network base model created')
        return keras_clf  # Return the KerasClassifier
    
    except Exception as e:
        raise CustomException(e, sys)