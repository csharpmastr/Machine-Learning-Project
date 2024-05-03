import os
import sys
from dataclasses import dataclass

from matplotlib.pyplot import axis
import numpy as np
from numpy import hstack
from numpy import vstack
from numpy import asarray

from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import average_precision_score 

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
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
# from src.utils import evaluate_base_model

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
            
            # get out-of-fold predictions
            meta_x, meta_y = get_out_of_fold(x_train, y_train, base_model)
            logging.info('Out of folds obtained')
            
            print('Meta X:', meta_x.shape)
            print('Meta Y:', meta_y.shape)
            
            # fit base models
            fit_base_models(x_train, y_train, base_model)
            logging.info('Base models fitted')
            
            # fit meta-learner
            meta_learner = fit_meta_learner(meta_x, meta_y)
            logging.info('Meta learner fitted')
            
            # evaluate base models alone
            evaluate_models(x_test, y_test)
            logging.info('Base models evaluated')
            # cv_score = evaluate_base_model(base_model, x_train, y_train)
            
            # evaluate meta learner
            yhat = meta_learner_predictions(x_test, base_model, meta_learner)
            print('Super Learner: %.3f' % (accuracy_score(y_test, yhat) * 100))
            logging.info('Meta Learner evaluated')
            
            # fit and evaluate base-model and meta-model
            # ensemble = get_meta_learner(x_train)
            
            print(x_train.shape)
            print(y_train.shape)
            
            # ensemble.fit(x_train, y_train.ravel())
            
            # logging.info('Base-Model and Meta-Learner has been trained')
            
            # make predictions on test data
            # y_hat = ensemble.predict(x_test)
            
            # summarize base-model
            # print(ensemble.data)
            
            # calculate confusion matrix
            
            confusion = confusion_matrix(y_test, yhat)
            
            # logging.info('Base-Model and Meta-Learner has been tested')
            # print('Meta Learner Accuracy: %.3f' % (accuracy_score(y_test, y_hat) * 100))
            
            # save ensemble model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=meta_learner
            )
            
            logging.info('Ensemble Model has been saved')
            
            return confusion
            
        except Exception as e:
            raise CustomException(e,sys)


# create list of models
def get_base_models():
    try:
        models = list()
        models.append(LogisticRegression(random_state=52, solver='lbfgs', multi_class='multinomial', max_iter=150))
        models.append(KNeighborsClassifier(n_neighbors=5, metric='euclidean'))
        models.append(SVC(gamma='scale', decision_function_shape='ovr', probability=True))
        models.append(RandomForestClassifier(max_depth=5))
        models.append(AdaBoostClassifier(random_state=12))
        models.append(BaggingClassifier(n_estimators=10))
        models.append(get_neural_network_model())
        return models
    
    except Exception as e:
        raise CustomException(e, sys)

# create meta learner
# def get_meta_learner(x):
#     try:
#         ensemble = SuperLearner(scorer=[accuracy_score, average_precision_score ], 
#                             folds=7)
    
#         # add base models
#         models = get_base_models()
#         ensemble.add(models)
        
#         logging.info('Base models added to ensemble')
        
#         # add the meta model
#         ensemble.add_meta(LogisticRegression(solver='lbfgs'))
#         logging.info('Used Logistic Regression as an experiment for the meta learner')
        
#         return ensemble
    
#     except Exception as e:
#         raise CustomException(e,sys)

# collect out-of-fold predictions from k-fol cv
def get_out_of_fold(x, y, models):
    try:
        encoder = OneHotEncoder(categories='auto', sparse_output=False)
        meta_x, meta_y = list(), list()
    
        # define split of data
        kfold = KFold(n_splits=10, shuffle=True)
        
        # for logging purposes
        logged = False
        
        # enumerate splits
        for train_ix, test_ix in kfold.split(x):
            fold_yhats = list()
            
            #get the data
            train_X, test_X = x[train_ix], x[test_ix]
            train_Y, test_Y = y[train_ix], y[test_ix]
            meta_y.extend(test_Y)
            meta_y = encoder.fit_transform(np.array(meta_y).reshape(-1, 1)) 
            
            # fit and make predictions on each base-model
            # flatten_train_y = np.argmax(train_Y, axis=1)
            for model in models:
                model.fit(train_X, train_Y.argmax(axis=1))
                yhat = model.predict_proba(test_X)
                
                if not logged:
                    logging.info('Models have been fitted')
                    print("train_Y shape:", train_Y.shape)
                    print("meta_y shape:", meta_y.shape)
                    print("yhat shape:", yhat.shape)
                    logged = True
                
                # store columns
                if yhat.ndim == 1:  # If `yhat` is already 1D, append directly
                    yhat_onehot = encoder.fit_transform(yhat.reshape(-1, 1))
                    fold_yhats.append(yhat_onehot)
            # store fold_yhats as columns
            print(fold_yhats)
            meta_x.append(hstack(fold_yhats))
            print(meta_x)
        return vstack(meta_x), asarray(meta_y)
    except Exception as e:
        raise CustomException(e, sys)

# fit all base-models
def fit_base_models(x, y, models):
    try:
        for model in models:
            model.fit(x, y)
    except Exception as e:
        raise CustomException(e, sys)

# fit meta-learner
def fit_meta_learner(x, y):
    try:
        model = LogisticRegression(solver='liblinear')
        model.fit(x, y)
        return model
    except Exception as e:
        raise CustomException(e, sys)

# evaluate list of models
def evaluate_models(x, y, models):
    try:
        for model in models:
            yhat = model.predict(x)
            acc = accuracy_score(y, yhat)
            print('%s: %.3f' % (model.__class__.__name__, acc*100))
    except Exception as e:
        raise CustomException(e, sys)
    
# make predictions with the Stacked Model
def meta_learner_predictions(x, models, meta_learner):
    try:
        meta_x = list()
        for model in models:
            yhat = model.predict(x)
            meta_x.append(yhat)
        meta_x = hstack(meta_x)
        
        # predict
        return meta_learner.predict(meta_x)

    except Exception as e:
        raise CustomException(e, sys)

def build_neural_network_model():
    try:
        # Define your neural network
        nn_model = tf.keras.Sequential([
            keras.layers.Input(shape=(12,)),  # Explicit input layer with 12 features
            keras.layers.Dense(24, activation='relu'),  # First hidden layer
            keras.layers.Dense(32, activation='relu'),  # Second hidden layer
            keras.layers.Dropout(0.2),  # Regularization
            keras.layers.Dense(3, activation='softmax')  # Output layer (multi-class classification)
        ])
        
        # Compile the model
        nn_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', 'Recall']
        )
        
        return nn_model  # Returns the model itself

    except Exception as e:
        raise CustomException(e, sys)

def get_neural_network_model():
    try:
        # Wrap the Keras model with KerasClassifier
        keras_clf = KerasClassifier(model=build_neural_network_model, epochs=5, batch_size=32, verbose=0)
        
        logging.info('Neural Network base model created')
        return keras_clf  # Return the KerasClassifier
    
    except Exception as e:
        raise CustomException(e, sys)