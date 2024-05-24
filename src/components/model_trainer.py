# from gc import callbacks
import os
import sys
from dataclasses import dataclass

# from matplotlib.pyplot import axis
import pandas as pd
import numpy as np
from numpy import hstack
from numpy import vstack
from numpy import asarray

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, hamming_loss, f1_score, precision_score, recall_score

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization
from scikeras.wrappers import KerasClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import EarlyStopping, save_object, plot_learning_curve

@dataclass
class ModelTrainerConfig:
    trained_meta_learner_file_path = os.path.join("artifacts", "meta_learner.pkl")
    trained_base_model_file_path = os.path.join("artifacts", "base_model.pkl")
    
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
            # base_model = tune_base_models(x_train, np.argmax(y_train, axis=1))
            base_model = get_base_models()
            
            logging.info('Base models obtained')
            
            # get out-of-fold predictions
            meta_x, meta_y = get_out_of_folds(x_train, y_train, base_model)
            logging.info('Out of folds obtained')
            
            # fit base models
            fit_base_models(x_train, y_train, base_model)
            logging.info('Base models fitted')
            
            # evaluate base models alone
            evaluate_models(x_test, y_test, base_model)
            logging.info('Base models evaluated')
            
            meta_learner = build_fit_neural_network_model(meta_x, meta_y)
            logging.info('Meta learner trained')
            
            yhat = meta_learner_predictions(x_test, base_model, meta_learner)
            
            y_test_reshape = np.argmax(y_test, axis=1)
            yhat_reshape = np.argmax(yhat, axis=1)
            
            h_loss = hamming_loss(y_test_reshape, yhat_reshape)
            class_report = classification_report(y_test_reshape, yhat_reshape)
            f1 = f1_score(y_test_reshape, yhat_reshape, average = 'micro')
            precision = precision_score(y_test_reshape, yhat_reshape, average='macro')
            recall = recall_score(y_test_reshape, yhat_reshape, average='macro')
            print('Meta Learner Neural Network: %.3f' % (accuracy_score(y_test_reshape, yhat_reshape) * 100))
            print('Hamming Loss:', h_loss)
            print('F1 Score:', f1)
            print('Precision Score:', precision)
            print('Recall Score:', recall)
            print(class_report)
            logging.info('Meta Learner evaluated')
            
            # calculate confusion matrix
            confusion = confusion_matrix(y_test_reshape, yhat_reshape)
            
            plot_learning_curve(meta_learner)
            
            logging.info('All Models have been evaluated in CV')
            
            # save base models
            save_object(
                file_path=self.model_trainer_config.trained_base_model_file_path,
                obj=base_model
            )
       
            # save ensemble model
            save_object(
                file_path=self.model_trainer_config.trained_meta_learner_file_path,
                obj= meta_learner
            )
            
            logging.info('Ensemble Model has been saved')
            
            return f1, confusion
            
        except Exception as e:
            raise CustomException(e,sys)

# function to tune base models
# def tune_base_models(x_train, y_train):
#     try:
#         # param grid initialization
#         param_grids = {
#             'ExtraTreesClassifier': {
#                 'n_estimators': [10, 50, 100],
#                 'criterion': ['gini', 'entropy', 'log_loss']
#             },
#             'KNeighborsClassifier': {
#                 'n_neighbors': range(1, 21, 2),
#                 'weights': ['uniform', 'distance'],
#                 'metric': ['euclidean', 'manhattan', 'minkowski']
#             },
#             'SVC': {
#                 'C': [0.01, 0.1, 1, 10, 50],
#                 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#                 'gamma': ['scale', 'auto'],
#                 'decision_function_shape': ['ovo', 'ovr'],
#                 'probability': [True]
#             },
#             'RandomForestClassifier': {
#                 'n_estimators': [50, 100, 200],
#                 'max_features': ['sqrt', 'log2'],
#                 'max_depth': [None, 10, 20, 30]
#             },
#             'AdaBoostClassifier': {
#                 'n_estimators': [50, 100, 200],
#                 'algorithm': ['SAMME'],
#                 'learning_rate': [0.01, 0.1, 1]
#             },
#             'BaggingClassifier': {
#                 'n_estimators': [10, 20, 50, 100],
#                 'max_samples': [0.5, 1.0]
#             }
#         }
        
#         tuned_models = []
        
#         for model_name, param_grid in param_grids.items():
#             model = eval(f"{model_name}()")
#             search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
#             search.fit(x_train, y_train)
#             best_model = search.best_estimator_
#             tuned_models.append(best_model)
#             print(f"Best parameters for {model_name}: {search.best_params_}")
#             print(f"Best score for {model_name}: {search.best_score_}")
        
#         # append GNB since no hyperparameters needed to be tuned
#         tuned_models.append(GaussianNB())
        
#         return tuned_models
    
#     except Exception as e:
#         raise CustomException(e, sys)
    

# create list of models
def get_base_models():
    try:
        models = list()
        models.append(ExtraTreesClassifier(n_estimators=50, criterion='gini'))
        models.append(KNeighborsClassifier(n_neighbors=8, metric='manhattan'))
        models.append(SVC(gamma='auto', decision_function_shape='ovo', probability=True, C=50, kernel='rbf'))
        models.append(RandomForestClassifier(max_depth=20, n_estimators=50, max_features='log2'))
        models.append(AdaBoostClassifier(random_state=12, algorithm='SAMME', n_estimators=50, learning_rate=0.01))
        models.append(BaggingClassifier(n_estimators=50, max_samples=1.0))
        models.append(GaussianNB())
        return models
    
    except Exception as e:
        raise CustomException(e, sys)

def get_out_of_folds(x, y, models):
    """Function to calculate base model's prediction that will be used for the 
    meta learner's training

    Args:
        x (arr): feature variables of the training data
        y (arr): target variable of the training data
        models (list): list of base models

    Raises:
        CustomException: For exception handling purposes

    Returns:
        meta_x: base model's prediction on feature variable
        meta_y: true labels on the training data
    """
    try:
        # declare stratified kfold
        skf = StratifiedKFold(n_splits=9, shuffle=True, random_state=42)
        
        # store oof predictions
        meta_x = list()
        
        # declare meta_y
        meta_y = list()
        
        # loop through each folds
        for train_ix, test_ix in skf.split(x, np.argmax(y, axis=1)):
            
            fold_yhat = []
            # split data based on indices
            train_X, test_X = x[train_ix], x[test_ix]
            train_Y, test_Y = y[train_ix], y[test_ix]
            
            # store test_y data
            meta_y.extend(test_Y)
            
            # flatten train_Y for model fitting
            flatten_Y = np.argmax(train_Y, axis=1)
            
            # train base models
            for model in models:
                model.fit(train_X, flatten_Y)
                
                # store predictions from base models
                yhat = model.predict_proba(test_X)
                # yhat = np.argmax(yhat, axis=1)
                fold_yhat.append(vstack(yhat))
                
            meta_x.append(hstack(fold_yhat))
        
        logging.info("Out of fold predictions stored")
        
        meta_x = vstack(meta_x)
        meta_y = asarray(meta_y)
        
        print('Meta_x inside oof:', meta_x.shape)
            
        return meta_x, meta_y
        
    except Exception as e:
        raise CustomException(e, sys)

# fit all base-models
def fit_base_models(x, y, models):
    """Function to train and fit the list of base models

    Args:
        x (arr): feature variables of the training data
        y (arr): target variable of the training data
        models (list): list of base models

    Raises:
        CustomException: For exception handling purposes
    """
    try:
        y_flatten = np.argmax(y, axis=1)
        for model in models:
            model.fit(x, y_flatten)
            
    except Exception as e:
        raise CustomException(e, sys)


# evaluate list of models
def evaluate_models(x, y, models):
    """Function to evaluate base models on prediction unseen testing data

    Args:
        x (arr): feature variables of the testing data
        y (arr): target variable of the testing data
        models (list): list of trained base-models

    Raises:
        CustomException: For exception handling purposes
    """
    try:
        # Convert one-hot encoded `y` to labels
        y_labels = np.argmax(y, axis=1)
        
        for model in models:
            yhat = model.predict(x)
            
            # Convert predictions to labels if needed
            if yhat.ndim > 1 and yhat.shape[1] > 1:
                yhat = np.argmax(yhat, axis=1)
            
            acc = accuracy_score(y_labels, yhat)
            print('%s: %.3f' % (model.__class__.__name__, acc*100))
        
        logging.info('Base models evaluated')
    except Exception as e:
        raise CustomException(e, sys)
    
# make predictions with the Stacked Model
def meta_learner_predictions(x, models, meta_learner):
    """Function to test the meta learning in predicting unseen testing data

    Args:
        x (arr): feature variables of the testing data
        models (list): list of trained base-models
        meta_learner (object): trained sci-keras model

    Raises:
        CustomException: For exception handling purposes

    Returns:
        prediction: Prediction of the meta learner
    """
    try:
        meta_x = list()
        
        for model in models:
                yhat = model.predict_proba(x)
                
                meta_x.append(yhat)
        
        meta_x = hstack(meta_x)

        meta_x = vstack(meta_x)
        # make prediction on meta learner
        prediction = meta_learner.predict(meta_x)
        
        logging.info('Meta Learner has been evaluated')
        
        return prediction

    except Exception as e:
        raise CustomException(e, sys)

def build_fit_neural_network_model(meta_x, meta_y):
    """Function to build, wrap, compile, and train the neural network 
    as the meta learner of the stacking ensemble algorithm

    Args:
        meta_x (arr): base model's prediction
        meta_y (arr): true labels on the training data

    Raises:
        CustomException: For exception handling purposes

    Returns:
        keras_clf: meta learner that will be used for compiling base model's prediction
        to create its own prediction
    """
    try:
        # Definition of the neural network
        nn_model = tf.keras.Sequential([
            keras.layers.Input(shape=(meta_x.shape[1],)),
            keras.layers.Dense(128, activation='relu'),  # First hidden layer   
            BatchNormalization(),      
            keras.layers.Dense(128, activation='relu'),  # Second hidden layer   
            BatchNormalization(),
            keras.layers.Dropout(0.4),  # Regularization
            keras.layers.Dense(64, activation='relu'),
            BatchNormalization(),
            keras.layers.Dense(64, activation='relu'),
            BatchNormalization(),
            keras.layers.Dropout(0.2), # Another penalty
            keras.layers.Dense(64, activation='relu'),
            BatchNormalization(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(meta_y.shape[1], activation='softmax')  # Output layer (multi-class classification)
        ])
        
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        # Compile the model
        nn_model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'Recall', 'Precision'],
            auto_scale_loss=True
        )
        
        # wrap the keras model is the KerasClassifier
        keras_clf = KerasClassifier(model=nn_model,
                                   epochs=100,
                                   batch_size=32,
                                   verbose=0)
        
        # declare model as a classifier
        keras_clf._estimator_type = 'classifier'
        
        logging.info('Neural Network meta learner created')
        
        # split data to have a validation set
        x_train, x_val, y_train, y_val = train_test_split(meta_x, meta_y, test_size=0.2, random_state=65)
        
        early_stopping = EarlyStopping(patience=10, monitor='val_accuracy', mode='max')
        
        keras_clf.fit(
            X=x_train,
            y=y_train,
            validation_data=(x_val, y_val),
            callbacks=[early_stopping]
        )
        
        logging.info('Neural Network meta learner has been trained')
        
        kf = KFold(n_splits=7, shuffle=True, random_state=32)
        cv_score = cross_val_score(keras_clf, x_train, y_train, cv=kf)
        
        print(f'Cross-Validation Scores: {cv_score}')
        print(f'Mean CV Score: {cv_score.mean()}')
        print(f'Standard Deviation of CV Scores: {cv_score.std()}')
        
        logging.info('Neural Network meta learner has been evaluated')

        return keras_clf
    except Exception as e:
        raise CustomException(e, sys)