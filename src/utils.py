import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import numpy as np
import dill
import pickle

from scikeras.wrappers import KerasClassifier

from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
import tensorflow as tf

from src.exception import CustomException

# function to visualize distribution before and after sampling
def vizSampling(sampled_data, pre_sampled_data, sampling_method=''):
    plt.figure(figsize=(12, 6))
    
    # Plot the distribution before SMOTE sampling
    plt.subplot(1, 2, 1)
    ax1 = sns.countplot(x='CLASS', data=pre_sampled_data)
    for p in ax1.patches:
        ax1.text(
        p.get_x() + p.get_width() / 2,
        p.get_height() + 0.3,
        '{:.0f}'.format(p.get_height()),
        ha='center',
        va='bottom'
        )
    plt.title(f'Distribution of Target Variable (Before {sampling_method})')
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    # Plot the distribution after SMOTE sampling
    plt.subplot(1, 2, 2)
    ax2 = sns.countplot(x='CLASS', data=sampled_data)
    for p in ax2.patches:
        ax2.text(
        p.get_x() + p.get_width() / 2,
        p.get_height() + 0.3,
        '{:.0f}'.format(p.get_height()),
        ha='center',
        va='bottom'
        )
        
    plt.title(f'Distribution of Target Variable (After {sampling_method})')
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.show()


# function to visualize scaled and sampled data
def vizScaling(scaled_data, pre_scaled_data, scaling_method='', sample_method=''):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))

    ax1.set_title('Before Scaling')
    ax1.set_xlabel('Values')
    ax1.set_ylabel('Count')
    for col in pre_scaled_data.columns:
        sns.kdeplot(pre_scaled_data[col], ax=ax1)
    
    ax2.set_title(f'After {scaling_method} and {sample_method}')
    ax2.set_xlabel('Values')
    ax2.set_ylabel('Count')
    for col in scaled_data.columns:
        sns.kdeplot(scaled_data[col], ax=ax2)
        
    ax1.legend(pre_scaled_data.columns)
    ax2.legend(scaled_data.columns)
    ax1.grid(True)
    ax2.grid(True)
    plt.tight_layout()


# function to save object to folder
def save_object(file_path, obj):
    try:   
        dir_path = os.path.dirname(file_path)
        
        print("Directory Path:", dir_path)
        print("File Path:", file_path)
        
        # create directory
        os.makedirs(dir_path, exist_ok=True)
            
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)
    
# function to load models/object from folder
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


# function to evaluate the model
def evaluate_base_model(models, x_train, y_train):
    rskf =  RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
    
    for model in models:
        if isinstance(model, KerasClassifier) or hasattr(model, 'predict_proba'):
        # One-hot encode for models expecting probabilities
            y_train_encoded = y_train.reshape(1, -1)
            scores = cross_val_score(model, x_train, y_train_encoded, cv=rskf, scoring='accuracy')
        else:
            # Use raw labels for other models
            scores = cross_val_score(model, x_train, y_train.ravel(), cv=rskf, scoring='accuracy')
    
    return scores

# function to plot the learning curve of the meta learner
def plot_learning_curve(model):
    try:
        # creating new figure
        plt.figure(figsize=(12, 6))
        
        # plot loss
        plt.subplot(1, 2, 1)  # Subplot for loss
        plt.plot(model.history_['loss'], label='Training Loss')
        plt.plot(model.history_['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)  # Subplot for accuracy
        plt.plot(model.history_['accuracy'], label='Training Accuracy')
        plt.plot(model.history_['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        raise CustomException(e, sys)


class EarlyStopping(tf.keras.callbacks.Callback):
    """
    Custom callback for early stopping based on validation accuracy
    and loss

    """
    
    def __init__(self, patience=10, monitor='val_accuracy', mode='max'):
        """
        Initialize the callback

        Args:
            patience (int, optional): Number of epochs to wait for improvement before stopping. Defaults to 10.
            monitor (str, optional): Metric to monitor for improvement. Defaults to 'val_accuracy'.
            mode (str, optional): Mode of improvement ('min' for loss, 'max' for accuracy). Defaults to 'max'.
        """
        
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.wait = 0
        self.best_val_acc = -float('inf')
        self.best_val_loss = float('inf')
        
    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch
        
        Args:
        epoch (int): The current epoch number.
        logs (dict, optional): Training and validation logs. Defaults to None
        """ 
        
        val_acc = logs.get('val_accuracy')
        val_loss = logs.get('val_loss')
        
        # Determine improvement based on mode
        if self.mode == 'max':
            improved = (val_acc > self.best_val_acc)
        else:
            improved = (val_loss < self.best_val_loss)
            
        # Update best values if there's improvement
        if improved:
            self.best_val_acc = val_acc
            self.best_val_loss = val_loss
            self.wait = 0  # Reset wait count on improvement
        else:
            self.wait += 1

        # Stop training if patience is exhausted
        if self.wait >= self.patience:
            self.model.stop_training = True
            print("Early stopping triggered: Validation accuracy/loss hasn't improved in", self.patience, "epochs.")
        