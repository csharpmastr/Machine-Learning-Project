import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import numpy as np
import dill

from scikeras.wrappers import KerasClassifier

from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

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
            dill.dump(obj, file_obj)
        
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
def plot_learning_curve(history):
    try:
        # creating new figure
        plt.figure(figsize=(12, 6))
        
        # plot loss
        plt.subplot(1, 2, 1)  # Subplot for loss
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)  # Subplot for accuracy
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        raise CustomException(e, sys)