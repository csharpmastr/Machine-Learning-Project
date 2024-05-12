import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import sys
import numpy as np
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

from src.exception import CustomException
from src.logger import logging

# from tensorflow.keras.preprocessing import ImageDataGenerator

#function to load data
def load_data(image_path, annotation_paths, target_size=(150, 150)):
    train_images = [cv2.imread(os.path.join(image_path, img)) for img in os.listdir(image_path)]
    train_annotations = {}
    for key, path in annotation_paths.items():
        train_annotations[key] = [cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE) for img in os.listdir(path)]
    return train_images, train_annotations

# load data
try:
    train_image_path = 'notebook\\data\\DR_dataset\\A. Segmentation\\1. Original Images\\a. Training Set'
    train_annot_path = {
        "Microaneurysms":'notebook\\data\\DR_dataset\\A. Segmentation\\2. All Segmentation Groundtruths\\a. Training Set\\1. Microaneurysms',
        "Haemorrhages":'notebook\\data\\DR_dataset\\A. Segmentation\\2. All Segmentation Groundtruths\\a. Training Set\\2. Haemorrhages',
        "Hard Exudates":'notebook\\data\\DR_dataset\\A. Segmentation\\2. All Segmentation Groundtruths\\a. Training Set\\3. Hard Exudates',
        "Soft Exudates":'notebook\\data\\DR_dataset\\A. Segmentation\\2. All Segmentation Groundtruths\\a. Training Set\\4. Soft Exudates',
        "Optic Disc":'notebook\\data\\DR_dataset\\A. Segmentation\\2. All Segmentation Groundtruths\\a. Training Set\\5. Optic Disc'
    }
    
    train_images, train_annotations = load_data(train_image_path, train_annot_path)
    
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Improved class_mode for image classification with multiple classes
    train_generator = datagen.flow_from_directory(
        train_image_path,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        color_mode="rgb",
        seed=42)
    
    print('Augmented Data Length:', len(train_generator))
    
    # Plot some augmented images (optional)
    """ num_images_to_plot = 5
    plt.figure(figsize=(10, 10))
    for i in range(num_images_to_plot):
        image, label = next(train_generator)
        plt.subplot(1, num_images_to_plot, i + 1)
        plt.imshow(train_images[0].astype(np.uint8))
        plt.axis('off')
        plt.title('Augmented Image')
    plt.show() """
    
    # Specific image display
    
    # Choose an index for the sample image and annotation
    sample_index = 5  # You can change this to select a different image

    # Display the sample image
    plt.imshow(cv2.cvtColor(train_images[sample_index], cv2.COLOR_BGR2RGB))
    plt.title(f"Sample Fundus Image (Index: {sample_index})")
    plt.show()
    
    logging.info('Train image has been loaded')

    # Select the annotation name you want to display
    annotation_name = "Microaneurysms"

    # Display the corresponding annotation
    plt.imshow(train_annotations[annotation_name][sample_index], cmap='gray')
    plt.title(f"Sample Annotation: {annotation_name} (Index: {sample_index})")
    plt.show()
    
    logging.info('Train annotations has been loaded')
    
    # Normalization of image data
    image_size = (150, 150)
    train_images_resized = [cv2.resize(img, image_size) for img in train_images]
    train_images_normalized = [img.astype(np.float32) / 255.0 for img in train_images_resized]

    # Display normalized image
    plt.imshow(cv2.cvtColor(train_images_normalized[0], cv2.COLOR_BGR2RGB))
    plt.title(f"Sample Normalized Fundus Image size: {train_images_normalized[0].shape[:2]}")
    plt.show()

    logging.info('Train Images have been normalized')

    # loading of training images
    # train_image = [cv2.imread(os.path.join(train_image_path, img)) for img in os.listdir(train_image_path)]
    
    # logging.info('Train image has been loaded')

    # loading of training annotations
    # train_annot = {
        # key: [cv2.imread(os.path.join(train_annot_path[key], img), cv2.IMREAD_GRAYSCALE) for img in os.listdir(train_annot_path[key])]
        # for key in train_annot_path
    # }
    
    # logging.info('Train annotations image has been loaded')
    
    # print('Train Annotations Length:', len(train_annot))
    
    # specific image (Data Exploration)
    
    # Display sample image
    # plt.imshow(cv2.cvtColor(train_image[0], cv2.COLOR_BGR2RGB))
    # plt.title('Sample Fundus Image')
    # plt.show()
    
    # Display sample annotation
    # plt.imshow(train_annot["Haemorrhages"][0], cmap='gray')
    # plt.title('Sample Annotation Image')
    # plt.show()

    # data pre-processing
    # resizing, scaling, flipping, zoom
    # image_size = (256, 256)
    
    # train_images_resized = [cv2.resize(img, image_size) / 255.0 for img in train_image]
    
        
    # convert images to int dtype
    # train_images_converted = [img.astype(np.uint8) if img.dtype != np.uint8 else img for img in train_images_resized]

    # check data type of each image
    # for img in train_images_converted:
    #     print("Image Data Type after conversion:", img.dtype)
    
    # normalize data
    # train_images_normalized = [cv2.resize(img, image_size).astype(np.float32) / 255 for img in train_images_converted]
    
    # display resized image
    # plt.imshow(cv2.cvtColor(train_images_normalized[0], cv2.COLOR_BGR2RGB))
    # plt.title(f"Sample Resized Fundus Image size: {train_images_normalized[0].shape[:2]}")
    # plt.show()
    
    # logging.info('Train Images has been resized')
    

    # normalization

    # prepping data for training
except Exception as e:
    raise CustomException(e, sys)
