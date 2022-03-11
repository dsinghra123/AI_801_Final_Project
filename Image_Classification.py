import cv2
import glob
import h5py
import imp
import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import shutil
import tensorflow as tf
import time

import skimage as sk
from skimage import util, io, transform

from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.models import Sequential
from tensorflow.keras import layers, mixed_precision
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD

from collections import Counter, deque
from moviepy.editor import *
from moviepy.video.io.VideoFileClip import VideoFileClip
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy import ndarray
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Parent directory path.  This directory is the parent directory for the entire project and likely the working directory
os.chdir('/home/jliew/deepAI')

# Directory to master image file
image_dir = './MasterData/'

strategy = tf.distribute.MirroredStrategy()
train_dir='/home/jliew/deepAI/MasterData/train'
val_dir='/home/jliew/deepAI/MasterData/valid'
test_dir='/home/jliew/deepAI/MasterData/test'

IMAGE_SIZE=(224,224)

train_data=tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    label_mode='categorical',
    image_size=IMAGE_SIZE
)
class_names=train_data.class_names
num_classes=len(class_names)
val_data=tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    label_mode='categorical',
    image_size=IMAGE_SIZE,

)
test_data=tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    label_mode='categorical',
    image_size=IMAGE_SIZE,
    shuffle=False
)

trainData_pf=train_data.prefetch(buffer_size=tf.data.AUTOTUNE)
valData_pf=val_data.prefetch(buffer_size=tf.data.AUTOTUNE)
testData_pf=test_data.prefetch(buffer_size=tf.data.AUTOTUNE)

def create_efficientModel():
    with strategy.scope():
        data_augmentation=keras.Sequential([
            layers.RandomFlip('horizontal'),
            layers.RandomRotation(0.2,fill_mode='nearest'),
        ],name='Data_Augmentation_Layer')

        mixed_precision.set_global_policy('mixed_float16')

        inputs=layers.Input(shape=(224,224,3),name='input_layer')

        base_model=keras.applications.efficientnet.EfficientNetB0(include_top=False)
        base_model.trainable=False

        x=data_augmentation(inputs)

        x=base_model(x,training=False)

        x=layers.GlobalAveragePooling2D(name='Global_Average_Pool_2D')(x)
        num_classes=len(train_data.class_names)
        outputs=layers.Dense(num_classes,activation='softmax',dtype=tf.float32,name="Output_layer")(x)

        model=keras.Model(inputs,outputs,name="model")
        model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=['accuracy']
        )
    model.summary()
    return model

def train_model(model):
    history_of_model=model.fit(
        trainData_pf,
        epochs=5,
        steps_per_epoch=int (len(trainData_pf)),
        validation_data=valData_pf,
        validation_steps=len(valData_pf)
    )
    return model

if __name__ == '__main__':
    model=create_efficientModel()
    model=train_model(model)
    model_0_result=model.evaluate(testData_pf)
    model_0_result
