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

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

from collections import Counter, deque
from moviepy.editor import *
from moviepy.video.io.VideoFileClip import VideoFileClip
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy import ndarray
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Parent directory path.  This directory is the parent directory for the entire project and likely the working directory 
os.chdir('D:/School_Files/AI_801/Project/final_project_git')

# Directory to master image file
image_dir = './Master_Data/' 

# Directory to train/test split data
#split_path = './simpsons_split'

# Directory to augmented image file
#aug_path = './simpsons_augmented'

# Directory to weights path
best_weights_path = "./best_weights_6conv_birds.hdf5"

# Directory to weights path after tuning
#tuned_best_weights_path = "./tuned_best_weights_6conv_simpsons.hdf5"

map_birds = dict(list(enumerate([os.path.basename(x) for x in glob.glob(image_dir + '/train/*')])))

pic_size = 128 #The size that each image will be modified to
batch_size = 32 #The batch size the images will be fed through the model
epochs = 50 #The number of epochs that will be run
num_classes = len(map_birds) #The number of classes for the analysis (number of characters)
#val_size = 0.20 #Size of the validation set as proportion of all images per character
#train_images = 2000 #Number of images in the training set for each character
#test_images = 300 #Number of images in the test set for each character
#pictures_per_class = 2000 #Number of images for each character

def load_pictures(BGR):

    train_pics = []
    train_labels = []
    
    test_pics = []
    test_labels = []
    
    valid_pics = []
    valid_labels = []
    
    for k, char in map_birds.items():
        pictures_train = [k for k in glob.glob(image_dir + '/train/%s/*' % char)]

        for pic in pictures_train:
            a = cv2.imread(pic)
            if BGR:
                a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
            a = cv2.resize(a, (pic_size,pic_size))
            train_pics.append(a)
            train_labels.append(k)
            
        pictures_test = [k for k in glob.glob(image_dir + '/test/%s/*' % char)]
        
        for pic in pictures_test:
            a = cv2.imread(pic)
            if BGR:
                a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
            a = cv2.resize(a, (pic_size,pic_size))
            test_pics.append(a)
            test_labels.append(k)
            
        pictures_valid = [k for k in glob.glob(image_dir + '/valid/%s/*' % char)]
        
        for pic in pictures_valid:
            a = cv2.imread(pic)
            if BGR:
                a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
            a = cv2.resize(a, (pic_size,pic_size))
            valid_pics.append(a)
            valid_labels.append(k)

    return np.array(train_pics), np.array(train_labels), np.array(test_pics), np.array(test_labels), np.array(valid_pics), np.array(valid_labels) 

def get_dataset(save=False, load=False, BGR=False):
    """
    Create the actual dataset split into train and test, pictures content is as float32 and
    normalized (/255.). The dataset could be saved or loaded from h5 files.
    :param save: saving or not the created dataset
    :param load: loading or not the dataset
    :param BGR: boolean to use true color for the picture (RGB instead of BGR for plt)
    :return: X_train, X_test, y_train, y_test (numpy arrays)
    """
    if load:
        h5f = h5py.File('dataset.h5','r')
        X_train = h5f['X_train'][:]
        X_val = h5f['X_val'][:]
        X_test = h5f['X_test'][:]
        h5f.close()    

        h5f = h5py.File('labels.h5','r')
        y_train = h5f['y_train'][:]
        y_val = h5f['y_val'][:]
        y_test = h5f['y_test'][:]
        h5f.close()    
    else:
        X_train, y_train, X_test, y_test, X_val, y_val = load_pictures(BGR)
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        y_val = keras.utils.to_categorical(y_val, num_classes)
        
        if save:
            h5f = h5py.File('dataset.h5', 'w')
            h5f.create_dataset('X_train', data=X_train)
            h5f.create_dataset('X_val', data=X_val)
            h5f.create_dataset('X_test', data=X_test)
            h5f.close()

            h5f = h5py.File('labels.h5', 'w')
            h5f.create_dataset('y_train', data=y_train)
            h5f.create_dataset('y_val', data=y_val)
            h5f.create_dataset('y_test', data=y_test)
            h5f.close()
            
    X_train = X_train.astype('float32') / 255.
    X_val = X_val.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    print("Train", X_train.shape, y_train.shape)
    print("Val", X_val.shape, y_val.shape)
    print("Test", X_test.shape, y_test.shape)
    if not load:
        dist = {k:tuple(d[k] for d in [dict(Counter(np.where(y_train==1)[1])), dict(Counter(np.where(y_val==1)[1])),
                                       dict(Counter(np.where(y_test==1)[1]))]) 
                for k in range(num_classes)}
        print('\n'.join(["%s : %d train pictures & %d val pictures" % (map_birds[k], v[0], v[1]) 
            for k,v in sorted(dist.items(), key=lambda x:x[1][0], reverse=True)]))
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_model_six_conv(input_shape):

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same')) 
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(512, (3, 3), padding='same')) 
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    
    model.compile(loss='categorical_crossentropy',
          optimizer=opt,
          metrics=['accuracy'])    
    
    return model, opt

def load_model_from_checkpoint(weights_path, input_shape=(pic_size,pic_size,3)):
    model, opt = create_model_six_conv(input_shape)
    model.load_weights(weights_path)
    return model

def lr_schedule(epoch):
    lr = 0.01
    return lr*(0.1**int(epoch/10))

def training(model, X_train, X_val, y_train, y_val, best_weights_path, data_augmentation=True):

    if data_augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=True)  # randomly flip images
        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(X_train)
        filepath = best_weights_path
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')
        callbacks_list = [LearningRateScheduler(lr_schedule) ,checkpoint]
        history = model.fit(datagen.flow(X_train, y_train,
                            batch_size=batch_size),
                            steps_per_epoch=X_train.shape[0] // batch_size,
                            epochs=epochs,
                            validation_data=(X_val, y_val),
                            callbacks=callbacks_list)        
    else:
        history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_val, y_val),
          shuffle=True)
    return model, history

if __name__ == '__main__':
    X_train, X_val, X_test, y_train, y_val, y_test = get_dataset(save=True)
    model, opt = create_model_six_conv(X_train.shape[1:])
    model, history = training(model, X_train, X_val, y_train, y_val, best_weights_path = best_weights_path, data_augmentation=True)
        
##############################################################################

# Create all image augmentation functions

# Random image rotation between -45 anf 45 deg
def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.randint(-45, 45)
    return sk.transform.rotate(image_array, random_degree)

# Random noise
def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)

# Flip the image on the y axis (horizontally)
def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]

# Randomly crop 1 to 75 pixels from each side of the image
def crop_image(image_array: ndarray):
    rand_1 = random.randint(10,75)
    rand_2 = random.randint(10,75)
    rand_3 = random.randint(10,75)
    rand_4 = random.randint(10,75)
    return sk.util.crop(image_array, ((rand_1, rand_2), (rand_3, rand_4), (0,0)), copy=False)

# Create a dictionary of the transformations we defined above for use in functions
available_transformations = {'rotate': random_rotation,
                             'noise': random_noise,
                             'horizontal_flip': horizontal_flip,
                             'crop': crop_image
                             }

##############################################################################

def image_augmentation(train_images = train_images, test_images = test_images):
    # Check if augmented directory exists
    folder_check = os.path.isdir(aug_path)
            
    if not folder_check:
        os.makedirs(aug_path)
        print("created folder : ", aug_path)
    else:
        print(aug_path, "already exists.")
        
        for folder in os.listdir(split_path):
            if folder == 'train':
                aug_images_number = train_images
                split = 'train'
            else:
                aug_images_number = test_images
                split = 'test'       
        
            for k, char in map_characters.items():
                folder_path = os.path.join(aug_path, split, char)             
                num_files_desired = aug_images_number            
                subfolder_check = os.path.isdir(folder_path)
                
                if not subfolder_check:
                    shutil.copytree(os.path.join(image_dir , char), folder_path)
                    print("copied folder : ", folder_path)
                    
                    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            
                    num_generated_files = len(os.listdir(os.path.join(image_dir , char)))
                    while num_generated_files <= num_files_desired:
                        # random image from the folder
                        image_path = random.choice(images)
                        # read image as an two dimensional array of pixels
                        image_to_transform = sk.io.imread(image_path)
                        # random num of transformation to apply
                        num_transformations_to_apply = random.randint(1, len(available_transformations))
                    
                        num_transformations = 0
                        transformed_image = None
                        while num_transformations <= num_transformations_to_apply:
                            # random transformation to apply for a single image
                            key = random.choice(list(available_transformations))
                            transformed_image = available_transformations[key](image_to_transform)
                            num_transformations += 1
                    
                            new_file_path = '%s/pic_%s.jpg' % (folder_path, num_generated_files)
                    
                            # write image to the disk
                            io.imsave(new_file_path, transformed_image)
                        num_generated_files += 1    
                        
                elif len(glob.glob(aug_path + '/%s/*' % char)) < aug_images_number:
                    
                    print('adding ', (aug_images_number - len(glob.glob(aug_path + '/%s/*' % char))), ' images to ', folder_path)
                    num_generated_files = len(glob.glob(aug_path + '/%s/*' % char))
                    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                    
                    while num_generated_files <= num_files_desired:
                        # random image from the folder
                        image_path = random.choice(images)
                        # read image as an two dimensional array of pixels
                        image_to_transform = sk.io.imread(image_path)
                        # random num of transformation to apply
                        num_transformations_to_apply = random.randint(1, len(available_transformations))
                    
                        num_transformations = 0
                        transformed_image = None
                        while num_transformations <= num_transformations_to_apply:
                            # random transformation to apply for a single image
                            key = random.choice(list(available_transformations))
                            transformed_image = available_transformations[key](image_to_transform)
                            num_transformations += 1
                    
                            new_file_path = '%s/pic_%s.jpg' % (folder_path, num_generated_files)
                    
                            # write image to the disk
                            io.imsave(new_file_path, transformed_image)
                        num_generated_files += 1
                else:
                    print(folder_path, "already exists.")