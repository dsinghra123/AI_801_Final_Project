###############################################################################
# Load all necessary libraries for the analysis
import cv2
import glob
import h5py
import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import tensorflow as tf

import keras

from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, mixed_precision
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

from collections import Counter, deque
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.metrics import confusion_matrix, classification_report
from difPy import dif

###############################################################################

# Identify primary directories for the analysis

# Parent directory path.  This directory is the parent directory for the entire project and likely the working directory 
os.chdir('/home/jliew/deepAI/latest/')

# Directory to master image file
image_dir = '../MasterData/' 

# Directory to augmented image file
aug_path = '../birds_augmented'

# Directory to weights path
best_weights_path_cnn = "../deep_best_weights_6conv_birds.hdf5"

best_weights_path_tuned = "../deep_best_weights_6conv_birds_tuned.hdf5"

best_weights_path_EN = "../deep_best_weights_6conv_birds_RGB.hdf5"

# Directory to weights path after tuning
#tuned_best_weights_path = "./tuned_best_weights_6conv_simpsons.hdf5"
efficient_dir = '../efficient/' 

# Directory to save model for app
best_model_save = "../deep_model_birds.hdf5"

# Directory to app image
app_image = './app_image/'
 
###############################################################################

# Identify all fixed variables that will be used in the functions below

# Create dictionary that provides key and name of each folder in the master image directory
map_birds = dict(list(enumerate([os.path.basename(x) for x in glob.glob(image_dir + '/train/*')])))

num_classes = 375
pic_size = 224 #The size that each image will be modified to
batch_size = 32 #The batch size the images will be fed through the model
epochs = 40 #The number of epochs that will be run
num_classes = len(map_birds) #The number of classes for the analysis (number of characters)
train_steps=int(np.ceil(num_classes/batch_size))
train_images = 120
strategy = tf.distribute.MirroredStrategy()

###############################################################################
    
# Initial exploratiory analytics for the dataset

# Get the min and max number of images from the dataset
species_size = []
for k, char in map_birds.items():
    species_size.append(len(glob.glob(image_dir + '/train/%s/*' % char)))

print('Minimum number of images is ' + str(min(species_size)))
print('Maximimum number of images is ' + str(max(species_size)))

# As seen in the output, the minimum number of images in the dataset is 120 and the maximum is 249

# Test to determine of all images are size (224, 224, 3) as the dataset claims
image_size = []
for k, char in map_birds.items():
    pictures_size = [k for k in glob.glob(image_dir + '/train/%s/*' % char)]

    for pic in pictures_size:
        a = cv2.imread(pic)
        image_size.append(a.shape)
    
print('Total number of images not equal to (224, 224, 3) = ' + str(sum(x != (224, 224, 3) for x in image_size)))

# As seen in the output, all images are (224, 224, 3) 

# Identify if duplicate images are present in the dataset

dup_images = []
for k, char in map_birds.items():
    search = dif(image_dir + '/train/%s/' % char)
    dup_images.append(search.result)
    
cat_count = 0
cat_sum = 0
 
for x in dup_images:
    y = len(x)
    if y > 0 :
        cat_count = cat_count + 1
        cat_sum += y
print('Number of species with duplicate images = ' + str(cat_count))
print('Total duplicate images = ' + str(cat_sum))

# As seen in the output, the number of species with duplicate images is 9
# The total number of duplicate images is 11
    
###############################################################################

# Dedup images

# First will create a copy of the images called dedup_train
shutil.copytree(image_dir + '/train/', image_dir + '/dedup_train/')

# Next we will loop through all folders in our new directory and delete the duplicate images
for k, char in map_birds.items():
    result = dif(image_dir + '/dedup_train/%s/' % char, show_output=True, delete=True)

###############################################################################
# Create a function to further augment the images, first create the necessary folders 

def image_augmentation(train_images = train_images):
    # Check if augmented directory exists
    folder_check = os.path.isdir(aug_path)
               
    if not folder_check:
        os.makedirs(aug_path)
        print("created folder : ", aug_path)
    else:
        print(aug_path, "already exists.")
        
    for k, char in map_birds.items():
        
        file = aug_path + '/train/' + '/%s/' % char
        # Creating partitions of the data after shuffling
        
        sub_folder_check = os.path.isdir(file)
        
        if not sub_folder_check:
            os.makedirs(file)
            print("Created Folder: ", file)
        else:
            print(file, "already exists.")           

# Create another function that selects only 120 images from each species of bird to ensure our data set is balanced            

def image_selection():
    
    for k, char in map_birds.items():   
        
        file = aug_path + '/train/' + '/%s/' % char
        src = image_dir +'/dedup_train/' + char  # Folder to copy images from
        dest = file       
        src_files = os.listdir(src)
        
        for file_name in src_files[:train_images]:
            full_file_name = os.path.join(src, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, dest)        
            else:
                print(dest, "already exists.")          
            
image_augmentation()            
image_selection()

# Ensure all augnented filders now only have 120 images
for k, char in map_birds.items():
    augmented_size = len(os.listdir(aug_path + '/train' + '/%s/' % char))
    print(augmented_size)

###############################################################################

# Function to get the images into numpy array format for analysis

def load_pictures(BGR):

    train_pics = []
    train_labels = []
    
    test_pics = []
    test_labels = []
    
    valid_pics = []
    valid_labels = []
    
    for k, char in map_birds.items():
        pictures_train = [k for k in glob.glob(aug_path + '/train/%s/*' % char)]

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

###############################################################################

def get_dataset(save=False, load=False, BGR=False):
    """
    Ensure image content is float32 and images are normalized (/255.). 
    The dataset could be saved or loaded from h5 files.
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
        dist = {k:tuple(d[k] for d in [dict(Counter(np.where(y_train==1)[1])), dict(Counter(np.where(y_test==1)[1]))])
                for k in range(num_classes)}
        print('\n'.join(["%s : %d train pictures & %d val pictures" % (map_birds[k], v[0], v[1]) 
            for k,v in sorted(dist.items(), key=lambda x:x[1][0], reverse=True)]))
    return X_train, X_val, X_test, y_train, y_val, y_test

###############################################################################

# Create a learning rate function for the models below

def lr_schedule(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

###############################################################################

#Build the CNN

def create_model_six_conv(input_shape):
    """
    CNN Keras model with 6 convolutions.
    :param input_shape: input shape, generally X_train.shape[1:]
    :return: Keras model, RMS prop optimizer
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(256, (3, 3), padding='same')) 
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
          optimizer=opt,
          metrics=['accuracy'])  
    return model, opt

###############################################################################

# Define the training model

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

###############################################################################

# Run the model
if __name__ == '__main__':
    X_train, X_val, X_test, y_train, y_val, y_test = get_dataset(load=True)
    model, opt = create_model_six_conv(X_train.shape[1:])
    model, history = training(model, X_train, X_val, y_train, y_val, best_weights_path = best_weights_path_cnn, data_augmentation=True)

###############################################################################

# Run Keras Tuner on the CNN to see if we can inprove accuracy

def build_model(hp):
    # create model object
    model = keras.Sequential([
    
    keras.layers.Conv2D(
        filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=32), 
        kernel_size=(3, 3),
        activation=hp.Choice('conv_1_activation', values = ['relu', 'elu']),
        padding='same', 
        input_shape=(pic_size,pic_size,3)),  
    keras.layers.Conv2D(
        filters=hp.Int('conv_2_filter', min_value=32, max_value=128, step=32), 
        kernel_size=(3, 3),
        activation=hp.Choice('conv_2_activation', values = ['relu', 'elu'])),
    keras.layers.MaxPooling2D(
        pool_size=(2, 2)),
    keras.layers.Dropout(0.1),
    
    keras.layers.Conv2D(
        filters=hp.Int('conv_3_filter', min_value=32, max_value=128, step=32),
        kernel_size=(3, 3),
        activation=hp.Choice('conv_3_activation', values = ['relu', 'elu']),
        padding='same'),  
    keras.layers.Conv2D(
        filters=hp.Int('conv_4_filter', min_value=32, max_value=128, step=32), 
        kernel_size=(3, 3),
        activation=hp.Choice('conv_4_activation', values = ['relu', 'elu'])),
    keras.layers.MaxPooling2D(
        pool_size=(2, 2)),
    keras.layers.Dropout(0.1),
    
    keras.layers.Conv2D(
        filters=hp.Int('conv_5_filter', min_value=128, max_value=512, step=128),
        kernel_size=(3, 3),
        activation=hp.Choice('conv_5_activation', values = ['relu', 'elu']),
        padding='same'),  
    keras.layers.Conv2D(
        filters=hp.Int('conv_6_filter', min_value=128, max_value=512, step=128),
        kernel_size=(3, 3),
        activation=hp.Choice('conv_6_activation', values = ['relu', 'elu'])),
    keras.layers.MaxPooling2D(
        pool_size=(2, 2)),
    keras.layers.Dropout(0.2),
    
    keras.layers.Flatten(),
    keras.layers.Dense(
        units=hp.Int('dense_1_units', min_value=512, max_value=2048, step=512),
        activation=hp.Choice('dense_1_activation', values = ['relu', 'elu'])),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    #compilation of model
    model.compile(
        optimizer=keras.optimizers.SGD(
            lr=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]), 
            decay=1e-6, 
            momentum=0.9, 
            nesterov=True),
        loss='categorical_crossentropy',
        metrics=['accuracy']) 
    
    return model

#creating randomsearch object
tuner = kt.Hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='keras_tuner_results',
                     project_name='bird_tuner')

tuner.search_space_summary()

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

X_train, X_val, X_test, y_train, y_val, y_test = get_dataset(save=True)
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
#run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)
keras.backend.clear_session()
tuner.search(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

###############################################################################

def tuned_model_six_conv(input_shape):
    """
    CNN Keras model with 6 convolutions.
    :param input_shape: input shape, generally X_train.shape[1:]
    :return: Keras model, RMS prop optimizer
    """
    model = Sequential()
    model.add(Conv2D(96, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('elu'))
    model.add(Conv2D(96, (3, 3)))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    
    model.add(Conv2D(96, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(384, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(384, (3, 3), padding='same')) 
    model.add(Activation('elu'))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(1536))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
          optimizer=opt,
          metrics=['accuracy'])  
    return model, opt

if __name__ == '__main__':
    X_train, X_val, X_test, y_train, y_val, y_test = get_dataset(load=True)
    model, opt = tuned_model_six_conv(X_train.shape[1:])
    model, history = training(model, X_train, X_val, y_train, y_val, best_weights_path = best_weights_path_tuned, data_augmentation=True)

###############################################################################

# Create EfficientNet Model

'''
def create_model_efficientnet(input_shape = pic_size):

    inputs = layers.Input(shape = (input_shape, input_shape, 3))

    outputs = EfficientNetB0(include_top= True, weights = None, classes = num_classes, drop_connect_rate=0.2)(inputs)

    model = Model(inputs, outputs)
    
    model.compile(optimizer = 'adam', 
                  loss = 'categorical_crossentropy', 
                  metrics = ['accuracy']) 
    return model
'''
def create_efficientModel():
    with strategy.scope():
        data_augmentation=keras.Sequential([
            layers.RandomFlip('horizontal'),
            layers.RandomRotation(0.2,fill_mode='nearest'),
        ],name='Data_Augmentation_Layer')

        mixed_precision.set_global_policy('mixed_float16')

        inputs=layers.Input(shape=(224,224,3),name='input_layer')

        base_model=keras.applications.efficientnet.EfficientNetB0(include_top=False)
        base_model.trainable=True

        x=data_augmentation(inputs)

        x=base_model(x,training=True)

        x=layers.GlobalAveragePooling2D(name='Global_Average_Pool_2D')(x)
        outputs=layers.Dense(num_classes,activation='softmax',dtype=tf.float32,name="Output_layer")(x)

        model=keras.Model(inputs,outputs,name="model")
        model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=['accuracy']
        )
    model.summary()
    return model

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
          verbose = 2,
          validation_data=(X_val, y_val),
          shuffle=True)
    return model, history

if __name__ == '__main__':
    X_train, X_val, X_test, y_train, y_val, y_test = get_dataset(load=True)
    model = create_efficientModel()
    model, history = training(model, X_train, X_val, y_train, y_val, best_weights_path = best_weights_path_EN, data_augmentation=True)

###############################################################################

def load_model_from_checkpoint(weights_path, input_shape = pic_size):

    model = create_efficientModel()
    model.load_weights(weights_path)
    model.compile(optimizer = 'adam',
                  loss = 'categorical_crossentropy', 
                  metrics = ['accuracy'])
    return model

if __name__ == '__main__':
    X_train, X_val, X_test, y_train, y_val, y_test = get_dataset(load=True)
    model = load_model_from_checkpoint(best_weights_path_EN, pic_size)
    
###############################################################################

preds = model.evaluate(X_test, y_test)

# Ceate image matrix with percentages

F = plt.figure(1, (15,20))
grid = ImageGrid(F, 111, nrows_ncols=(4, 4), axes_pad=0, label_mode="1")

for i in range(16):
    char = map_birds[i]
    image = cv2.imread(np.random.choice([k for k in glob.glob(image_dir + '/test/%s/*' % char) if char in k]))
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pic = cv2.resize(image, (pic_size, pic_size)).astype('float32') / 255.
    a = model.predict(pic.reshape(1, pic_size, pic_size, 3))[0]
    actual = char.split('_')[0].title()
    text = sorted(['{:s} : {:.1f}%'.format(map_birds[k].split('_')[0].title(), 100*v) for k,v in enumerate(a)], 
       key=lambda x:float(x.split(':')[1].split('%')[0]), reverse=True)[:3]
    img = cv2.resize(img, (352, 352))
    cv2.rectangle(img, (0,260),(352,352),(255,255,255), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Actual : %s' % actual, (10, 280), font, 0.7,(0,0,0),2,cv2.LINE_AA)
    for k, t in enumerate(text):
        cv2.putText(img, t,(10, 300+k*18), font, 0.65,(0,0,0),2,cv2.LINE_AA)
    grid[i].imshow(img)
       
eval_result = model.evaluate(X_test, y_test)
print("[test loss, test accuracy]:", eval_result)

y_pred = model.predict(X_test)

plt.figure(figsize=(8,8))
cnf_matrix = confusion_matrix(np.where(y_test > 0)[1], np.argmax(y_pred, axis=1))
classes = list(map_birds.values())
plt.imshow(cnf_matrix, interpolation='nearest')
plt.colorbar()
tick_marks = np.arange(len(classes))
_ = plt.xticks(tick_marks, classes, rotation=90)
_ = plt.yticks(tick_marks, classes)

###############################################################################

model.save(best_model_save)

F = plt.figure(1, (15,20))
grid = ImageGrid(F, 111, nrows_ncols=(4, 4), axes_pad=0, label_mode="1")

for i in range(16):
    char = map_birds[i]
    image = cv2.imread(np.random.choice([k for k in glob.glob(image_dir + '/test/%s/*' % char) if char in k]))
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pic = cv2.resize(image, (pic_size, pic_size)).astype('float32') / 255.
    a = model.predict(pic.reshape(1, pic_size, pic_size, 3))[0]
    actual = char.split('_')[0].title()
    text = sorted(['{:s} : {:.1f}%'.format(map_birds[k].split('_')[0].title(), 100*v) for k,v in enumerate(a)], 
       key=lambda x:float(x.split(':')[1].split('%')[0]), reverse=True)[:3]
    img = cv2.resize(img, (352, 352))
    cv2.rectangle(img, (0,260),(352,352),(255,255,255), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Actual : %s' % actual, (10, 280), font, 0.7,(0,0,0),2,cv2.LINE_AA)
    for k, t in enumerate(text):
        cv2.putText(img, t,(10, 300+k*18), font, 0.65,(0,0,0),2,cv2.LINE_AA)
    grid[i].imshow(img)

x = cv2.imread(app_image)
plt.imshow(x)
plt.show()

model = keras.models.load_model(best_model_save)

for file in glob.glob(app_image + '*.jpg'): 
    image = cv2.imread(file)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (350, 350)).astype('float32') / 255
    pic = cv2.resize(image, (pic_size, pic_size)).astype('float32') / 255
    a = model.predict(pic.reshape(1, pic_size, pic_size, 3))[0]
    text = sorted(['{:s} : {:.1f}%'.format(map_birds[k].split('_')[0].title(), 100*v) for k,v in enumerate(a)], 
                  key=lambda x:float(x.split(':')[1].split('%')[0]), reverse=True)[:3]
    cv2.rectangle(img, (0,275),(250, 350),(255,255,255), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for k, t in enumerate(text):
        cv2.putText(img, t,(10, 300+k*18), font, 0.50,(0,0,0),2,cv2.LINE_AA)
        plt.imshow(img)
    plt.show()
    
# Jake, here is the code you asked for

import matplotlib.pyplot as plt
import keras
import cv2
import glob

app_image = './app_image/'
best_model_save = "./deep_model_birds.hdf5"

pic_size = 224 #The size that each image will be modified to

# Create dictionary that provides key and name of each folder in the master image directory
map_birds = dict(list(enumerate([os.path.basename(x) for x in glob.glob(image_dir + '/train/*')])))
    
model = keras.models.load_model(best_model_save)

for file in glob.glob(app_image + '*.jpg'): 
    image_BGR = cv2.imread(file)
    image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
    image_RGB = cv2.resize(image_RGB, (350, 350)).astype('float32') / 255
    pic = cv2.resize(image_BGR, (pic_size, pic_size)).astype('float32') / 255
    a = model.predict(pic.reshape(1, pic_size, pic_size, 3))[0]
    text = sorted(['{:s} : {:.1f}%'.format(map_birds[k].split('_')[0].title(), 100*v) for k,v in enumerate(a)], 
                  key=lambda x:float(x.split(':')[1].split('%')[0]), reverse=True)[:3]
    cv2.rectangle(image_RGB, (0,275),(250, 350),(255,255,255), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for k, t in enumerate(text):
        cv2.putText(image_RGB, t,(10, 300+k*18), font, 0.50,(0,0,0),2,cv2.LINE_AA)
        plt.imshow(image_RGB)
    plt.show()

