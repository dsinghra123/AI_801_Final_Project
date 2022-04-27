# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 15:24:04 2022

@author: jakec
"""

import kivy
kivy.require('1.0.6') # replace with your current kivy version !

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics.texture import Texture
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import cv2
import glob    
import os
import csv

# Directory to master image file
image_dir = './data/' 
num_classes = 375
pic_size = 224 #The size that each image will be modified to
batch_size = 32 #The batch size the images will be fed through the model
epochs = 30 #The number of epochs that will be run

# Create dictionary that provides key and name of each folder in the master image directory
map_birds = {}       
with open('Names.csv') as csv_file:
    reader = csv.reader(csv_file)
    temp_map_birds = dict(reader)
# There might be a better way to do this but we need to convert the string 
#  keys in the dict to ints so that the search later works correctly
for key in temp_map_birds:
    map_birds[int(key)] = temp_map_birds[key]
    
num_classes = len(map_birds) #The number of classes for the analysis (number of characters)
train_images = 120
  
def generateResults(imagePath=''):
    
    # Directory to master image file
    image_dir = './data/' 
    best_model_save = "./model_birds.hdf5"
    
    pic_size = 224 #The size that each image will be modified to

    model = load_model(best_model_save)
    
    if(imagePath!=''):
        file = imagePath
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
        #plt.show()
        return image_RGB


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

class MyLayout(Widget):
    
    loadfile = ObjectProperty(None)
    
    def dismiss_popup(self):
        self._popup.dismiss()
    
    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()
        
    def load(self, path, filename):
        data = cv2.flip(generateResults(filename[0]), 0)
        flatBuf = data.flatten()
        texture = Texture.create(size=(350, 350), colorfmt='rgb')
        texture.blit_buffer(flatBuf, bufferfmt="float", colorfmt='rgb')
        self.ids.result_image.color = (1, 1, 1, 1)
        self.ids.result_image.texture = texture

        self.dismiss_popup()
    
class BirdFinderApp(App):
    def build(self):
        return MyLayout()

if __name__ == '__main__':
    BirdFinderApp().run()
