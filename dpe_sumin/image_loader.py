import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from matplotlib.image import imread
from skimage.transform import resize

class Loader:
    def __init__(self):
        self.img_size = 512
        self.channel = 3

        f = open('./filesAdobe.txt', "r")
        lines = f.readlines()

        img_list = []
        for line in lines:
            img_list.append(line.rstrip("\n"))
        self.img_list = img_list

    def load_image(self):
        enhanced_data = []
        raw_data = []

        for line in self.img_list:
            raw_image = self.img_pad(cv2.imread("/data/fivek/test_0624/raw/%s.dng"%line))
            enhanced_image = self.img_pad(cv2.imread("/data/new/fivek_dataset/CLEAR/%s.tif"%line))

            raw_data.append(raw_image)
            enhanced_data.append(enhanced_image)

        return np.array(raw_data), np.array(enhanced_data)

    def img_pad(self,image):
        return resize(image,(512,512))

    def load_from_npy(self):
        raw = np.load('/data/new_raw.npz')["arr_0"]
        enhanced = np.load('/data/new_enhanced.npz')["arr_0"]

        return raw, enhanced
    
    def load_from_npy_old(self):
        raw = np.load('/data/raw.npy')
        enhanced = np.load('/data/en.npy')

        return raw, enhanced
    
    def load_with_plt(self):
        enhanced_data = []
        raw_data = []
        
        raw_img_list = os.listdir("/data/new/fivek_dataset/INPUT")
        raw_img_list.sort()
        clean_img_list = os.listdir("/data/new/fivek_dataset/CLEAR")
        clean_img.list.sort()
        for idx in range(4900):
            raw_image = self.img_pad(imread("/data/new/fivek_dataset/INPUT/%s"%raw_img_list[idx]))
            enhanced_image = self.img_pad(imread("/data/new/fivek_dataset/CLEAR/%s"%clean_img_list[idx]))

            raw_data.append(raw_image)
            enhanced_data.append(enhanced_image)

        return np.array(raw_data), np.array(enhanced_data)
        
        

class DataFlow:
    # 현재
    def __init__(self,is_training):
        if is_training:
            b = 3
        else:
            b = 1
        self.input1_src = keras.Input(dtype=np.uint8, shape=[b, 512, 512, 3])
        self.input1 = tf.cast(self.input1_src) / self.input1_src.dtype.max

        self.input2_src = keras.Input(dtype=np.uint8, shape=[b, 512, 512, 3])
        self.input2 = tf.cast(self.input2_src) / self.input2_src.dtype.max

        if is_training:
            self.input1_label_src = keras.Input(dtype=np.uint8, shape=[b, 512, 512, 3])
            self.input1_label = tf.cast(self.input1_label_src) / self.input1_label_src.dtype.max

            self.input2_label_src = keras.Input(dtype=np.uint8, shape=[b, 512, 512, 3])
            self.input2_label = tf.cast(self.input2_label_src) / self.input2_label_src.dtype.max