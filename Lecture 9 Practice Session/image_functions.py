from PIL import Image
import tensorflow as tf
import numpy as np
import os
import matplotlib as plt
import matplotlib.pyplot as plt2

def saveImages(data, path,epoch = ""):
    i = 0
    add = ""
    
    for array in data:
        if len(array.shape) == 2:
                array2 = np.asarray(array*255 , np.uint8)
                array2 = np.reshape(array2,[imwidth,imwidth])
                add = "x"+str(epoch).zfill(3)
                im = Image.fromarray(array2)
                im.save(os.path.join(path,add+str(i)+".png"))
                i = i+1
        if len(array.shape)==3: 
            if array.shape[2] == 1:
                array = array[:, :, 0]
                add = "z"

            im = Image.fromarray(array)
            im.save(os.path.join(path,add+str(i)+".png"))
            i = i+1

def load_preview_pictures(filelist, var):
    """Loads a set of images before TensorFlow is executed"""
    pictures = []
    for filename in filelist:
        with Image.open(filename) as png:
            # converting to greyscale if selected
            if var == 3:
                png = png.convert(mode="RGB")
            else:
                png = png.convert(mode="L")
            # converting PIL.Image into numPy array
            picture = np.asarray(png)
            # make array writable
            picture.setflags(write=True)
            # converting to 2-D array if greyscale
            if var != 3:
                picture = picture[:, :, None]
            pictures.append(picture)
    return pictures
def read_images(input_paths,label_paths):
    """ read filepaths and return decoded images """
    input_content = tf.read_file(input_paths)
    label_content = tf.read_file(label_paths)
    input_images = tf.image.decode_png(input_content, channels=3)
    labels = tf.image.decode_png(label_content, channels=1)
    return input_images, labels

def create_stars_data(input_image, label):
    """ manipulates images as needed for the stars (and similar) dataset """
    input_image = tf.divide(input_image, np.uint8(255), name="normalize_x")
    input_image = tf.image.resize_images(input_image, [imwidth, imwidth])
    #input_image = tf.reshape(input_image,[3,imwidth,imwidth])

    label = tf.divide(label, np.uint8(255), name="normalize_y")
    label = tf.image.resize_images(label, [imwidth, imwidth])
    label = tf.round(label, name="round_y")
    #label = tf.reshape(label,[1,imwidth*imwidth])
    return input_image, label
def showImagesDataGrey(data):
            fig=plt2.figure(figsize=(16, 16))
            columns = len(data)
            rows = 1
            i=0
            for array in data:
                i = i+1
                array2 = np.asarray(array*255 , np.uint8)
                array2 = np.reshape(array2,[imwidth,imwidth])
                fig.add_subplot(rows, columns,i)
                plt2.imshow(array2,cmap='gray')
            plt2.show()
def showImagesDataColor(data):
            fig=plt2.figure(figsize=(16, 16))
            columns = len(data)
            rows = 1
            i=0
            for array in data:
                i = i+1
                array2 = np.asarray(array*255 , np.uint8)
                #array2 = np.reshape(array2,[imwidth,imwidth,3])
                fig.add_subplot(rows, columns,i)
                plt2.imshow(array2,cmap='rgb')
            plt2.show()
def showImagesPathGrey(paths):
    i_ = 0
    rows= 1
    columns = len(paths)
    fig=plt2.figure(figsize=(16, 16))
    for i in paths:
        yimage = plt.image.imread(i)
        i_ = i_+1
        fig.add_subplot(rows, columns,i_)
        plt2.imshow(yimage,cmap='gray')
def showWeights(data,size):
            fig=plt2.figure(figsize=(16, 16))
            columns = 8
            rows = 1
            i=0
            for data2 in data:
                i=i+1
                data2 = 255*data2
                array2 = data2.astype(np.uint8)
                array2 = np.reshape(array2,[size,size,3])
                fig.add_subplot(rows, columns,i)
                plt2.imshow(array2)
            plt2.show()