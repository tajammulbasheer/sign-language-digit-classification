'''
Code to convert image data into numpy arrays
And creating labels for each numpy array element 
Specifying what image it represents
'''

import os
import numpy as np
from matplotlib.pyplot import imread
from skimage.transform import resize
from keras.utils import to_categorical

img_size = 64
grayscale_images = True
num_class = 10
test_size = 0.2

# reading an image and resizing 
def get_img(data_path):
    img = imread(data_path)
    img = resize(img, (img_size, img_size, 1 if grayscale_images else 3))
    return img

def get_dataset(dataset_path='Dataset'):
    try:
        X = np.load('npy_dataset/X.npy')
        Y = np.load('npy_dataset/Y.npy')
    except:
        labels = ['0','1','2','3','4','5','6','7','8','9']
        X = []
        Y = []
        for i, label in enumerate(labels):
            datas_path = dataset_path+'/'+label
            for data in os.listdir(datas_path):
                img = get_img(datas_path+'/'+data)
                X.append(img)
                Y.append(i)
        X = np.array(X).astype('float32')
        Y = np.array(Y).astype('float32')
        Y = to_categorical(Y, num_class)
        if not os.path.exists('npy_dataset/'):
            os.makedirs('npy_dataset/')
        np.save('npy_dataset/X.npy', X)
        np.save('npy_dataset/Y.npy', Y)
    return 

def main():
    get_dataset(dataset_path='dataset')

if __name__ == '__main__':
    main()