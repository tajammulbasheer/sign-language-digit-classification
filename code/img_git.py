''' ploting images from training set  '''
import os
import random
import matplotlib.pyplot as plt
from utils import save_fig

def plot_images(path):
    images = []
    for folder in os.listdir(path):
        for image in os.listdir(path + '/' + folder):
            images.append(os.path.join(path, folder, image))

    plt.figure(1, figsize=(6,2))
    plt.axis('off')
    for i in range(12):
        i += 1
        random_img = random.choice(images)
        imgs = plt.imread(random_img)
        ax = plt.subplot(2,6,i)
        plt.axis('off')
        plt.imshow(imgs)
    save_fig('for_git')
    plt.show()

def main():
    os.chdir('sign_lang_digits')
    PATH = os.getcwd() + '/dataset/train' 
    plot_images(PATH)
if __name__ == '__main__':
  main()