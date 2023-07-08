''' ploting images from training set  '''
import os
import random
import matplotlib.pyplot as plt
from utils import save_fig

# give the path of train dir here 
path = os.getcwd() + '/sign_lang_digits/dataset/train' 

def plot_images(path):
  images = []
  for folder in os.listdir(path):
    for image in os.listdir(path + '/' + folder):
      images.append(os.path.join(path, folder, image))

  plt.figure(1, figsize=(6, 6))
  plt.axis('off')
  n = 0
  for i in range(16):
    n += 1
    random_img = random.choice(images)
    imgs = plt.imread(random_img)
    ax = plt.subplot(4, 4,n)
    plt.axis('off')
    plt.imshow(imgs)
  save_fig('train_img')
  plt.show()

def main():
    
    plot_images(path)
if __name__ == '__main__':
  main()