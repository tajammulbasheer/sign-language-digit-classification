''' preparing data my creating train test and div sets '''

import os
import shutil
import random

# give the path of train dir here 
path = os.getcwd() + '/sign_lang_digits/dataset/train' 
def data_preparation(path):
    os.chdir(path)
    if os.path.isdir('train/0/') is False:
        os.mkdir('train')
        os.mkdir('valid')
        os.mkdir('test')

        for i in range(0,10):
            shutil.move(f'{i}', 'train')
            os.mkdir(f'valid/{i}')
            os.mkdir(f'test/{i}')
        
            valid_samples = random.sample(os.listdir(f'train/{i}'),70)
            for j in valid_samples:
                shutil.move(f'train/{i}/{j}',f'valid/{i}')

            test_samples = random.sample(os.listdir(f'train/{i}'),6)
            for k in test_samples:
                shutil.move(f'train/{i}/{k}',f'test/{i}')
    return

def main():
    data_preparation(path)
if __name__ == '__main__':
  main()