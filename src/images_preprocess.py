import os
from astropy.io import fits
import matplotlib.pyplot as plt
import tqdm
import skimage
import gzip
from PIL import Image
import numpy as np


def fit_to_png(fit_dir, img_dir):
    dirs = os.listdir(fit_dir)
    for fit_img in tqdm.tqdm(dirs, desc='dirs'):
        try:
            image_file = fits.open(fit_dir+fit_img)
            image_data = image_file[0].data
            png_name = img_dir+fit_img[:-7]+'.png'
            plt.imsave(png_name, image_data[:, :-1], cmap='gray')
        except Exception as e:
            print(e)


def fit_to_dict(FIT_PATH):
    images = {}
    for fit_img in os.listdir(FIT_PATH):
        image_file = fits.open(FIT_PATH+fit_img)
        image_data = image_file[0].data
        images[fit_img[:-7]] = image_data[:, :-1]
    return images


def pgm_gz_to_png(gz_path, img_dir):
    images = os.listdir(gz_path)
    for file in tqdm.tqdm(images, desc='dirs'):
        if file[-3:] == '.gz':
            with gzip.open(gz_path + '/' + file, 'rb') as f_in:
                img = skimage.io.imread(f_in)
                print(img.shape)
                plt.imsave(file[:-3]+'.png', img, cmap='gray', vmax=1000)


def cut_auroramax_images(old_path: str, new_path: str) -> None:
    '''Function cuts auroramax 1080p images and create black boxes on 
        watermarks

    Args:
        old_path (str): path to directory with images
        new_path (str): path to directory with processed images
    '''
    dirs = os.listdir(old_path)
    for img in tqdm.tqdm(dirs, desc='dirs'):
        img_array = np.array(Image.open(old_path+img))
        img_array = img_array[:,370:1550,:]
        img_array[-145:,:155,:]=0
        img_array[:50,:200,:]=0
        img_array[:50,870:,:]=0
        img_array[1030:,870:,:]=0
        plt.imsave(new_path+img, img_array)


if __name__ == '__main__':
    print('hello')
