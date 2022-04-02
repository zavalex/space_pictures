import os
from astropy.io import fits
import matplotlib.pyplot as plt
import tqdm

def fit_to_png(fit_dir, img_dir):
    dirs = os.listdir(fit_dir)
    for fit_img in tqdm.tqdm(dirs, desc='dirs'):
        try:
            image_file = fits.open(fit_dir+fit_img)
            image_data=image_file[0].data
            png_name = img_dir+fit_img[:-7]+'.png'
            plt.imsave(png_name, image_data[:,:-1], cmap='gray')
        except Exception as e:
            print(e)

def fit_to_dict(FIT_PATH):
    images = {}
    for fit_img in os.listdir(FIT_PATH):
        image_file = fits.open(FIT_PATH+fit_img)
        image_data=image_file[0].data
        images[fit_img[:-7]]=image_data[:,:-1]
    return images