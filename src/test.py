import sys
import torch
from torchvision import models, transforms
from fit_preprocess import fit_to_png, fit_to_dict
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
import matplotlib.pyplot as plt
import os
import io
from PIL import Image

def main():
    fit_dir='/home/alex/pytorch_test/test_files/fits/'
    img_dir='/home/alex/pytorch_test/test_files/images/'
    #fit_to_png('/home/alex/pytorch_test/test_files/fits/',
    #        '/home/alex/pytorch_test/test_files/images/')
    #images = fit_to_dict(fit_dir)
    #print(images)
    images = fit_to_dict(fit_dir)
    print(images['2018-01-03-11-59-30-6396'])
    img_buf = io.BytesIO()
    plt.imsave('/home/alex/pytorch_test/test_files/1.png', images['2018-01-03-11-59-30-6396'], cmap='gray')
    im = Image.open(img_buf)
    img_buf.close()
    print(im)

if __name__ == "__main__":
    main()