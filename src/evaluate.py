# Command line arguments:
# 1 - path to model
# 2 - folder path to fit files
# 3 - folder path for images
# 4 - path to result file (csv-format)

# Example:
# python evaluate.py /home/alex/pytorch_test/models/effnetb0_1.pt /home/alex/pytorch_test/test_files/07/ /home/alex/pytorch_test/test_files/07_images/ /home/alex/pytorch_test/test_files/predictions.csv

import torch
import torch.nn as nn
from torchvision import models, transforms
import torchvision
import sys
from images_preprocess import fit_to_dict, fit_to_png
#import numpy as np
from PIL import Image
import os
from torch.autograd import Variable
from timeit import default_timer as timer
import tqdm

device = torch.device('cuda:0')
#device = torch.device('cpu')

def load_model(PATH):
    #device = torch.device('cpu')
    model = models.efficientnet_b0(pretrained=False)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=4, bias=True)
    )
    model.load_state_dict(torch.load(PATH, map_location=device))
    return model

transformation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:3,:,:]),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def eval_model(model, images):
    result = {}
    model.eval()
    with torch.no_grad():
        for key, value in images.items():
            pass
    return result

def predict_image(image, model):
    image_tensor = transformation(image).float()
    image_tensor = image_tensor.unsqueeze(0)
    input = Variable(image_tensor)
    input = input.to(device)
    model.cuda()
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index

def eval_model_dataloader(model, images, batch_size:int=32):
    dataset = torchvision.datasets.ImageFolder(images, transformation)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    pass

def eval_model_on_png(model, IMAGE_PATH):
    result = {}
    model.eval()
    with torch.no_grad():
        dirs = os.listdir(IMAGE_PATH)
        for name in tqdm.tqdm(dirs, desc='dirs'):
            img = Image.open(IMAGE_PATH+name)
            result[name]=predict_image(img, model)
    return result

def main():
    args = sys.argv[1:]
    MODEL_PATH = args[0]
    FITS_PATH = args[1]
    IMAGES_PATH = args[2]
    RESULT_PATH = args[3]
    model = load_model(MODEL_PATH)
    #images = fit_to_dict(FITS_PATH)
    start = timer()
    fit_to_png(FITS_PATH, IMAGES_PATH)
    end = timer()
    print('FIT to PNG:', end - start)
    start = timer()
    result = eval_model_on_png(model,IMAGES_PATH)
    end = timer()
    print('Evaluation on PNG:', end - start)
    with open(RESULT_PATH, 'w') as f:
        for key in result.keys():
            f.write("%s,%s\n"%(key,result[key]))
    filtered_dict = {k:v for k,v in result.items() if v==0 or v==2}
    with open('filtered_result.csv', 'w') as f:
        for key in filtered_dict.keys():
            f.write("%s,%s\n"%(key,filtered_dict[key]))

if __name__ == "__main__":
    main()