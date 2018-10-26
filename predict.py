import argparse
from torch import nn
import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
import seaborn as sb
import json
from PIL import Image

parser = argparse.ArgumentParser(description='Short sample app')
parser.add_argument('path', type=str)
parser.add_argument('checkpoint', type=str)
parser.add_argument('--category_names', nargs='?', type=str, default="cat_to_name.json")
parser.add_argument('--k_top', nargs='?', type=int, default=5)
parser.add_argument('--gpu', dest='GPU', action='store_true')
parser.add_argument('--cpu', dest='GPU', action='store_false')
parser.set_defaults(GPU=False)
args = parser.parse_args()

#setting variables

if args.path[-4:] != '.jpg':
    args.path = args.path + '.jpg'
if args.checkpoint[-4:] != '.pth':
    args.checkpoint = args.checkpoint + '.pth'
if args.category_names[-5:] != '.json':
    args.category_names = args.category_names + '.pth'  
print(args)

def loader(filepath):
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.classifier=checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict']) 
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)
    img.thumbnail((256, 256))
    img = img.crop((16, 16, 240, 240))
    img = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img/255
    img = (img - mean)/std
    img = img.transpose((2,0,1)) 
    img = torch.tensor( img)
    return img


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)   
    ax.imshow(image)   
    return ax

def predict(image_path, model, gpu, topk=5):
    img = process_image(image_path)
    img = img.unsqueeze_(0)
    img = img.float()
    if gpu:
        img = img.cuda()
        model = model.cuda()
    output = model.forward(img)
    probability = F.softmax(output.data)
    return probability.topk(topk)[0].cpu().numpy()[0], probability.topk(topk)[1].cpu().numpy()[0]

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)
    
the_model = loader(args.checkpoint)   
image_path = args.path
percentage, cat_num = predict(image_path, the_model, args.GPU, args.k_top )  

for i in range(len(cat_num)):
    print(cat_to_name[str(cat_num[i])] + ' has the probabilty of ' + str(percentage[i]))



