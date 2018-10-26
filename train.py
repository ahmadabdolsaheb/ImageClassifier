import argparse
from torch import nn
import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import torch.nn.functional as F
import json
from collections import OrderedDict

#parsing arguments
parser = argparse.ArgumentParser(description='Short sample app')
parser.add_argument('dir', type=str)
parser.add_argument('--save_dir', nargs='?', type=str, default="")
parser.add_argument('--arch', nargs='?', type=str, default="densenet121")
parser.add_argument('--learning_rate', nargs='?', type=float, default=0.001)
parser.add_argument('--hidden_units', nargs='?', type=int, default=512)
parser.add_argument('--epochs', nargs='?', type=str, default=5)
parser.add_argument('--gpu', dest='GPU', action='store_true')
parser.add_argument('--cpu', dest='GPU', action='store_false')
parser.set_defaults(GPU=False)
args = parser.parse_args()

#setting variables
print(args)
data_dir = args.dir
train_dir = data_dir + '/train'
test_dir = data_dir + '/test'
print(train_dir)

traintransform = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),  
                                      transforms.Normalize((0.485, 0.456, 0.406), 
                                                           (0.229, 0.224, 0.225)),                                      
                                    ])
testtransform = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),  
                                      transforms.Normalize((0.485, 0.456, 0.406), 
                                                           (0.229, 0.224, 0.225)),                                      
                                    ])

# TODO: Load the datasets with ImageFolder
traindataset = datasets.ImageFolder(train_dir, transform=traintransform)
testdataset = datasets.ImageFolder(test_dir, transform=testtransform)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(traindataset, batch_size=8, shuffle=True)
testloader = torch.utils.data.DataLoader(testdataset, batch_size=8)


#normalize arguents
if args.save_dir != '':
    args.save_dir = args.save_dir + '/'
print(args.save_dir)
if args.arch == "densenet121":
    model = models.densenet121(pretrained=True)
    entry = 1024
    print("densenet121")
else:
    model = models.vgg13(pretrained=True)
    entry = 25088
    print("vgg13")
    
# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(entry, args.hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(args.hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
#define classifier   
model.classifier = classifier

#validation function, one pass forward through the netwrok 
def validation(model, testloader, criterion):
    model.eval()
    accuracy = 0
    test_loss = 0
    for ii, (inputs, labels) in enumerate(testloader):  
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)   
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    print("Epoch: {}/{}.. ".format(e+1, epochs),
      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
      "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
      "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
    model.train() 

#train the network
epochs = args.epochs
steps = 0
running_loss = 0
print_every = 100

if args.GPU:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('gpu')
else:
    device = "cpu"
    print('cpu')
    
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
model.to(device)

for e in range(epochs):
    model.train()
    for ii, (inputs, labels) in enumerate(trainloader):
        steps += 1
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()   
        running_loss += loss.item()
        if steps % print_every == 0:
            print(steps)
    validation(model, testloader, criterion)
    running_loss = 0
print('training finished')

#creating the checkpoint
model.class_to_idx = traindataset.class_to_idx
checkpoint = {
    'arch' : args.arch, 
    'epochs': epochs,
    'batch_size': 8,
    'classifier': classifier,
    'optimizer': optimizer.state_dict(),
    'state_dict': model.state_dict(),
    'class_to_idx': model.class_to_idx
    }

#saving the network
torch.save(checkpoint,  args.save_dir + 'checkpoint.pth')