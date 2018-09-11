import json
import argparse
import helpers

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from torchvision import datasets, transforms

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', help='Dataset directory')
parser.add_argument('--save_dir', default='.', help='Checkpoint save directory')
parser.add_argument('--arch', choices=['resnet18', 'alexnet', 'vgg16', 'squeezenet', 'densenet'], default='vgg16', help='Architecture')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Training learning rate')
parser.add_argument('--hidden_units', type=int, nargs='+', default=[2048, 512], help='Hidden layers')
parser.add_argument('--epochs', type=int, default=25, help='Number of epochs')
parser.add_argument('--gpu', action="store_true", help='Use GPU')
args = parser.parse_args()

# Check if we can process on GPU
device = torch.device("cpu")
if args.gpu:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print('GPU is not available')
        exit()

# Load category mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

image_size = 224
input_size = int((image_size * image_size) / 2)
output_size = len(cat_to_name)

# Set directories
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Set transoforms
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(image_size),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(image_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Transform images
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# Load images
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)


# Load model
model = helpers.getModel(args.arch, input_size, output_size, args.hidden_units, train_data.class_to_idx)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=args.learning_rate, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = helpers.trainModel(device, args.epochs, model, train_data, trainloader, testloader, optimizer, criterion, scheduler)

torch.save({'arch': args.arch,
            'input_size': input_size,
            'output_size': output_size,
            'hidden_layers': args.hidden_units,
            'state_dict': model.state_dict(),
            'optimizer.state_dict': optimizer.state_dict,
            'class_to_idx': model.class_to_idx
           }, args.save_dir + '/checkpoint.pth')

print('Checkpoint saved')
