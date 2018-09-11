from PIL import Image
import numpy as np
import math

import torch
from torch import nn
from torchvision import transforms, models
from collections import OrderedDict

# Get network model
def getModel(arch, input_size, output_size, hidden_layers, class_to_idx):

    if arch == 'resnet18':
        model = models.resnet18(pretrained=True)

    if arch == 'alexnet':
        model = models.alexnet(pretrained=True)

    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)

    if arch == 'squeezenet':
        model = models.squeezenet1_0(pretrained=True)

    if arch == 'densenet':
        model = models.densenet161(pretrained=True)

    if arch == 'inception':
        model = models.inception_v3(pretrained=True)


    for param in model.parameters():
        param.requires_grad = False

    if len(hidden_layers) == 0:
        print('requires at least one hidden layer')
        exit()

    layers = []

    layers.append(('fc_first', nn.Linear(input_size, hidden_layers[0])))
    layers.append(('relu_first', nn.ReLU()))
    layers.append(('do_first', nn.Dropout(p=0.5)))

    i = 0
    while i < len(hidden_layers) - 1:
        layers.append(('fc_' + str(i), nn.Linear(hidden_layers[i], hidden_layers[i+1])))
        layers.append(('relu_' + str(i), nn.ReLU()))
        layers.append(('do_' + str(i), nn.Dropout(p=0.5)))
        i += 1

    layers.append(('fc_last', nn.Linear(hidden_layers[len(hidden_layers) - 1], output_size)))
    layers.append(('relu_last', nn.ReLU()))

    layers.append(('output', nn.LogSoftmax(dim=1)))

    model.classifier = nn.Sequential(OrderedDict(layers))
    model.class_to_idx = class_to_idx

    return model

# Evaluate model
def evaluateModel(device, model, testloader):
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model(inputs)
            probability, predictions = torch.max(output.data, 1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    print('Accuracy {}%'.format(100 * correct / total))

# Train model
def trainModel(device, epochs, model, train_data, trainloader, testloader, optimizer, criterion, scheduler):
    for epoch in range(epochs):
        model.train()
        print('Epochs {}/{}'.format(epoch+1, epochs))

        scheduler.step()

        running_loss = 0.0

        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        print('Training Loss: {}'.format(running_loss / len(train_data)))
        evaluateModel(device, model, testloader)

    print('Training Complete')
    return model

# Retrieve model from checkpoint
def getModelFromCheckpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = getModel(checkpoint['arch'], checkpoint['input_size'], checkpoint['output_size'], checkpoint['hidden_layers'], checkpoint['class_to_idx'])
    model.load_state_dict(checkpoint['state_dict'])
    return model


# Process a single image
def process_image(image):
    img_loader = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])

    pil_image = Image.open(image)
    pil_image = img_loader(pil_image).float()

    np_image = np.array(pil_image)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std
    np_image = np.transpose(np_image, (2, 0, 1))

    return np_image

# Get predictions
def predict(image_path, model, topk, device):

    image = process_image(image_path)

    torchImage = torch.from_numpy(image)
    torchImage.unsqueeze_(0)
    torchImage = torchImage.float().to(device)
    output = model(torchImage)

    return torch.topk(output.data, topk)

# Display predictions
def showPredictions(image_path, model, cat_to_name, topk, device):

    probs, classes = predict(image_path, model, topk, device)

    probs = probs.detach().cpu().numpy()[0]
    probs = [ math.exp(x) for x in probs ]

    classes = classes.cpu().numpy()[0]

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}

    index = 0
    for classIndex in classes:
        print('{} - {}%'.format(cat_to_name[str(idx_to_class[classIndex])], round(probs[index]*100, 2)))
        index =+ 1
