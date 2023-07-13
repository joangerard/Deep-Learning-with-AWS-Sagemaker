#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import boto3
import os
import sys
import logging

import argparse

NUM_CLASSES = 133

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, device):
    '''
    test the model
    '''
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('Test Accuracy: {:.2f}'.format(accuracy))
    logger.info('Test Accuracy: {:.2f}'.format(accuracy))

def train(model, train_loader, criterion, optimizer, device, epoch):
    '''
    train the model
    '''
    model.train()
    for e in range(epoch):
        running_loss=0
        correct=0
        for data, target in train_loader:
            data=data.to(device)
            target=target.to(device)
            optimizer.zero_grad()
            pred = model(data)             #No need to reshape data since CNNs take image inputs
            loss = criterion(pred, target)
            running_loss+=loss
            loss.backward()
            optimizer.step()
            pred=pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        info_str = f"Epoch {e}: Loss {running_loss/len(train_loader.dataset)}, \
         Accuracy {100*(correct/len(train_loader.dataset))}%"
        print(info_str)
        logger.info(info_str)
    return model

def net():
    '''
    get a pre-trained model to do transfer learning
    '''
    pretrained_model = models.resnet50(pretrained=True)
    
    for param in pretrained_model.parameters():
        param.requires_grad = False
        
    pretrained_model.fc = nn.Sequential(nn.Linear(pretrained_model.fc.in_features, NUM_CLASSES))
    
    return pretrained_model

def create_data_loaders(data, batch_size):
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

def main(args):

    logger.info(f'hyperparams: {args}')
    '''
    Initialize a model by calling the net function
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    model=net()
    model=model.to(device)
    print('Getting the model... DONE')
    
    '''
    Create loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=args.lr, momentum=args.momentum)
    print('Creating criterion and optimizer... DONE')
    
    '''
    train the model
    '''
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = ImageFolder(root=f'{args.ds_path_s3}/train', transform=transform)
    train_loader = create_data_loaders(train_dataset, args.batch_size)
    model=train(model, train_loader, loss_criterion, optimizer, device, args.epochs)
    print('Training model... DONE')
    
    '''
    Test the model to see its accuracy
    '''
    test_dataset = ImageFolder(root=f'{args.ds_path_s3}/test', transform=transform)
    test_loader = create_data_loaders(test_dataset, batch_size=args.batch_size)
    test(model, test_loader, device)
    print('Evaluating model... DONE')
    
    '''
    Save the trained model
    '''
    torch.save(model, args.path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int,default=32)
    parser.add_argument('--bucket_name',type=str,default='sagemaker-us-east-1-272259209864')
    parser.add_argument('--ds_path_s3',type=str,default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--lr',type=float)
    parser.add_argument('--momentum',type=float,default=0.9)
    parser.add_argument('--path',type=str,default='model.h5')
    '''
    TODO: Specify any training args that you might need
    '''
    
    args=parser.parse_args()
    
    main(args)
