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
import logging
import sys
import smdebug.pytorch as smd

from smdebug import modes
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook

import argparse

NUM_CLASSES = 133

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, valid_loader, criterion, device, hook):
    '''
    test the model
    '''
            
    print("START testing")
        
    model.eval()
    hook.set_mode(modes.EVAL)
    correct_test = 0
    val_loss = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss
            pred=outputs.argmax(dim=1, keepdim=True)
            correct_test += pred.eq(labels.view_as(pred)).sum().item()
            
    info_str = f"Val Loss: {val_loss/len(valid_loader.dataset)}, \
            Val Accuracy: {100*(correct_test/len(valid_loader.dataset))}%"
    logger.info(info_str)

def train(model, train_loader, criterion, optimizer, device, hook):
    '''
    train the model
    '''
    print("START TRAINING")
    model.train()
    hook.set_mode(modes.TRAIN)
    running_loss=0
    correct_train=0
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
        correct_train += pred.eq(target.view_as(pred)).sum().item()

    info_str = f"Train Loss: {running_loss/len(train_loader.dataset)}, \
        Train Accuracy: {100*(correct_train/len(train_loader.dataset))}%"        
    logger.info(info_str

def net():
    '''
    get a pre-trained model to do transfer learning
    '''
    pretrained_model = models.resnet50(pretrained=True)
    
    for param in pretrained_model.parameters():
        param.requires_grad = False
        
    pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, NUM_CLASSES)
    
    return pretrained_model

def create_data_loaders(data, batch_size, shuffle):
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)

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
    
    hook = smd.Hook.create_from_json_file()
    hook.register_module(model)
    hook.register_loss(loss_criterion)
    '''
    train the model
    '''
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
                
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = ImageFolder(root=f'{args.ds_path_s3}/train', transform=transform_train)
    valid_dataset = ImageFolder(root=f'{args.ds_path_s3}/valid', transform=transform_test)
                
    train_loader = create_data_loaders(train_dataset, args.batch_size, True)
    valid_loader = create_data_loaders(valid_dataset, args.batch_size, False)
                
    for e in range(args.epochs):
        train(model, train_loader,loss_criterion, optimizer, device, hook)
        test(model, valid_loader, loss_criterion, device, hook)
        
    print('Training model... DONE')
    
    
    '''
    Save the trained model
    '''
    torch.save(
        model.cpu().state_dict(),
        os.path.join(
            args.model_path,
            "model.pth"
        )
    )

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int,default=32)
    parser.add_argument('--bucket_name',type=str,default='sagemaker-us-east-1-272259209864')
    parser.add_argument('--ds_path_s3',type=str,default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--lr',type=float)
    parser.add_argument('--momentum',type=float,default=0.9)
    parser.add_argument('--model_path',type=str,default=os.environ["SM_MODEL_DIR"])
    
    args=parser.parse_args()
    
    main(args)
