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

import argparse

NUM_CLASSES = 133

#TODO: Import dependencies for Debugging andd Profiling

def test(model, test_loader, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
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
    print('Accuracy: {:.2f}%'.format(accuracy))

def train(model, train_loader, criterion, optimizer, device, epoch):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
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
            loss = cost(pred, target)
            running_loss+=loss
            loss.backward()
            optimizer.step()
            pred=pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        print(f"Epoch {e}: Loss {running_loss/len(train_loader.dataset)}, \
         Accuracy {100*(correct/len(train_loader.dataset))}%")
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    pretrained_model = models.resnet50(pretrained=True)
    
    for param in pretrained_model.parameters():
        param.requires_grad = False
        
    pretrained_model.fc = nn.Sequential(nn.Linear(pretrained_model.fc.in_features, NUM_CLASSES))
    
    return pretrained_model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    pass

def main(args):
    
    print(args)
    '''
    TODO: Initialize a model by calling the net function
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    
    model=net()
    model=model.to(device)
    
    '''ß
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    print('PARAMS:',args.lr, args.momentum)
    optimizer = optim.SGD(model.fc.parameters(), lr=args.lr, momentum=args.momentum)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = ImageFolder(root=f's3://{args.bucket_name}/{args.train_ds_path_s3}/', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    model=train(model, train_loader, loss_criterion, optimizer, device, args.epochs)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test_dataset = ImageFolder(root=f's3://{args.bucket_name}/{args.test_ds_path_s3}/', transform=transform)
    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test(model, test_loader, criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model, args.path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_sz', type=int)
    parser.add_argument('--bucket_name',type=str)
    parser.add_argument('--train_ds_path_s3',type=str)
    parser.add_argument('--test_ds_path_s3',type=str)
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--lr',type=float)
    parser.add_argument('--momentum',type=float)
    '''
    TODO: Specify any training args that you might need
    '''
    
    args=parser.parse_args()
    
    main(args)
