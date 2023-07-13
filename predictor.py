import io
import logging
import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

NUM_CLASSES = 133

def net(device):
    logger.info("Model creation started")
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, NUM_CLASSES))

    model = model.to(device)
    logger.info("Model creation done")

    return model


def model_fn(model_dir):
    model = net("cpu")

    logger.info("Retrieving training model...", model_dir)


    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
   
    return model

def input_fn(request_body, content_type):
    '''
    before predicting the image do some preprocess to put the image into the correct format
    '''
    image = Image.open(io.BytesIO(request_body))

    transformation = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()])

    return transformation(image).unsqueeze(0)