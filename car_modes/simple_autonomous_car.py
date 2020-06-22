import sys
sys.path.insert(0, '..')

import datetime
import time
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import car_config
import parts
import cv2
from PIL import Image
import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        kernel = (3, 3)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=kernel, stride=(2, 2), bias=False),
            nn.ELU(alpha=1.0),
            nn.Conv2d(24, 48, kernel_size=kernel, stride=(2, 2), bias=False),
            nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4), dilation=(1, 1)),
            nn.Dropout(p = 0.25)
        )
        self.linear = nn.Sequential(
            nn.Linear(864, 50),
            nn.ELU(alpha=1.0),
            nn.Linear(50, 10),
            nn.Linear(10, 1),
        )


    def forward(self, X):
        out = self.conv(X)
        out = out.reshape(out.size(0), -1)
        out = self.linear(out)
        return out

    def loss(self, Out, Targets):
        loss = torch.sum((Out - Targets) ** 2)
        return loss / Out.size(0)


my_car = car_config.my_car()
bluepill = parts.BluePill(**car_config.bluepill_configs[my_car])
timer = parts.Timer(frequency=20)
cam = parts.PiCamera()

PATH = '../learning_models/saved_models/model1.tch'
model = CNN()
model.load_state_dict(torch.load(PATH, map_location='cpu'))
model.to('cpu')
model.eval()


def predict(img, sigma=0.33):
    CROP_SIZE = 50

    img = img[-CROP_SIZE:, :, :]
    a, b, c = img.shape
    img = img.reshape(1, c, a, b)
    img = torch.Tensor(img)
    print(img.shape)
    out = model(img).detach().numpy()
    return out

if __name__ == '__main__':
    iter = 0
    try:
        while True:
            iter += 1
            print(f'Iter: {iter}')
            timer.tick()
            car_status = bluepill.get_status()
            im = cam.get_image() / 255.0
            ang = predict(im)
            thr = 0.2
            bluepill.drive(ang, car_status.user_throttle)
            # bluepill.drive(ang, thr)
            print(ang, car_status.user_throttle)
    finally:
        bluepill.stop_and_disengage_autonomy()

    '''
    #  Model testing
    DATA = np.load('../learning_models/preprocess/training_data.npy', allow_pickle=True)
    DATA = DATA[:1000]

    X = np.array(DATA[:, 0])
    Y = np.array(DATA[:, 1])
    img_shape = X[0].shape

    X = np.concatenate(X).reshape(-1, *img_shape)

    targets = [x[0] for x in Y] # only angle
    targets = np.array(targets)

    a, b, c = X.shape
    X = X.reshape(a, 1, b, c)
    preds = model(torch.Tensor(X))

    print('Predicting...')
    accur = np.mean(((preds.detach().numpy() - targets) ** 2))
    print(accur)
    '''
