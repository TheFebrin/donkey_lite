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


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        kernel = (5, 5)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=24, kernel_size=kernel,
                      stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=kernel,
                      stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel,
                    stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),
                    stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),
                    stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        cnn_output_shape = 256
        self.linear = nn.Sequential(
            nn.Linear(cnn_output_shape, 500),
            nn.BatchNorm1d(500),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(500, 1),
            nn.Dropout(0.1),
            nn.Tanh()
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

PATH = '../learning_models/saved_models/model.tch'
model = model = torch.load(PATH)
model.to('cpu')
model.eval()


def predict(image):
    pass


if __name__ == '__main__':
    try:
        while True:
            timer.tick()
            car_status = bluepill.get_status()
            im = cam.get_image() / 255.0
            ang = predict(im)
            thr = 0
            bluepill.drive(ang, car_status.user_throttle)
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
