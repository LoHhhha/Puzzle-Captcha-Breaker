import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, ReLU, Softmax, Sigmoid, BatchNorm2d


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mode = Sequential(
            Conv2d(3, 64, 2, padding=1),
            BatchNorm2d(64),
            Conv2d(64, 64, 2, padding=1),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(2, stride=2),
            Conv2d(64, 128, 2, padding=1),
            BatchNorm2d(128),
            Conv2d(128, 128, 2, padding=1),
            BatchNorm2d(128),
            ReLU(),
            MaxPool2d(2, stride=2),
            Conv2d(128, 256, 2, padding=1),
            BatchNorm2d(256),
            Conv2d(256, 256, 2, padding=1),
            BatchNorm2d(256),
            Conv2d(256, 256, 2, padding=1),
            BatchNorm2d(256),
            ReLU(),
            MaxPool2d(2, stride=2),
            Conv2d(256, 512, 2, padding=1),
            BatchNorm2d(512),
            Conv2d(512, 512, 2, padding=1),
            BatchNorm2d(512),
            Conv2d(512, 512, 2, padding=1),
            BatchNorm2d(512),
            ReLU(),
            MaxPool2d(2, stride=2),
            Conv2d(512, 512, 2, padding=1),
            BatchNorm2d(512),
            Conv2d(512, 512, 2, padding=1),
            BatchNorm2d(512),
            Conv2d(512, 512, 2, padding=1),
            BatchNorm2d(512),
            ReLU(),
            MaxPool2d(2, stride=2),
            Flatten(),
            Linear(43008, 4096),
            ReLU(),
            Linear(4096, 1024),
            ReLU(),
            Linear(1024, 2),
            Sigmoid()
        )

    def forward(self, net_input):
        net_output = self.mode(net_input)
        return net_output


if __name__ == '__main__':
    net = Net()
    out_input = torch.ones((4, 3, 320, 160))
    out_output = net(out_input)
    print(out_output.shape)
