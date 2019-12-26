import torch.nn as nn
from torch.nn import functional as F


class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()
        # 32 x 32 x 3 (input)
        self.conv1 = nn.Conv2d(3, 100, 5, stride=1, padding=2)  # 32 x 32 x 100
        self.bn1 = nn.BatchNorm2d(100)
        self.mp1e = nn.MaxPool2d(2, stride=2, return_indices=True)  # 11 x 11 x 256
        self.conv2 = nn.Conv2d(100, 150, 5, stride=1, padding=2)  # 16 x 16 x 150
        self.bn2 = nn.BatchNorm2d(150)
        self.mp2e = nn.MaxPool2d(2, stride=2, return_indices=True)  # 16 x 16 x 150
        self.conv3 = nn.Conv2d(150, 200, 3, stride=1, padding=1)  # 8 x 8 x 200

        self.conv4 = nn.ConvTranspose2d(200, 150, 3, stride=1, padding=1)  # 8 x 8 x 200
        self.mp1d = nn.MaxUnpool2d(2)  # 8 x 8 x 150
        self.bn3 = nn.BatchNorm2d(150)
        self.conv5 = nn.ConvTranspose2d(150, 100, 5, stride=1, padding=2)  # 16 x 16 x 150
        self.mp2d = nn.MaxUnpool2d(2)  # 16 x 16 x 100
        self.bn4 = nn.BatchNorm2d(100)
        self.conv6 = nn.ConvTranspose2d(100, 3, 5, stride=1, padding=2)  # 32 x 32 x 100

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x, i_mp1e = self.mp1e(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x, i_mp2e = self.mp2e(x)

        features = self.conv3(x)

        # Decoder
        y = self.conv4(features)
        y = self.bn3(y)
        y = F.relu(y)
        y = self.mp1d(y, i_mp2e)

        y = self.conv5(y)
        y = self.bn4(y)
        y = F.relu(y)
        y = self.mp1d(y, i_mp1e)

        output = self.conv6(y)

        return features, output
