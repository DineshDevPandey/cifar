import torch.nn as nn
from torch.nn import functional as F


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
                                                                            # 8 x 8 x 200 input
        self.conv1 = nn.Conv2d(200, 300, 3, 1, padding=1)                   # 8 x 8 x 300
        self.mp1 = nn.MaxPool2d(2, stride=2)                                # 4 x 4 x 300
        self.conv2 = nn.Conv2d(300, 400, 3, 1, padding=1)                   # 4 x 4 x 400
        self.mp2 = nn.MaxPool2d(2, stride=2)                                # 2 x 2 x 400
        self.bn1 = nn.BatchNorm2d(400)
        self.fc1 = nn.Linear( 2 * 2 *400, 500)
        self.dropout1 = nn.Dropout(0.5)
        self.bn2 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.mp1(x)

        x = self.conv2(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.mp2(x)

        x = x.view(-1, 2* 2 * 400)
        x = self.fc1(x)
        x = F.relu(x)

        #       x = self.bn2(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x
