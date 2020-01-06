import torch.nn as nn
from torch.nn import functional as F


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 8 x 8 x 200 input
        self.conv1 = nn.Conv2d(200, 512, 3, 1, padding=1)  # 8 x 8 x 300
        self.bn1 = nn.BatchNorm2d(512)
        self.do1 = nn.Dropout(0.5)
        self.mp1 = nn.MaxPool2d(2, stride=2)  # 4 x 4 x 300

        self.conv2 = nn.Conv2d(512, 1024, 3, 1, padding=1)  # 4 x 4 x 400
        self.bn2 = nn.BatchNorm2d(1024)
        self.do2 = nn.Dropout(0.5)
        self.mp2 = nn.MaxPool2d(2, stride=2)  # 2 x 2 x 400

        self.fc3 = nn.Linear(2 * 2 * 1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.do3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(512, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.do4 = nn.Dropout(0.5)

        self.fc5 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.do5 = nn.Dropout(0.5)

        self.fc6 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.mp1(self.do1(self.bn1(F.relu(self.conv1(x)))))
        x = self.mp2(self.do2(self.bn2(F.relu(self.conv2(x)))))

        x = x.view(-1, 2 * 2 * 1024)

        x = self.do3(self.bn3(F.relu(self.fc3(x))))
        x = self.do4(self.bn4(F.relu(self.fc4(x))))
        x = self.do5(self.bn5(F.relu(self.fc5(x))))

        x = self.fc6(x)
        return x
