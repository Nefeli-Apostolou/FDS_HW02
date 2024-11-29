import torch
from torch import nn

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # Convolutional layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()

        # Convolutional layer 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Max pooling 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional layer 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Max pooling 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

        # Fully connected layer (adjusted in_features to 8192)
        self.fc = nn.Linear(in_features=8192, out_features=10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        # Forward pass through conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Forward pass through conv2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool1(x)

        # Forward pass through conv3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool2(x)

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)

        # Dropout for regularization
        x = self.dropout(x)

        # Fully connected layer
        x = self.fc(x)
        return x