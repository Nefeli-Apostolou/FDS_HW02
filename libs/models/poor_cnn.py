import torch.nn as nn
import torch

class PoorPerformingCNN(nn.Module):
    def __init__(self):
        super(PoorPerformingCNN, self).__init__()
        ##############################
        ###     CHANGE THIS CODE   ###
        ##############################  
        # 2D Convolution layer:
        # Takes input with 3 channels and returns output with 4 channels
        # Output height = Input height because H_out = (H_in + 2*padding - kernel_size)/stride +1
        # H_out = (H_in + 2*1 - 3)/1 + 1 = H_in
        # Output width = Input width for the same reason
        # Output dims: 32x32x4 (for CIFAR-10)
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)

        # ReLU layer: 
        # Returns a tensor of the same shape as the input
        # All negative values are replaced by 0
        # Output dims for 32x32x3 inputs: 32x32x4 (for CIFAR-10)
        self.relu1 = nn.ReLU()

        # Max-pooling layer:
        # Reduces spatial dimensions
        # H_out = (H_in + 2*padding - kernel_size)/stride + 1
        # W_out = (W_in + 2*padding - kernel_size)/stride + 1
        # If we apply this CNN to the CIFAR-10 dataset where images are 32x32
        # H_out = (32 + 2*0 - 2)/2 + 1 = 16
        # W_out = (32 + 2*0 - 2)/2 + 1 = 16
        # Output dims for 32x32x4 inputs: 16x16x4 (for CIFAR-10)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Convolutional layer:
        # We corrected the in_channels parameter, from 5 to 4
        # conv2 is defined with 5 input channels, but the ouput of conv1 has 4 channels
        # and the layers between conv2 and conv1 do not modify the number of channels
        # So in_channels must be 4 here, not 5:
        # Height and Width are not modified
        # Output dims for 16x16x4 input: 16x16x8 (for CIFAR-10)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)

        # Another ReLU layer:
        self.relu2 = nn.ReLU()

        # Linear layer:
        # Applies affine linear transformation to the input data
        # We adjusted the input size of the fully connected layer from 8 * 4 * 4 to 8 * 8 * 8
        # This layer is applied at the end, so we need to consider
        # the shape of the data after it went through the previous transformations
        # Looking at the forward function, we can calculate this to be (64,8,8,8)
        # Therefore, for each input sample, the shape is (8,8,8)
        # Then, x = x.view(-1, 8 * 8 * 8) flattens the tensor, right before the linear layer is applied
        # This means that 8 * 8 * 8 is the input size for nn.Linear()
        # We also changed the output size of each sample from 28 to 10
        # The CIFAR-10 dataset has 10 classes and this is the last layer
        # so we want the output to be a distribution over the 10 classes
        self.fc1 = nn.Linear(8 * 8 * 8, 10)

    def forward(self, x):
        # Conv2d layer + ReLU layer + MaxPool2d layer
        # Input dimensions: 64 x 32 x 32 x 3 (batch_size x height x width x num_channels)
        # Output dimensions: 64 x 16 x 16 x 4
        x = self.pool(self.relu1(self.conv1(x)))

        # Conv2d layer + ReLU layer + MaxPool2d layer
        # Input dimensions: 64 x 16 x 16 x 4 (batch_size x height x width x num_channels)
        # Output dimensions: 64 x 8 x 8 x 8 
        x = self.pool(self.relu2(self.conv2(x)))

        # Reshape the data into a vector
        x = x.view(-1, 8 * 8 * 8)

        # Apply the last (linear) layer that outputs a distribution over the classes
        x = self.fc1(x)

        return x