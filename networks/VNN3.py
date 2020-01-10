import torch.nn as nn
import torch.nn.functional as f


# Define a Convolutional Neural Network
class CNN3(nn.Module):

    def __init__(self):
        super(CNN3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 5), padding=(0, 0), stride=(1, 1))
        self.relu1 = nn.ReLU()
        self.maxPool1 = nn.MaxPool2d(kernel_size=(1, 5), padding=(0, 0), stride=(1, 5))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 5), padding=(0, 0), stride=(1, 1))
        self.relu2 = nn.ReLU()
        self.maxPool2 = nn.MaxPool2d(kernel_size=(1, 5), padding=(0, 0), stride=(1, 5))
        self.fc1 = nn.Linear(in_features=32, out_features=7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxPool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxPool2(x)
        x = x.view(-1, self.num_flat_features(x))
        x = f.dropout(x, p=0.5, training=self.training)
        x = self.fc1(x)
        return x

    # flatten the data matrix to input the linear layer
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def name(self):
        return "CNN3"
