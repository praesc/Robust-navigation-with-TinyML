import torch.nn as nn
import torch.nn.functional as f


# Define a Convolutional Neural Network
class CNN1(nn.Module):

    def __init__(self):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(1, 5), padding=(0, 0), stride=(1, 1))
        self.relu1 = nn.ReLU()
        self.maxPool1 = nn.MaxPool2d(kernel_size=(1, 5), padding=(0, 0), stride=(1, 3))
        self.fc1 = nn.Linear(in_features=180, out_features=2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(1, 5), padding=(0, 0), stride=(1, 1))
        self.relu2 = nn.ReLU()
        self.maxPool2 = nn.MaxPool2d(kernel_size=(1, 5), padding=(0, 0), stride=(1, 3))
        self.fc2 = nn.Linear(in_features=52, out_features=7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxPool1(x)
        x1 = x.view(-1, self.num_flat_features(x))
        x1 = f.dropout(x1, p=0.5, training=self.training)
        y1 = self.fc1(x1)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxPool2(x)
        x = x.view(-1, self.num_flat_features(x))
        x = f.dropout(x, p=0.5, training=self.training)
        y2 = self.fc2(x)
        return y1, y2

    # flatten the data matrix to input the linear layer
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def name(self):
        return "preCNN1"
