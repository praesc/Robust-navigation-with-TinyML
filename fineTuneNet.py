import torch
import torch.nn as nn
import torch.optim as optim


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def transfer_net(net, loop, trainSet, classes, path):
    trainloader = torch.utils.data.DataLoader(trainSet,
                                                      batch_size=128,
                                                      shuffle=True,
                                                      num_workers=2)
    testloader = torch.utils.data.DataLoader(trainSet,
                                                     batch_size=1,
                                                     shuffle=True,
                                                     num_workers=2)

    # Set the model to training mode
    # To activate dropout during training
    net.train()

    # Cut old net and define new one. Example VNN1.
    if not path:
        net.conv2 = Identity()
        net.relu2 = Identity()
        net.maxPool2 = Identity()
        net.fc1 = nn.Linear(in_features=180, out_features=2)
    else:
        net.conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(1, 5), padding=(0, 0), stride=(1, 1))
        net.relu2 = nn.ReLU()
        net.maxPool2 = nn.MaxPool2d(kernel_size=(1, 5), padding=(0, 0), stride=(1, 3))
        net.fc1 = nn.Linear(in_features=52, out_features=7)

    # Define a Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # Define what layers should be retrained
    for idx, params in enumerate(net.parameters()):
        if idx < 2:
            params.requires_grad = False
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    #optimizer = optim.SGD(net.fc1.parameters(), lr=0.001, momentum=0.9)

    # Train the network
    print("\n----------------------------------------------------------------")
    print("Training %s..." % (net.name()))
    print('[epoch,  data]')
    for epoch in range(loop):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the input and output
            images, labels = data

            # convert the inputs to 1 channel
            images = images[:, 0, :, :].unsqueeze(1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 50 == 49:    # print every 100 mini-batches
                print('[%5d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

    print('%s has finished Training' % (net.name()))

    # Set the model to test mode
    # To deactivate dropout during testing
    net.eval()

    # Test the network
    print("\n----------------------------------------------------------------")
    print("Testing %s..." % (net.name()))
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            # get the input and output
            images, labels = data

            # convert the inputs to 1 channel
            images = images[:, 0, :, :].unsqueeze(1)

            # forward
            outputs = net(images)

            # determine predicted classes
            _, predicted = outputs.max(1)
            c = (predicted == labels).squeeze()

            # count total true
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(1):
                label = labels[i]
                class_correct[label] += c.item()
                class_total[label] += 1

    # print accuracy for each classes
    for i in range(len(classes)):
        print('Accuracy of %15s : %3.2f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

    # print accuracy for network
    print('Accuracy of %s on the train dataset: %3.2f %%' % (net.name(), 100 * correct / total))
