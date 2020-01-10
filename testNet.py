import torch

classes = ('Crossing', 'EndSpeedLimit', 'FinishLine', 'LeftTurn',
           'RightTurn', 'StartSpeedLimit', 'Straight')


def test_net(net, testSet):
    testloader = torch.utils.data.DataLoader(testSet,
                                             batch_size=1,
                                             shuffle=True,
                                             num_workers=2)
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
    print('Accuracy of %s on the test dataset: %3.2f %%' % (net.name(), 100 * correct / total))
