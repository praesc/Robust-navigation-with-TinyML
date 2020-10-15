from networks.VNN4 import CNN4
import quantize 

from torchvision import transforms, datasets
import torch.nn.modules.linear as linear
#from thop import profile

import os
import torch
import pickle
import numpy as np

classes = ('Crossing', 'EndSpeedLimit', 'FinishLine', 'LeftTurn',
           'RightTurn', 'StartSpeedLimit', 'Straight')
pre_classes = ('Dset-1.0', 'Dset-2.0')

# loading and transforming the train dataset
data_transform = transforms.Compose([
    transforms.ToTensor()
    ])

testset = datasets.ImageFolder(root='datasets/combined-test-all',
                               transform=data_transform)
testset_1 = datasets.ImageFolder(root='datasets/Dset-1.0/test',
                               transform=data_transform)
testset_1_5 = datasets.ImageFolder(root='datasets/Dset-1.5/test',
                               transform=data_transform)
testset_2 = datasets.ImageFolder(root='datasets/Dset-2.0/test',
                               transform=data_transform)
testset_pre = datasets.ImageFolder(root='datasets/pre-model/test',
                               transform=data_transform)

testloader = torch.utils.data.DataLoader(testset,
                                     batch_size=1,
                                     shuffle=True,
                                     num_workers=2)
# Load the CNNs models
cnn = CNN4()
if not os.path.exists("Dset-1.0Checkpoint/" + cnn.name() + ".pt"):
    raise ValueError("Net Dset-1.0 not found")
net1 = torch.load("Dset-1.0Checkpoint/" + cnn.name() + ".pt")

if not os.path.exists("Dset-1.5Checkpoint/" + cnn.name() + ".pt"):
    raise ValueError("Net Dset-1.5 not found")
net1_5 = torch.load("Dset-1.5Checkpoint/" + cnn.name() + ".pt")

if not os.path.exists("Dset-2.0Checkpoint/" + cnn.name() + ".pt"):
    raise ValueError("Net Dset-2.0 not found")
net2 = torch.load("Dset-2.0Checkpoint/" + cnn.name() + ".pt")

net1.eval()
net1_5.eval()
net2.eval()

# Quantize the networks
#quantize.quantize(net1, testset_1, classes)
#quantize.quantize(net1_5, testset_1_5, classes)
quantize.quantize(net2, testset_2, classes)

def q_infer(net, inputs):
    # floating point data are scaled and rounded to [-128,127], which are used in
    # the fixed-point operations on the actual hardware (i.e., micro-controller)
    quant_inputs = (inputs * (2 ** net.in_dec_bits)).round()

    # To quantify the impact of quantized data, we scale them back to
    # original range to run inference using quantized data
    inputs = torch.nn.Parameter(quant_inputs / (2 ** net.in_dec_bits))

    # quantize layer by layer
    # basically forward function of the network but layer by layer
    for named_children in net.named_children():
        # flatten function if next layer is linear
        if isinstance(getattr(net, named_children[0]), linear.Linear) and len(list(inputs.shape)) > 2:
            inputs = inputs.flatten()

        # forward for one layer
        outputs = getattr(net, named_children[0])(inputs)

        # we only quantize layer with weight
        if hasattr(getattr(net, named_children[0]), 'weight'):
            # floating point data are scaled and rounded to [-128,127], which are used in
            # the fixed-point operations on the actual hardware (i.e., micro-controller)
            quant_outputs = (outputs * (2 ** getattr(net, named_children[0]).out_dec_bits)).round()

            # To quantify the impact of quantized data, we scale them back to
            # original range to run inference using quantized data
            outputs = torch.nn.Parameter(quant_outputs / (2 ** getattr(net, named_children[0]).out_dec_bits))

        # output of this layer is now input of the next layer
        inputs = outputs

    return torch.Tensor([outputs.data.tolist()])


# Test the network
print("\n----------------------------------------------------------------")
class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))
correct = 0
total = 0

preds = []
count = 0
pos = 0

ML = 1
CNN = 0

# Load the ML model
if ML:
    with open('predictor/Desk-LM/dt_LM_2againstall', 'rb') as model:
        preModel1 = pickle.load(model)
    with open('predictor/Desk-LM/dt_LM_151', 'rb') as model:
        preModel2 = pickle.load(model)

# Load the CNN model
elif CNN:
    preModel = torch.load('preModelCheckpoint/preCNN.pt')
    preModel.eval()
    quantize.quantize(preModel, testset_pre, pre_classes)

with torch.no_grad():
    for data in testloader:
        # get the input and output
        images, labels = data

        # convert the inputs to 1 channel
        images = images[:, 0, :, :].unsqueeze(1)
        img = images.numpy()[0][0][0]

        # 3 featues - Mean
        #left = np.mean(np.asarray(img[0:47])) *255
        #middle = np.mean(np.asarray(img[47:96])) *255
        #right = np.mean(np.asarray(img[96:])) *255
        #feat = [left, middle, right]

        # All features and model does PCA
        feat = []
        for idx, val in enumerate(img):
            val *= 255
            feat.append(val)

        # Forward preModel
        if ML:
            label = preModel1.predict([feat])
            if label == 'Dset-1.0':
                label = preModel2.predict([feat])
        elif CNN:
            #outputs = preModel(images)
            outputs = q_infer(preModel, images)
            _, predicted = outputs.max(1)
            c = predicted.squeeze()
            label = pre_classes[c.item()]

        # Forward VNN
        if label == 'Dset-2.0':
            #outputs = net1(images)
            outputs = q_infer(net2, images)
        elif label == 'Dset-1.5':
            #outputs = net2(images)
            outputs = q_infer(net2, images)
        else:
            outputs = q_infer(net2, images)

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
print('Accuracy of %s on the test dataset: %3.2f %%' % (net1.name(), 100 * correct / total))
