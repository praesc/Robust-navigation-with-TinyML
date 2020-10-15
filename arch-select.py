from networks.VNN1 import CNN1
from networks.VNN2 import CNN2
from networks.VNN4 import CNN4
import quantize

from torchvision import transforms, datasets
import torch.nn.modules.linear as linear
import torch
#from thop import profile

import os
import pickle
import numpy as np


classes = ('Crossing', 'EndSpeedLimit', 'FinishLine', 'LeftTurn',
           'RightTurn', 'StartSpeedLimit', 'Straight')

# loading and transforming the train dataset
data_transform = transforms.Compose([
    transforms.ToTensor()
    ])

testset = datasets.ImageFolder(root='datasets/combined-test-all',
                               transform=data_transform)

testloader = torch.utils.data.DataLoader(testset,
                                     batch_size=1,
                                     shuffle=True,
                                     num_workers=2)
# Load the CNNs models
cnn1 = CNN1()
cnn2 = CNN2()
cnn4 = CNN4()
if not os.path.exists("archCheckpoint/" + cnn1.name() + ".pt"):
    raise ValueError("Net Dset-1.0 not found")
vnn1 = torch.load("archCheckpoint/" + cnn1.name() + ".pt")

if not os.path.exists("archCheckpoint/" + cnn2.name() + ".pt"):
    raise ValueError("Net Dset-1.0 not found")
vnn2 = torch.load("archCheckpoint/" + cnn2.name() + ".pt")

if not os.path.exists("archCheckpoint/" + cnn4.name() + ".pt"):
    raise ValueError("Net Dset-2.0 not found")
vnn4 = torch.load("archCheckpoint/" + cnn4.name() + ".pt")

vnn1.eval()
vnn1.eval()
vnn4.eval()

# Quantize the networks
#quantize.quantize(vnn1, testset, classes)
quantize.quantize(vnn2, testset, classes)
quantize.quantize(vnn4, testset, classes)

def q_infer(net, inputs, labels):
    quant_inputs = (inputs * (2 ** net.in_dec_bits)).round()

    # To quantify the impact of quantized data, we scale them back to
    # original range to run inference using quantized data
    inputs = torch.nn.Parameter(quant_inputs / (2 ** net.in_dec_bits))

    # quantize layer by layer
    # basically forward function of the network but layer by layer
    for named_children in net.named_children():
        # if named_children[0] == 'conv2':
        #    inputs = torch.reshape(inputs_pre, (1,4,1,45))
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
        # inputs_pre = inputs.clone()
        inputs = outputs

    # count total true
    nb_image_correct = False
    if labels.item() == outputs.data.tolist().index(outputs.max().item()):
        nb_image_correct = True

    return nb_image_correct


# Test the network
print("\n----------------------------------------------------------------")
class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))
correct = 0
total = 0

preds = []
count = 0
pos = 0

count_vnn2 = 0
count_vnn4 = 0

# Load the ML model
with open('predictor/Desk-LM/dt_pca3_arch_VNN2-4_84', 'rb') as model:
    preModel = pickle.load(model)

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
        pre_label = preModel.predict([feat])

        # Forward VNN
        if pre_label == 'right':
            isCorrect = q_infer(vnn2, images, labels)
            count_vnn2 += 1
        else:
            isCorrect = q_infer(vnn4, images, labels)
            count_vnn4 += 1

        # count total true
        total += labels.size(0)

        label = labels[0]
        if isCorrect:
            class_correct[labels[0]] += 1
            correct += 1
        class_total[label] += 1


# print accuracy for each classes
for i in range(len(classes)):
    print('Accuracy of %15s : %3.2f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

# print accuracy for network
print('Accuracy of %s on the test dataset: %3.2f %%' % (vnn1.name(), 100 * correct / total))
print('Images classified by %s: %3.3f %%' % (vnn1.name(), count_vnn2 / total))
print('Images classified by %s: %3.3f %%' % (vnn4.name(), count_vnn4 / total))