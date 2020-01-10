# =============================================================================
# Author : Romain Donze
# =============================================================================

from networks.VNN1 import CNN1
from trainNet import train_net
from testNet import test_net
from quantize import quantize
from genCode import gen_code

from torchvision import transforms, datasets
from thop import profile

import os
import torch
import torch.onnx

# loading and transforming the train dataset
data_transform = transforms.Compose([
    transforms.ToTensor()
    ])

trainset = datasets.ImageFolder(root="datasets/Dset-2.0/training",
                                transform=data_transform)

testset = datasets.ImageFolder(root='datasets/Dset-2.0/test',
                               transform=data_transform)


def run_net(net, epoch, trainSet, testSet, folder):

    if not os.path.exists(folder + "Checkpoint"):
        os.makedirs(folder + "Checkpoint")

    net = torch.load(folder + "Checkpoint/" + net.name() + ".pt") if os.path.exists(folder + "Checkpoint/" + net.name() + ".pt") else net
    train_net(net, epoch, trainSet)
    torch.save(net, folder + "Checkpoint/" + net.name() + ".pt")
    test_net(net, testSet)
    quantize(net, testSet)
    gen_code(net, folder)


cnn1 = CNN1()
run_net(cnn1, 2000, trainset, testset, "Dset-2.0")



