# =============================================================================
# Author : Romain Donze
# =============================================================================

from networks.VNN1 import CNN1
from networks.VNN2 import CNN2
from networks.VNN3 import CNN3
from networks.VNN4 import CNN4
from trainNet import train_net
from testNet import test_net
from quantize import quantize
from genCode import gen_code

from torchvision import transforms, datasets
#from thop import profile

import os
import torch
import torch.onnx

trainset_path = 'datasets/combined-train-all'
testset_path = 'datasets/combined-test-all'

# loading and transforming the train dataset
data_transform = transforms.Compose([
    transforms.ToTensor()
    ])

trainset = datasets.ImageFolder(root=trainset_path,
                                transform=data_transform)

testset = datasets.ImageFolder(root=testset_path,
                               transform=data_transform)


def run_net(net, epoch, trainSet, testSet, folder):
    # Find classes
    classes_training = []
    classes_test = []
    for dir in os.listdir(trainset_path):
        classes_training.append(dir)
    for dir in os.listdir(testset_path):
        classes_test.append(dir)
    if classes_test != classes_training:
        print('Wrong dataset')
        return
    classes = tuple(sorted(classes_test))

    # Check if checkpoint exists
    if not os.path.exists(folder + 'Checkpoint'):
        os.makedirs(folder + 'Checkpoint')

    # Start training
    path = folder + 'Checkpoint/' + net.name() + '.pt'
    net = torch.load(path) if os.path.exists(path) else net
    train_net(net, epoch, trainSet, classes)
    torch.save(net, path)
    test_net(net, testSet, classes)
    quantize(net, testSet, classes)
    gen_code(net, folder)


cnn = CNN3()
run_net(cnn, 1000, trainset, testset, 'Dset-2.0')
