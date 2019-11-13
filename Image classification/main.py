from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np


from resnet import ResNet18 as ResNet

from torch.autograd import Variable
from weight_loss import CrossEntropyLoss as CE
from data_transform import DataTransform as DT
import data_utils as du
from robust_adam import Adam

parser = argparse.ArgumentParser(description='Pytorch CIFAR10 Training')
parser.add_argument('-lr','--learning_rate', default='0.1', type=float, help='learning rate')
parser.add_argument('-r','--resume', action='store_true',help='resume from checkpoint')

args = parser.parse_args()
use_cuda = torch.cuda.is_available()


normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

print('==> Preparing data..')
transforms_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
    ])

transforms_test = transforms.Compose([
    transforms.ToTensor(),
    normalize
    ])


best_acc = 0
start_epoch = 0
trainset = torchvision.datasets.CIFAR10(root='./data', train = True, download=True, transform=None)


train_data, train_labels = trainset.data, np.squeeze(trainset.targets)

unlabeled_idxs, labeled_idxs, _ = du.split(trainset, sn=5000, v_sn=0)

train_datasets = DT(trainData=train_data[labeled_idxs,:,:,:], trainLabel=train_labels[labeled_idxs], transform=transforms_train)

trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=100, shuffle=True, num_workers=4)


testset = torchvision.datasets.CIFAR10(root='./data', train = False, download=True, transform=transforms_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


net = ResNet()
criterion = torch.nn.CrossEntropyLoss(size_average=True)
weight_criterion = CE(aggregate='mean')

if use_cuda:
    net.cuda()
    criterion.cuda()
    weight_criterion.cuda()
    cudnn.benchmark = True



def rampup(global_step, rampup_length=80):
    if global_step <rampup_length:
        global_step = np.float(global_step)
        rampup_length = np.float(rampup_length)
        phase = 1.0 - np.maximum(0.0, global_step) / rampup_length
    else:
        phase = 0.0
    return np.exp(-5.0 * phase * phase)


def rampdown(epoch, num_epochs=200, rampdown_length=50):
    if epoch >= (num_epochs - rampdown_length):
        ep = (epoch - (num_epochs - rampdown_length)) * 0.5
        return np.exp(-(ep * ep) / rampdown_length)
    else:
        return 1.0

def step_rampup(epoch, rampup_length=80):
    if epoch<=rampup_length:
        return 1.0
    else:
        return 0.0


def train():

    alpha= 0.6
    Z_now = torch.zeros(len(train_datasets), 16).float().cuda()
    Z = torch.zeros(len(train_datasets), 16).float().cuda()
    z = torch.zeros(len(train_datasets), 16).float().cuda()

    for epoch in range(start_epoch, start_epoch+200):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        rampup_value = rampup(epoch)
        rampdown_value = rampdown(epoch)
        learning_rate = rampdown_value *rampup_value * 0.003
        adam_beta1 = rampdown_value * 0.9 + (1.0 - rampdown_value) * 0.5
        adam_beta2 = step_rampup(epoch) * 0.99 + (1- step_rampup(epoch))* 0.999


        optimizer = Adam(net.parameters(), lr=learning_rate, betas=(adam_beta1, adam_beta2), eps=1e-8)

        if epoch==0:
            u_w = 0.0    
        else:
            u_w = rampup_value        

        u_w_m = u_w*400              # u_w_m = u_w*300 for 1000 training images, u_w_m = u_w*500 for 5000 training images might be better

        u_w = torch.autograd.Variable(torch.FloatTensor([u_w]).cuda(), requires_grad=False)
        u_w_m = torch.autograd.Variable(torch.FloatTensor([u_w_m]).cuda(), requires_grad=False)

        for batch_idx, data in enumerate(trainloader):
            inputs, targets, index = data

            z_comp = z[index,:]
            z_comp = Variable(z_comp, requires_grad=False)

            if use_cuda:
                inputs, targets =inputs.cuda(), targets.cuda()

            optimizer.zero_grad()

            inputs, targets = Variable(inputs), Variable(targets)
            outputs, out_f, alpha_f= net(inputs)
            
            Z_now[index,:] = alpha_f.data.clone().view(alpha_f.size(0)//16, 16)
            

            loss_0 = criterion(outputs, targets)
            loss_1 = weight_criterion(out_f, targets.repeat(4*4,1).permute(1,0).contiguous().view(-1), weights=alpha_f)
            loss_2 = F.mse_loss(alpha_f, z_comp.view(-1), size_average=True)

            loss = loss_0 + loss_1 + u_w_m*loss_2

            loss.backward()
            optimizer.step()

            train_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        Z = alpha*Z+ (1-alpha)*Z_now
        z = Z/(1-alpha**(epoch+1))

        print("Epoch[{}]: Loss: {:.4f} Train accuracy: {}".format(epoch, loss.data[0], 100.*correct/total))

        test(epoch)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0.0
    total = 0.0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets =Variable(inputs, volatile=True), Variable(targets)
        outputs, _, _= net(inputs)

        _, predicted = torch.max(outputs.data,1)
        total +=targets.size(0)
        correct +=predicted.eq(targets.data).cpu().sum()

    print("Epoch[{}] Test accuracy: {}".format(epoch, 100.*correct/total))

    # Save checkpoint.
    acc =100.*float(correct)/total
    if acc > best_acc:
        print('Saving..')
        state = {
                'net': net,
                'acc': acc,
                'epoch': epoch,
            }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc
    print("Best accuracy is:{}".format(best_acc))




if __name__=='__main__':
    train()



