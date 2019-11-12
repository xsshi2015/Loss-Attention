from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
from scipy import io

from dataloader import MnistBags
from model import CNN
from weight_loss import CrossEntropyLoss as CE
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--target_number', type=int, default=[3, 5, 9], metavar='T',
                    help='bags have a positive labels if they contain at least one 9')
parser.add_argument('--mean_bag_length', type=int, default=10, metavar='ML',
                    help='average bag length')
parser.add_argument('--var_bag_length', type=int, default=2, metavar='VL',
                    help='variance of bag length')
parser.add_argument('--num_bags_train', type=int, default=50, metavar='NTrain',
                    help='number of bags in training set')
parser.add_argument('--num_bags_test', type=int, default=1000, metavar='NTest',
                    help='number of bags in test set')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

print('Load Train and Test Set')
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}



def train(epoch, train_loader, model, criterion, weight_criterion, optimizer):
    model.train()
    train_loss = 0.0
    count=0
    for batch_idx, (data, label) in enumerate(train_loader):
        count +=1

        bag_label = label[0]
        bag_label = bag_label.type(torch.LongTensor)

        instance_labels = bag_label*torch.squeeze(torch.ones(data.size(0),1)).type(torch.LongTensor)


        if args.cuda:
            data = Variable(data.cuda(),requires_grad =False)
            bag_label = Variable(bag_label.cuda(),requires_grad =False)
            instance_labels = Variable(instance_labels.cuda(),requires_grad =False)
        else:
            data = Variable(data, requires_grad =False)
            bag_label = Variable(bag_label, requires_grad =False)
            instance_labels = Variable(instance_labels,requires_grad =False)


        optimizer.zero_grad()


        y, y_c, alpha = model(data)

        loss_1 = criterion(y, bag_label)
        loss_2 = weight_criterion(y_c, instance_labels, weights=alpha)

        loss = loss_1+ 2.0*loss_2

        train_loss += loss.item()

        # backward pass
        loss.backward()
        # step
        optimizer.step()

    # calculate loss and error for epoch
    train_loss /= len(train_loader)

    print('Epoch: {}, Loss: {:.4f}'.format(epoch, train_loss))


def test(test_loader, model):
    model.eval()
    correct = 0
    total = 0

    true_label = []
    predict_labels = []

    for batch_idx, (data, label) in enumerate(test_loader):
    
        bag_label = np.squeeze(label[0].numpy())
        instance_labels = np.squeeze(label[1].numpy())

        if args.cuda:
            data = Variable(data.cuda(),requires_grad =False)
        else:
            data = Variable(data, requires_grad =False)
        y,_, _ = model(data)
        _, predicted = torch.max(y.data, 1)
        
        predict_labels.append(predicted.data.cpu().numpy())
        true_label.append(bag_label)

    predict_labels = np.squeeze(predict_labels)
    true_label = np.squeeze(true_label)

    accuracy = accuracy_score(true_label, predict_labels)

    print('Test Accuracy {:.4f}'.format(accuracy))


    return accuracy



if __name__ == "__main__":
    
    num_bags_train = np.squeeze([50, 100, 150, 200])
    m = 50
    TA = np.squeeze(np.zeros((1,m)))
    for iter in range(m):
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
            print('\nGPU is ON!')

            print('Init Model')
        model = CNN()
        criterion = torch.nn.CrossEntropyLoss(size_average=True)
        weight_criterion = CE(aggregate='mean')

        if args.cuda:
            model.cuda()
            criterion.cuda()

        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)


        train_loader = data_utils.DataLoader(MnistBags(target_number=args.target_number,
                                                    mean_bag_length=args.mean_bag_length,
                                                    var_bag_length=args.var_bag_length,
                                                    num_bag=num_bags_train[0],
                                                    seed=args.seed,
                                                    train=True),
                                            batch_size=1,
                                            shuffle=True,
                                            **loader_kwargs)

        test_loader = data_utils.DataLoader(MnistBags(target_number=args.target_number,
                                                    mean_bag_length=args.mean_bag_length,
                                                    var_bag_length=args.var_bag_length,
                                                    num_bag=args.num_bags_test,
                                                    seed=args.seed,
                                                    train=False),
                                            batch_size=1,
                                            shuffle=False,
                                            **loader_kwargs)


        print('Start Training')
        Acc = np.squeeze(np.zeros((args.epochs,1)))
        for epoch in range(1, args.epochs + 1):
            train(epoch, train_loader, model, criterion,weight_criterion, optimizer)
            print('Start Testing')
            acc = test(test_loader, model)
            Acc[epoch-1] = acc
        
        TA[iter] = np.amax(Acc)
    
        io.savemat('TA_loss_multi.mat', mdict={'TA': TA})

        print("The mean and std of accuracy is:{}/{}".format(np.mean(TA, axis=1), np.std(TA, axis=1)))