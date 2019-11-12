"""Pytorch dataset object that loads MNIST dataset as bags."""

import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms


class MnistBags(data_utils.Dataset):
    def __init__(self, target_number=9, mean_bag_length=10, var_bag_length=2, num_bag=250, seed=1, train=True):
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.train = train

        self.r = np.random.RandomState(seed)

        self.num_in_train = 60000
        self.num_in_test = 10000

        if self.train:
            self.train_bags_list, self.train_labels_list = self._create_bags()
        else:
            self.test_bags_list, self.test_labels_list = self._create_bags()

    def _create_bags(self):
        if self.train:
            loader = data_utils.DataLoader(datasets.MNIST('../datasets',
                                                          train=True,
                                                          download=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=self.num_in_train,
                                           shuffle=False)
        else:
            loader = data_utils.DataLoader(datasets.MNIST('../datasets',
                                                          train=False,
                                                          download=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=self.num_in_test,
                                           shuffle=False)

        for (batch_data, batch_labels) in loader:
            all_imgs = batch_data
            all_labels = np.squeeze(batch_labels.numpy())

        bags_list = []
        labels_list = []

        for i in range(self.num_bag):
            bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
            # print(bag_length)

            if bag_length < 1:
                bag_length = 1
            target_num = np.squeeze(np.random.choice(np.squeeze(self.target_number), 2, replace=False))
            r_target_num = np.squeeze(np.setdiff1d(np.squeeze(self.target_number), target_num))
            class_num = np.squeeze(np.where(np.squeeze(self.target_number)==r_target_num))
            if self.train:
                r_index = np.squeeze(np.arange(all_labels.shape[0]))
                for iter in range(target_num.size):
                    r_idx = np.squeeze(np.where(all_labels[r_index]==target_num[iter]))
                    r_index = np.squeeze(np.setdiff1d(r_index, r_index[r_idx]))

                indices = np.squeeze(np.random.choice(r_index, bag_length, replace=False))
            else:
                r_index = np.squeeze(np.arange(all_labels.shape[0]))
                for iter in range(target_num.size):
                    r_idx = np.squeeze(np.where(all_labels[r_index]==target_num[iter]))
                    r_index = np.squeeze(np.setdiff1d(r_index, r_index[r_idx]))
                    
                indices = np.squeeze(np.random.choice(r_index, bag_length, replace=False))

            labels_in_bag = all_labels[indices]

            labels_in_bag = torch.from_numpy(labels_in_bag).type(torch.LongTensor)
            r_target_num = torch.from_numpy(r_target_num).type(torch.LongTensor)
            class_num = torch.from_numpy(class_num).type(torch.LongTensor)
            
            labels_in_bag[labels_in_bag != r_target_num] = 0
            labels_in_bag[labels_in_bag == r_target_num] = class_num+1

            bags_list.append(all_imgs[indices])
            labels_list.append(labels_in_bag)

        return bags_list, labels_list

    def __len__(self):
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            label = [max(self.train_labels_list[index]), self.train_labels_list[index]]
        else:
            bag = self.test_bags_list[index]
            label = [max(self.test_labels_list[index]), self.test_labels_list[index]]

        return bag, label


if __name__ == "__main__":

    train_loader = data_utils.DataLoader(MnistBags(target_number=9,
                                                   mean_bag_length=10,
                                                   var_bag_length=2,
                                                   num_bag=100,
                                                   seed=1,
                                                   train=True),
                                         batch_size=1,
                                         shuffle=True)

    test_loader = data_utils.DataLoader(MnistBags(target_number=9,
                                                  mean_bag_length=10,
                                                  var_bag_length=2,
                                                  num_bag=100,
                                                  seed=1,
                                                  train=False),
                                        batch_size=1,
                                        shuffle=False)

    len_bag_list_train = []
    mnist_bags_train = 0
    for batch_idx, (bag, label) in enumerate(train_loader):
        len_bag_list_train.append(int(bag.squeeze(0).size()[0]))
        mnist_bags_train += label[0].numpy()[0]
    print('Number positive train bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
        mnist_bags_train, len(train_loader),
        np.mean(len_bag_list_train), np.min(len_bag_list_train), np.max(len_bag_list_train)))

    len_bag_list_test = []
    mnist_bags_test = 0
    for batch_idx, (bag, label) in enumerate(test_loader):
        len_bag_list_test.append(int(bag.squeeze(0).size()[0]))
        mnist_bags_test += label[0].numpy()[0]
    print('Number positive test bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
        mnist_bags_test, len(test_loader),
        np.mean(len_bag_list_test), np.min(len_bag_list_test), np.max(len_bag_list_test)))