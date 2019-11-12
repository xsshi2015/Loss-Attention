import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, dim=512):
        super(AttentionLayer, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, 1)

    def forward(self, features, W_1, b_1, flag):
        if flag==1:
            out_c = F.linear(features, W_1, b_1)
            out = out_c - out_c.max()
            out = out.exp()
            out = out.sum(1, keepdim=True)
            alpha = out / out.sum(0)
            

            alpha01 = features.size(0)*alpha.expand_as(features)
            context = torch.mul(features, alpha01)
        else:
            context = features
            alpha = torch.zeros(features.size(0),1)
                
        return context, out_c, torch.squeeze(alpha)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.Dropout(),
            nn.ReLU(),
        )

        self.att_layer = AttentionLayer(self.L)
        self.linear = nn.Linear(self.L*self.K, 4)
        

    def forward(self, x, flag=1):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL

        out, out_c, alpha = self.att_layer(H, self.linear.weight, self.linear.bias, flag)
        out = out.mean(0,keepdim=True)

        y = self.linear(out)
        return y, out_c, alpha
