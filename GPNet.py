import torch
from torch import nn
import torchvision
from torchvision import models
import numpy as np
import torch.nn.functional as F

from train_GPNet import args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

def mask_f(input,thre):
    zero = torch.zeros_like(input)
    input = torch.where(input < thre, zero, input)
    return input
def mask_f_b_bia(input,bia):
    #print("bia",bia)
    input_bia=input+bia
    input2=torch.clamp(input_bia,min=0,max=1)
    #input = torch.clamp(input, min=0, max=1)
    B = torch.bernoulli(input2)
    #print('B',B)
    zero = torch.zeros_like(input)
    input = torch.where(B == 0, zero, input)
    return input
def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix

class GPNet(nn.Module):
    def __init__(self):
        super(GPNet, self).__init__()

        resnet50 = models.resnet50(pretrained=True)
        layers = list(resnet50.children())[:-2]

        self.conv = nn.Sequential(*layers)
        self.avg = nn.AvgPool2d(kernel_size=14, stride=1)
        self.map1 = nn.Linear(2048 * 2, args.dimensionality)

        self.map2 = nn.Linear(args.dimensionality, 2048)
        self.fc = nn.Linear(2048, 200)
        self.drop = nn.Dropout(p=0.5)    # seem 1.fc 2.dropout
        self.sigmoid = nn.Sigmoid()

    def forward(self, images, targets=None, flag='train'):

        conv_out = self.conv(images)
        pool_out = self.avg(conv_out).squeeze()

        if args.mask=='F':pool_out2=mask_f(pool_out,args.mask_thred)
        elif args.mask=='B':pool_out2=mask_f_b_bia(pool_out,args.mask_bia)
        else: pool_out2=pool_out

        if flag == 'train':

            intra_pairs, inter_pairs,intra_labels, inter_labels = self.get_pairs(pool_out2, targets)

            features1 = torch.cat([pool_out2[intra_pairs[:, 0]], pool_out2[inter_pairs[:, 0]]], dim=0)
            features2 = torch.cat([pool_out2[intra_pairs[:, 1]], pool_out2[inter_pairs[:, 1]]], dim=0)
            labels1 = torch.cat([intra_labels[:, 0], inter_labels[:, 0]], dim=0)
            labels2 = torch.cat([intra_labels[:, 1], inter_labels[:, 1]], dim=0)


            mutual_features = torch.cat([features1, features2], dim=1)
            map1_out = self.map1(mutual_features)
            map2_out = self.drop(map1_out)
            map2_out = self.map2(map2_out)


            gate1 = torch.mul(map2_out, features1)
            gate1 = self.sigmoid(gate1)

            gate2 = torch.mul(map2_out, features2)
            gate2 = self.sigmoid(gate2)

            if args.GP:
                gate1_chunk=torch.chunk(gate1,2,dim=0)
                gate2_chunk=torch.chunk(gate2,2,dim=0)
                features1_chunk=torch.chunk(features1,2,dim=0)
                features2_chunk=torch.chunk(features2,2,dim=0)

                features1_self_0 = features1_chunk[0] - torch.mul(gate2_chunk[0], features1_chunk[0])
                features1_other_0 = features1_chunk[0] - torch.mul(gate1_chunk[0], features1_chunk[0])

                features2_self_0 = features2_chunk[0] - torch.mul(gate1_chunk[0], features2_chunk[0])
                features2_other_0 = features2_chunk[0] - torch.mul(gate2_chunk[0], features2_chunk[0])

                features1_self_1 = torch.mul(gate1_chunk[1], features1_chunk[1]) + features1_chunk[1]  # x1self=x1@g1+x1
                features1_other_1 = torch.mul(gate2_chunk[1], features1_chunk[1]) + features1_chunk[1]

                features2_self_1 = torch.mul(gate2_chunk[1], features2_chunk[1]) + features2_chunk[1]
                features2_other_1 = torch.mul(gate1_chunk[1], features2_chunk[1]) + features2_chunk[1]

                features1_self = torch.cat([features1_self_0, features1_self_1], dim=0)
                features1_other = torch.cat([features1_other_0, features1_other_1], dim=0)
                features2_self = torch.cat([features2_self_0, features2_self_1], dim=0)
                features2_other = torch.cat([features2_other_0, features2_other_1], dim=0)


                logit1_self = self.fc(self.drop(features1_self))
                logit1_other = self.fc(self.drop(features1_other))
                logit2_self = self.fc(self.drop(features2_self))
                logit2_other = self.fc(self.drop(features2_other))

            else:

                features1_self = torch.mul(gate1, features1) + features1  # x1self=x1@g1+x1
                features1_other = torch.mul(gate2, features1) + features1

                features2_self = torch.mul(gate2, features2) + features2
                features2_other = torch.mul(gate1, features2) + features2

                logit1_self = self.fc(self.drop(features1_self))  # p1self
                logit1_other = self.fc(self.drop(features1_other))
                logit2_self = self.fc(self.drop(features2_self))
                logit2_other = self.fc(self.drop(features2_other))

            return logit1_self, logit1_other, logit2_self, logit2_other, labels1, labels2,pool_out,self.sigmoid(map1_out)

        elif flag == 'val':
            return self.fc(pool_out)


    def get_pairs(self, embeddings, labels):
        distance_matrix = pdist(embeddings).detach().cpu().numpy()

        labels = labels.detach().cpu().numpy().reshape(-1,1)
        num = labels.shape[0]
        dia_inds = np.diag_indices(num)
        lb_eqs = (labels == labels.T)
        lb_eqs[dia_inds] = False
        dist_same = distance_matrix.copy()
        dist_same[lb_eqs == False] = np.inf
        intra_idxs = np.argmin(dist_same, axis=1)

        dist_diff = distance_matrix.copy()
        lb_eqs[dia_inds] = True
        dist_diff[lb_eqs == True] = np.inf
        inter_idxs = np.argmin(dist_diff, axis=1)

        intra_pairs = np.zeros([embeddings.shape[0], 2])
        inter_pairs  = np.zeros([embeddings.shape[0], 2])
        intra_labels = np.zeros([embeddings.shape[0], 2])
        inter_labels = np.zeros([embeddings.shape[0], 2])
        for i in range(embeddings.shape[0]):
            intra_labels[i, 0] = labels[i]
            intra_labels[i, 1] = labels[intra_idxs[i]]
            intra_pairs[i, 0] = i
            intra_pairs[i, 1] = intra_idxs[i]

            inter_labels[i, 0] = labels[i]
            inter_labels[i, 1] = labels[inter_idxs[i]]
            inter_pairs[i, 0] = i
            inter_pairs[i, 1] = inter_idxs[i]

        intra_labels = torch.from_numpy(intra_labels).long().to(device)
        intra_pairs = torch.from_numpy(intra_pairs).long().to(device)
        inter_labels = torch.from_numpy(inter_labels).long().to(device)
        inter_pairs = torch.from_numpy(inter_pairs).long().to(device)

        return intra_pairs, inter_pairs, intra_labels, inter_labels



















