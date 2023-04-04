import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp, global_mean_pool as m_gmp
import numpy as np
from collections import Counter
import pandas as pd

# GCN based model
class MPFFPSDC(torch.nn.Module):
    def __init__(self, n_output=2, n_filters=32, embed_dim=128, num_features_xd=78, num_features_xt=954, output_dim=128,
                 dropout=0.2,file=None):
        super(MPFFPSDC, self).__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_dim
        # SMILES1 graph branch
        self.n_output = n_output
        self.head = 4
        self.drug1_conv1 = GCNConv(num_features_xd, output_dim)
        self.drug1_conv2 = GCNConv(output_dim, output_dim)
        self.drug1_conv3 = GCNConv(output_dim, output_dim * 2)
        self.drug1_fc_g1 = torch.nn.Linear(output_dim * 2, output_dim * 2)
        self.drug1_fc_g2 = torch.nn.Linear(output_dim * 2, output_dim)

        # SMILES2 graph branch
        self.drug2_conv1 = GCNConv(num_features_xd, output_dim)
        self.drug2_conv2 = GCNConv(output_dim, output_dim * 2)
        self.drug2_conv3 = GCNConv(output_dim, output_dim * 2)
        self.drug2_fc_g1 = torch.nn.Linear(n_output * 4, n_output * 2)
        self.drug2_fc_g2 = torch.nn.Linear(n_output * 2, output_dim)

        # DL cell featrues
        self.reduction = nn.Sequential(
            nn.Linear(num_features_xt, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim * 2)
        )

        # combined layers
        self.fc1 = nn.Linear(4 * output_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.out = nn.Linear(512, self.n_output)

        # attention
        self.num_features_xd = num_features_xd
        self.w_q = nn.Linear(output_dim * 2, output_dim * 2, bias=False)
        self.w_k = nn.Linear(output_dim * 2, output_dim * 2, bias=False)
        self.w_v = nn.Linear(output_dim * 2, output_dim * 2, bias=False)
        self.w_k1 = nn.Linear(output_dim * 2,output_dim*2)
        self.batch1dim = nn.BatchNorm1d(output_dim)
        self.batch2dim = nn.BatchNorm1d(output_dim * 2)
        self.batchhdim = nn.BatchNorm1d(output_dim * 4)

    def forward(self, data1, data2):
        x1, edge_index1, batch1, cell = data1.x, data1.edge_index, data1.batch, data1.cell
        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch

        # deal drug1
        x1_gcn = self.drug1_conv1(x1, edge_index1)
        x1_gcn = self.relu(x1_gcn)
        x1_2gcn = self.drug1_conv2(x1_gcn, edge_index1)
        x1_2gcn = self.relu(x1_2gcn)
        x1_2gcn = self.drug1_conv3(x1_2gcn, edge_index1)
        x1_2gcn = self.batch2dim(x1_2gcn)
        x1_2gcn = self.relu(x1_2gcn)
        x1 = x1_2gcn


        # deal drug2
        x2_gcn = self.drug1_conv1(x2, edge_index2)
        x2_gcn = self.relu(x2_gcn)
        x2_2gcn = self.drug1_conv2(x2_gcn, edge_index2)
        x2_2gcn = self.relu(x2_2gcn)
        x2_2gcn = self.drug1_conv3(x2_2gcn, edge_index2)
        x2_2gcn = self.batch2dim(x2_2gcn)
        x2_2gcn = self.relu(x2_2gcn)
        x2 = x2_2gcn


        # deal cell
        cell_vector = F.normalize(cell, 2, 1)
        cell_vector = self.reduction(cell_vector)

        def get_batch_num(batch_num):
            batch_num = batch_num.cpu().numpy()
            batch_num = Counter(batch_num)
            return batch_num

        batch_num = get_batch_num(batch1)
        batch_num1 = get_batch_num(batch2)

        def padd(x, batch_num):
            end = 0
            maxlen = max(batch_num.values())
            l = []
            for i in range(len(batch_num)):
                start = end
                end = start + batch_num[i]
                l1 = x[start:end]
                pad = torch.nn.ZeroPad2d(padding=(0, 0, 0, maxlen - batch_num[i]))
                l1 = pad(l1)
                l1 = l1.unsqueeze(0)
                l.append(l1)
            l2 = torch.cat(l, dim=0)
            return l2

        def attention(Q, K, V):
            scores = torch.matmul(Q, K) / np.sqrt(self.output_dim * 2)
            #

            scores = scores.reshape(scores.shape[0], scores.shape[1], 1, -1)
            # scores = nn.Softmax(dim=-1)(scores).transpose(-1,-2)
            scores = nn.Softmax(dim=-1)(scores)
            context = torch.matmul(scores, V).view(scores.shape[0], scores.shape[1], -1)

            return context,scores

        # cell attention drugA
        A = gmp(x1, batch1)
        XA = padd(x1, batch_num)  # padding layer
        Q = self.w_q(cell_vector).view(-1, self.head, 64).transpose(0, 1)  # attention Q
        KA = self.w_k(XA).view(cell_vector.shape[0], -1, self.head, 64).transpose(1, 2)
        VA = self.w_v(XA).view(cell_vector.shape[0], -1, self.head, 64).transpose(1, 2)
        Q = Q.unsqueeze(-1).transpose(0, 1)
        QA = Q
        attena,scoresA = attention(KA, QA, VA)
        attena = attena.transpose(1, 2).contiguous().view(cell_vector.shape[0], -1)
        attena = attena + A
        attena = self.drug1_fc_g1(attena)

        # cell attention drugA
        B = gmp(x2, batch2)
        XB = padd(x2, batch_num1)
        KB = self.w_k(XB).view(cell_vector.shape[0], -1, self.head, 64).transpose(1, 2)
        VB = self.w_v(XB).view(cell_vector.shape[0], -1, self.head, 64).transpose(1, 2)
        QB = Q
        attenb,scoresB = attention(KB, QB, VB)
        attenb = attenb.transpose(1, 2).contiguous().view(cell_vector.shape[0], -1)
        attenb = attenb + B
        attenb = self.drug1_fc_g1(attenb)

        #FUSION
        drug = attena + attenb
        xc = torch.cat((drug, cell_vector), 1)
        xc = F.normalize(xc,2,1)
        xc = self.dropout(xc)
        xc = self.batchhdim(xc)
        xc = self.relu(xc)

        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        # xc = self.fc2(xc)
        # xc = self.relu(xc)
        # xc = self.dropout(xc)
        out = self.out(xc)
        return out
