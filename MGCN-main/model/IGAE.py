import torch
from torch import nn
from torch.nn import Module, Parameter

# class NGNNLayer(Module):
#     def __init__(self, in_features, out_features, reduction=4, order=4):
#         super(NGNNLayer, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.order = order
#         self.main_layer = [NGNN(self.in_features, self.out_features, i) for i
#                             in range(1, self.order + 1)]
#         self.main_layers = torch.nn.ModuleList(self.main_layer)

#         self.fc1 = nn.Linear(out_features, out_features // reduction)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(out_features // reduction, out_features)

#         self.softmax = nn.Softmax(dim=0)
#     def forward(self, features, adj, active=True):

#         abstract_features = [self.main_layers[i](features, adj, active=active) for i in range(self.order)]
#         feats_mean = [torch.mean(abstract_features[i], 0, keepdim=True) for i in range(self.order)]
#         feats_mean = torch.cat(feats_mean, dim=0)
#         feats_a = self.fc2(self.relu(self.fc1(feats_mean)))
#         feats_a = self.softmax(feats_a)

#         feats = []
#         for i in range(self.order):
#             feats.append(abstract_features[i] * feats_a[i])

#         output = feats[0]
#         for i in range(1, self.order):
#             output += feats[i]

#         return output

# class NGNN(Module):
#     def __init__(self, in_features, out_features, order=1):
#         super(NGNN, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.order = order
#         self.weight = Parameter(torch.FloatTensor(in_features, out_features))
#         self.act = nn.Tanh()        #elu, prelu
#         torch.nn.init.xavier_uniform_(self.weight)

#     def forward(self, features, adj, active=True):
#         output = torch.mm(features, self.weight)
#         if active:
#             output = self.act(output)

#         output = torch.spmm(adj, output)
#         for _ in range(self.order-1):
#             output = torch.spmm(adj, output)
#         return output
# class NGNNLayer(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(NGNNLayer, self).__init__()
#         # 线性变换权重
#         self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
#         # 初始化权重
#         nn.init.xavier_uniform_(self.weight)

#     def forward(self, x, adj):
#         """
#         前向传播
#         :param x: 节点特征矩阵 (num_nodes, input_dim)
#         :param adj: 邻接矩阵 (num_nodes, num_nodes)
#         :return: 输出特征矩阵 (num_nodes, output_dim)
#         """
#         # 图卷积操作: A * X * W
#         support = torch.mm(x, self.weight)  # X * W
#         output = torch.mm(adj, support)    # A * (X * W)
#         return output
import torch  
import torch.nn as nn  
import torch.nn.functional as F  

class NGNNLayer(nn.Module):  
    """  
    一个简单的图卷积层（Graph Convolutional Layer）。  
    输入：  
        - input_features: 输入节点特征矩阵 (N, in_features)  
        - adj: 图的邻接矩阵 (N, N)，已经加了自环且归一化  
    输出：  
        - 输出节点特征矩阵 (N, out_features)  
    """  
    def __init__(self, in_features, out_features, bias=True):  
        super(NGNNLayer, self).__init__()  
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))  
        if bias:  
            self.bias = nn.Parameter(torch.FloatTensor(out_features))  
        else:  
            self.register_parameter('bias', None)  
        self.reset_parameters()  

    def reset_parameters(self):  
        nn.init.xavier_uniform_(self.weight)  
        if self.bias is not None:  
            nn.init.zeros_(self.bias)  

    def forward(self, input_features, adj):  
        support = torch.matmul(input_features, self.weight)  # (N, out_features)  
        output = torch.matmul(adj, support)                # (N, out_features)  
        if self.bias is not None:  
            output = output + self.bias  
        return output  
class IGAE_encoder(nn.Module):

    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, n_input):
        super(IGAE_encoder, self).__init__()
        self.gnn_1 = NGNNLayer(n_input, gae_n_enc_1,bias=True)
        self.gnn_2 = NGNNLayer(gae_n_enc_1, gae_n_enc_2,bias=True)
        self.gnn_3 = NGNNLayer(gae_n_enc_2, gae_n_enc_3,bias=False)
        self.s = nn.Sigmoid()

    def forward(self, x, adj):
        z = self.gnn_1(x, adj)
        z = self.gnn_2(z, adj)
        z_igae = self.gnn_3(z, adj)
        z_igae_adj = self.s(torch.mm(z_igae, z_igae.t()))
        return z_igae, z_igae_adj


class IGAE_decoder(nn.Module):

    def __init__(self, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super(IGAE_decoder, self).__init__()
        self.gnn_4 = NGNNLayer(gae_n_dec_1, gae_n_dec_2,bias=False)
        self.gnn_5 = NGNNLayer(gae_n_dec_2, gae_n_dec_3,bias=True)
        self.gnn_6 = NGNNLayer(gae_n_dec_3, n_input,bias=True)

        self.s = nn.Sigmoid()

    def forward(self, z_igae, adj):
        z = self.gnn_4(z_igae, adj)
        z = self.gnn_5(z, adj)
        z_hat = self.gnn_6(z, adj)

        z_hat_adj = self.s(torch.mm(z_hat, z_hat.t()))
        return z_hat, z_hat_adj

class IGAE(nn.Module):

    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super(IGAE, self).__init__()
        self.encoder = IGAE_encoder(
            gae_n_enc_1=gae_n_enc_1,
            gae_n_enc_2=gae_n_enc_2,
            gae_n_enc_3=gae_n_enc_3,
            n_input=n_input)

        self.decoder = IGAE_decoder(
            gae_n_dec_1=gae_n_dec_1,
            gae_n_dec_2=gae_n_dec_2,
            gae_n_dec_3=gae_n_dec_3,
            n_input=n_input)

    def forward(self, x, adj):
        z_igae, z_igae_adj = self.encoder(x, adj)
        z_hat, z_hat_adj = self.decoder(z_igae, adj)
        adj_hat = z_igae_adj + z_hat_adj
        return z_igae, z_hat, adj_hat
