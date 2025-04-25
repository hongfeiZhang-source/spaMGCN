import torch
from torch import nn
import torch.nn.functional as F
from model_single.AE import AE
from model_single.IGAE import IGAE
from torch.nn.parameter import Parameter

class AttentionFusionModule(nn.Module):
    def __init__(self, feature_dim, feature_num=2):
        super(AttentionFusionModule, self).__init__()
        self.attention_network = nn.Sequential(
            nn.Linear(feature_num * feature_dim, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, feature_num),
            nn.Softmax(dim=-1)
        )
        self.feature_num = feature_num

    def forward(self, z: list):
        combined = torch.cat(z, dim=1)
        attention_weights = self.attention_network(combined)
        weighted_z = []
        for i in range(0, self.feature_num):
            weighted_z_i = attention_weights[:, i:i + 1] * z[i]
            weighted_z.append(weighted_z_i)

        fused_z = sum(weighted_z)

        return fused_z
class spaMGCN(nn.Module):

    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3,
                 ae_n_dec_1, ae_n_dec_2, ae_n_dec_3,
                 gae_n_enc_1, gae_n_enc_2, gae_n_enc_3,
                 gae_n_dec_1, gae_n_dec_2, gae_n_dec_3,
                 n_input, n_z, n_clusters, sigma, v=1.0, n_node=None, device=None):
        super(spaMGCN, self).__init__()

        self.ae = AE(
            ae_n_enc_1=ae_n_enc_1,
            ae_n_enc_2=ae_n_enc_2,
            ae_n_enc_3=ae_n_enc_3,
            ae_n_dec_1=ae_n_dec_1,
            ae_n_dec_2=ae_n_dec_2,
            ae_n_dec_3=ae_n_dec_3,
            n_input=n_input,
            n_z=n_z)

        self.gae = IGAE(
            gae_n_enc_1=gae_n_enc_1,
            gae_n_enc_2=gae_n_enc_2,
            gae_n_enc_3=gae_n_enc_3,
            gae_n_dec_1=gae_n_dec_1,
            gae_n_dec_2=gae_n_dec_2,
            gae_n_dec_3=gae_n_dec_3,
            n_input=n_input)
        

        # self.ae1 = AE(
        #     ae_n_enc_1=ae_n_enc_1,
        #     ae_n_enc_2=ae_n_enc_2,
        #     ae_n_enc_3=ae_n_enc_3,
        #     ae_n_dec_1=ae_n_dec_1,
        #     ae_n_dec_2=ae_n_dec_2,
        #     ae_n_dec_3=ae_n_dec_3,
        #     n_input=n_input1,
        #     n_z=n_z)

        # self.gae1 = IGAE(
        #     gae_n_enc_1=gae_n_enc_1,
        #     gae_n_enc_2=gae_n_enc_2,
        #     gae_n_enc_3=gae_n_enc_3,
        #     gae_n_dec_1=gae_n_dec_1,
        #     gae_n_dec_2=gae_n_dec_2,
        #     gae_n_dec_3=gae_n_dec_3,
        #     n_input=n_input1)

        self.s = nn.Sigmoid()
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, n_z), requires_grad=True)
        self.v = v
        self.sigma = sigma 
        # self.atten_omics1 = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics1)
        # self.atten_cross = AttentionLayer(n_z, n_z)

        # self.attention_fusion = AttentionFusionModule(n_z, feature_num=4)
        # self.attention_fusion_x1 = AttentionFusionModule(n_z, feature_num=2)
        # self.attention_fusion_x2 = AttentionFusionModule(n_z, feature_num=2)
    def init(self, X_tilde1, adj1):
        sigma = self.sigma
        z_ae1, z_ae2, z_ae3 = self.ae.encoder(X_tilde1)
        z_igae1 = self.gae.encoder.gnn_1(X_tilde1, adj1)
        z_igae2 = self.gae.encoder.gnn_2((1 - sigma) * z_ae1 + sigma * z_igae1, adj1)
        z_igae3 = self.gae.encoder.gnn_3((1 - sigma) * z_ae2 + sigma * z_igae2, adj1, active=False)
        z_tilde = (1 - sigma) * z_ae3 + sigma * z_igae3

        
        z_igae_adj = self.s(torch.mm(z_tilde, z_tilde.t()))

        

        # ##组学二
        # z_ae11, z_ae21, z_ae31 = self.ae1.encoder(X_tilde2)
        # z_igae11 = self.gae1.encoder.gnn_1(X_tilde2, adj1)
        # z_igae21 = self.gae1.encoder.gnn_2((1 - sigma) * z_ae11 + sigma * z_igae11, adj1)
        # z_igae31 = self.gae1.encoder.gnn_3((1 - sigma) * z_ae21 + sigma * z_igae21, adj1, active=False)
        # z_tilde1 = (1 - sigma) * z_ae31 + sigma * z_igae31
        # z_igae_adj1 = self.s(torch.mm(z_tilde1, z_tilde1.t()))

        
        x_hat = self.ae.decoder(z_ae3)
        z_hat, z_hat_adj = self.gae.decoder(z_tilde, adj1)
        adj_hat = z_igae_adj + z_hat_adj

        # x_hat1 = self.ae1.decoder(z_ae31)
        # z_hat1, z_hat_adj1 = self.gae1.decoder(z_tilde1, adj1)
        # adj_hat1 = z_igae_adj1 + z_hat_adj1

        
        # fea,a=self.atten_cross(z_tilde,z_tilde1)        
        fea=z_tilde
        # fea1=torch.cat([z_ae3,z_ae31],dim=1)
        # fea2=torch.cat([z_igae3,z_igae31],dim=1)

        # q = 1.0 / (1.0 + torch.sum(torch.pow((fea).unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        # q = q.pow((self.v + 1.0) / 2.0)
        # q = (q.t() / torch.sum(q, 1)).t()

        # q1 = 1.0 / (1.0 + torch.sum(torch.pow(z_ae3.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        # q1 = q1.pow((self.v + 1.0) / 2.0)
        # q1 = (q1.t() / torch.sum(q1, 1)).t()

        # q2 = 1.0 / (1.0 + torch.sum(torch.pow(z_igae3.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        # q2 = q2.pow((self.v + 1.0) / 2.0)
        # q2 = (q2.t() / torch.sum(q2, 1)).t()

        # q3 = 1.0 / (1.0 + torch.sum(torch.pow(z_ae31.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        # q3 = q3.pow((self.v + 1.0) / 2.0)
        # q3 = (q3.t() / torch.sum(q3, 1)).t()

        # q4 = 1.0 / (1.0 + torch.sum(torch.pow(z_igae31.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        # q4 = q4.pow((self.v + 1.0) / 2.0)
        # q4 = (q4.t() / torch.sum(q4, 1)).t()
        

        return  fea

    def forward(self, X_tilde1, adj1):
        sigma = self.sigma
        z_ae1, z_ae2, z_ae3 = self.ae.encoder(X_tilde1)
        z_igae1 = self.gae.encoder.gnn_1(X_tilde1, adj1)
        z_igae2 = self.gae.encoder.gnn_2((1 - sigma) * z_ae1 + sigma * z_igae1, adj1)
        z_igae3 = self.gae.encoder.gnn_3((1 - sigma) * z_ae2 + sigma * z_igae2, adj1, active=False)
        z_tilde = (1 - sigma) * z_ae3 + sigma * z_igae3
        z_igae_adj = self.s(torch.mm(z_tilde, z_tilde.t()))  

        # ##组学二
        # z_ae11, z_ae21, z_ae31 = self.ae1.encoder(X_tilde2)
        # z_igae11 = self.gae1.encoder.gnn_1(X_tilde2, adj1)
        # z_igae21 = self.gae1.encoder.gnn_2((1 - sigma) * z_ae11 + sigma * z_igae11, adj1)
        # z_igae31 = self.gae1.encoder.gnn_3((1 - sigma) * z_ae21 + sigma * z_igae21, adj1, active=False)
        # z_tilde1 = (1 - sigma) * z_ae31 + sigma * z_igae31
        # z_igae_adj1 = self.s(torch.mm(z_tilde1, z_tilde1.t()))

        
        x_hat = self.ae.decoder(z_ae3)
        z_hat, z_hat_adj = self.gae.decoder(z_tilde, adj1)
        adj_hat = z_igae_adj + z_hat_adj

        # x_hat1 = self.ae1.decoder(z_ae31)
        # z_hat1, z_hat_adj1 = self.gae1.decoder(z_tilde1, adj1)
        # adj_hat1 = z_igae_adj1 + z_hat_adj1

        
        # fea,a=self.atten_cross(z_tilde,z_tilde1)        
        fea=z_tilde
        # fea1=torch.cat([z_ae3,z_ae31],dim=1)
        # fea2=torch.cat([z_igae3,z_igae31],dim=1)

        # q = 1.0 / (1.0 + torch.sum(torch.pow((fea).unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        # q = q.pow((self.v + 1.0) / 2.0)
        # q = (q.t() / torch.sum(q, 1)).t()

        # q1 = 1.0 / (1.0 + torch.sum(torch.pow(z_ae3.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        # q1 = q1.pow((self.v + 1.0) / 2.0)
        # q1 = (q1.t() / torch.sum(q1, 1)).t()

        # q2 = 1.0 / (1.0 + torch.sum(torch.pow(z_igae3.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        # q2 = q2.pow((self.v + 1.0) / 2.0)
        # q2 = (q2.t() / torch.sum(q2, 1)).t()

        # q3 = 1.0 / (1.0 + torch.sum(torch.pow(z_ae31.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        # q3 = q3.pow((self.v + 1.0) / 2.0)
        # q3 = (q3.t() / torch.sum(q3, 1)).t()

        # q4 = 1.0 / (1.0 + torch.sum(torch.pow(z_igae31.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        # q4 = q4.pow((self.v + 1.0) / 2.0)
        # q4 = (q4.t() / torch.sum(q4, 1)).t()

        

        return x_hat, z_hat, adj_hat, z_ae3, z_igae3, fea
    # def forward(self, X_tilde1, adj1):
    #     sigma = self.sigma
    #     z_ae1, z_ae2, z_ae3 = self.ae.encoder(X_tilde1)
    #     z_igae1 = self.gae.encoder.gnn_1(X_tilde1, adj1)
    #     z_igae2 = self.gae.encoder.gnn_2((1 - sigma) * z_ae1 + sigma * z_igae1, adj1)
    #     z_igae3 = self.gae.encoder.gnn_3((1 - sigma) * z_ae2 + sigma * z_igae2, adj1, active=False)
    #     z_tilde = (1 - sigma) * z_ae3 + sigma * z_igae3
    #     z_igae_adj = self.s(torch.mm(z_igae3, z_igae3.t()))
    #     x_hat = self.ae.decoder(z_ae3)
    #     z_hat, z_hat_adj = self.gae.decoder(z_tilde, adj1)
    #     adj_hat = z_igae_adj + z_hat_adj

    #     q = 1.0 / (1.0 + torch.sum(torch.pow(z_tilde.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
    #     q = q.pow((self.v + 1.0) / 2.0)
    #     q = (q.t() / torch.sum(q, 1)).t()

    #     q1 = 1.0 / (1.0 + torch.sum(torch.pow(z_ae3.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
    #     q1 = q1.pow((self.v + 1.0) / 2.0)
    #     q1 = (q1.t() / torch.sum(q1, 1)).t()

    #     q2 = 1.0 / (1.0 + torch.sum(torch.pow(z_igae3.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
    #     q2 = q2.pow((self.v + 1.0) / 2.0)
    #     q2 = (q2.t() / torch.sum(q2, 1)).t()

    #     return x_hat, z_hat, adj_hat, z_ae3, z_igae3, q, q1, q2, z_tilde
class AttentionLayer(nn.Module):
    
    """\
    Attention layer.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features.     

    Returns
    -------
    Aggregated representations and modality weights.

    """
    
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)
        
    def forward(self, emb1, emb2):
        emb = []
        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
        self.emb = torch.cat(emb, dim=1)
        
        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu=  torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6)  
        
        emb_combined = torch.matmul(torch.transpose(self.emb,1,2), torch.unsqueeze(self.alpha, -1))
    
        return torch.squeeze(emb_combined), self.alpha      
