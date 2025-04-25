import torch
from torch import nn
import torch.nn.functional as F
from model.AE import AE
from model.IGAE import IGAE
from torch.nn.parameter import Parameter
class CrossAttention(torch.nn.Module):
    def __init__(self, emb_dim=64)-> None:
        super(CrossAttention, self).__init__()
        self.query_linear = torch.nn.Linear(emb_dim, emb_dim)
        self.key_linear = torch.nn.Linear(emb_dim, emb_dim)
        self.value_linear = torch.nn.Linear(emb_dim, emb_dim)
        self.scale_factor = 1.0 / (emb_dim ** 0.5)
        # Layer Normalization
        self.layer_norm = torch.nn.LayerNorm(emb_dim)
    def forward(self, query, key, value):
        query_proj = self.query_linear(query)  # 
        key_proj = self.key_linear(key)  # 
        value_proj = self.value_linear(value)  # 

        attention_scores = torch.matmul(query_proj, key_proj.transpose(-2, -1))#T)  # 
        attention_scores = attention_scores * self.scale_factor # 
        attention_weights = torch.softmax(attention_scores, dim=-1)  # 

        attended_values = torch.matmul(attention_weights, value_proj)  # 
        attended_values = self.layer_norm(attended_values)
        return attended_values

class ZINBDecoder(torch.nn.Module):
    def __init__(self, nhid1, nfeat):
        super(ZINBDecoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid1, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.ReLU()
        )
        self.pi = torch.nn.Linear(nhid1, nfeat)
        self.disp = torch.nn.Linear(nhid1, nfeat)
        self.mean = torch.nn.Linear(nhid1,  nfeat)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)


    def forward(self, emb):
        x = self.decoder(emb)
        pi = torch.sigmoid(self.pi(x))
        disp = self.DispAct(self.disp(x))
        mean = self.MeanAct(self.mean(x))
        return [pi, disp, mean]

class MixtureNBLogNormal(torch.nn.Module):
    def __init__(self, nhid1, nfeat):
        super(MixtureNBLogNormal, self).__init__()
        # 解码器网络
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid1, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.ReLU()
        )

        # 背景和前景的神经网络参数
        self.back_mean = torch.nn.Linear(nhid1, nfeat)  # m_i^back
        self.back_log_sigma = torch.nn.Linear(nhid1, nfeat)  # log(σ_i^back)
        self.pi = torch.nn.Linear(nhid1, nfeat)  # π_i^protein
        self.alpha = torch.nn.Linear(nhid1, nfeat)  # α_i^protein
        self.back_dispersion = torch.nn.Linear(nhid1, nfeat)  # ϕ for background
        self.fore_dispersion = torch.nn.Linear(nhid1, nfeat)  # ϕ for foreground

    def forward(self, emb, y_protein):
        # 解码器生成隐藏层表示
        x = self.decoder(emb)

        # 背景强度的均值和log标准差
        m_back = self.back_mean(x)  # 背景的 m_i^back
        log_sigma_back = self.back_log_sigma(x)  # 背景的 log(σ_i^back)
        sigma_back = torch.exp(log_sigma_back)  # 背景的 σ_i^back

        # 通过log-normal分布对背景强度 ν_i^back 进行采样
        eps = torch.randn_like(m_back)  # 采样标准正态分布
        v_back = torch.exp(m_back + sigma_back * eps)  # 背景强度 ν_i^back ~ LogNormal

        # 计算前景强度
        alpha_protein = torch.exp(self.alpha(x))  # α_i^protein
        v_fore = (1 + alpha_protein) * v_back  # 前景强度 ν_i^fore

        # 零膨胀参数
        pi_protein = torch.sigmoid(self.pi(x))  # π_i^protein

        # 前景和背景的离散度
        dispersion_back = F.softplus(self.back_dispersion(x))  # 背景的 dispersion
        dispersion_fore = F.softplus(self.fore_dispersion(x))  # 前景的 dispersion

        # 负二项分布的计算（背景和前景）
        nb_back = self.negative_binomial(y_protein, v_back, dispersion_back)
        nb_fore = self.negative_binomial(y_protein, v_fore, dispersion_fore)

        # 使用更稳定的 logsumexp 计算混合负二项分布的对数似然
        mixture_nb = torch.logsumexp(torch.stack([
            torch.log(pi_protein + 1e-8) + nb_back,
            torch.log(1 - pi_protein + 1e-8) + nb_fore
        ]), dim=0)

        # 返回log likelihood的均值
        return torch.mean(mixture_nb)
    def negative_binomial(self, y, mean, dispersion):
        """计算负二项分布的log likelihood"""
        eps = 1e-8  # 防止log(0)的情况
        log_prob = (
            torch.lgamma(y + dispersion) - torch.lgamma(dispersion) - torch.lgamma(y + 1)
            + dispersion * (torch.log(dispersion + eps) - torch.log(dispersion + mean + eps))
            + y * (torch.log(mean + eps) - torch.log(dispersion + mean + eps))
        )
        return log_prob
class spaMGCN_ZINB(nn.Module):

    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3,
                 ae_n_dec_1, ae_n_dec_2, ae_n_dec_3,
                 gae_n_enc_1, gae_n_enc_2, gae_n_enc_3,
                 gae_n_dec_1, gae_n_dec_2, gae_n_dec_3,
                 n_input,n_input1, n_z, n_clusters, sigma, v=1.0, n_node=None, device=None):
        super(spaMGCN_ZINB, self).__init__()

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
        

        self.ae1 = AE(
            ae_n_enc_1=ae_n_enc_1,
            ae_n_enc_2=ae_n_enc_2,
            ae_n_enc_3=ae_n_enc_3,
            ae_n_dec_1=ae_n_dec_1,
            ae_n_dec_2=ae_n_dec_2,
            ae_n_dec_3=ae_n_dec_3,
            n_input=n_input1,
            n_z=n_z)

        self.gae1 = IGAE(
            gae_n_enc_1=gae_n_enc_1,
            gae_n_enc_2=gae_n_enc_2,
            gae_n_enc_3=gae_n_enc_3,
            gae_n_dec_1=gae_n_dec_1,
            gae_n_dec_2=gae_n_dec_2,
            gae_n_dec_3=gae_n_dec_3,
            n_input=n_input1)
        self.decoder = MixtureNBLogNormal(n_z, n_input)
        self.decoder1 = MixtureNBLogNormal(n_z, n_input1)
        self.s = nn.Sigmoid()
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, n_z), requires_grad=True)
        self.v = v
        self.sigma = sigma 
        # self.mlp=nn.Linear(n_z*2,n_z)
        # 替换现有的MLP融合
        # layers_hidden = [64, 128, 64, 32]

        # 创建模型
        
        self.mlp = nn.Sequential(
            nn.Linear(n_z*2, n_z*2),
            # nn.BatchNorm1d(n_z*2),
            nn.LeakyReLU(),
            nn.Linear(n_z*2, n_z)
        )
        self.atten_cross = AttentionLayer(n_z, n_z)

    def forward(self, X_tilde1,X_tilde2, adj1):
        sigma = self.sigma
        z_ae1, z_ae2, z_ae3 = self.ae.encoder(X_tilde1)
        z_igae1 = self.gae.encoder.gnn_1(X_tilde1, adj1)
        z_igae2 = self.gae.encoder.gnn_2((1 - sigma) * z_ae1 + sigma * z_igae1, adj1)
        z_igae3 = self.gae.encoder.gnn_3((1 - sigma) * z_ae2 + sigma * z_igae2, adj1)
        z_tilde = (1 - sigma) * z_ae3 + sigma * z_igae3

        z_igae_adj = self.s(torch.mm(z_tilde, z_tilde.t()))  

        ##组学二
        z_ae11, z_ae21, z_ae31 = self.ae1.encoder(X_tilde2)
        z_igae11 = self.gae1.encoder.gnn_1(X_tilde2, adj1)
        z_igae21 = self.gae1.encoder.gnn_2((1 - sigma) * z_ae11 + sigma * z_igae11, adj1)
        z_igae31 = self.gae1.encoder.gnn_3((1 - sigma) * z_ae21 + sigma * z_igae21, adj1)
        z_tilde1 = (1 - sigma) * z_ae31 + sigma * z_igae31
        z_igae_adj1 = self.s(torch.mm(z_tilde1, z_tilde1.t()))

        # z_tilde1 = self.ADT_in_RNA_Att(z_tilde, z_tilde1, z_tilde1)## 
        # z_tilde = self.RNA_in_ADT_Att(z_tilde1, z_tilde, z_tilde)##  
        fea=torch.cat([z_tilde,z_tilde1],dim=1)
        fea=self.mlp(fea)

        # x_hat = self.ae.decoder(fea)
        x_hat = 0
        log_likelihood = self.decoder(fea, X_tilde1)
        loss1 = -log_likelihood
        z_hat, z_hat_adj = self.gae.decoder(fea, adj1)
        adj_hat = z_igae_adj + z_hat_adj

        # x_hat1 = self.ae1.decoder(fea)
        x_hat1 = 0
        log_likelihood1 = self.decoder1(fea, X_tilde2)
        loss2 = -log_likelihood1
        z_hat1, z_hat_adj1 = self.gae1.decoder(fea, adj1)
        adj_hat1 = z_igae_adj1 + z_hat_adj1
        loss=loss1+loss2
        

        return x_hat, z_hat, adj_hat, z_ae3, z_igae3,x_hat1, z_hat1, adj_hat1, z_ae31, z_igae31,loss, fea

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
