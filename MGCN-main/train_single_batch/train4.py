import torch
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np  
import pandas as pd  
from sklearn.cluster import KMeans  
import community as louvain  
from scipy.cluster.hierarchy import linkage, fcluster  
from sklearn.metrics import pairwise_distances   
from utils.misc import *
import os
import scanpy as sc
from train.utils import clustering
from utils.preprocess import *
from utils.utils import *
def create_adj(adata):
    cell_position_omics1 = adata.obsm['spatial']
    # adj_omics1 = construct_graph_by_radius(cell_position_omics1, initial_radius=50)
    adj_omics1 =construct_graph_by_coordinate(cell_position_omics1, n_neighbors=5)
    adata.uns['adj_spatial'] = adj_omics1
    adj_spatial_omics1 = adata.uns['adj_spatial']
    adj_spatial_omics1 = transform_adjacent_matrix(adj_spatial_omics1)
    adj_spatial_omics1 = adj_spatial_omics1.toarray()
    adj_spatial_omics1 = adj_spatial_omics1 + adj_spatial_omics1.T
    adj_spatial_omics1 = np.where(adj_spatial_omics1>1, 1, adj_spatial_omics1)
    adj = preprocess_graph(adj_spatial_omics1)
    return adj

# Calculate InfoNCE-like loss. Considering spatial neighbors as positive pairs for each spot
def Noise_Cross_Entropy(emb, adj):
    sim = cosine_sim_tensor(emb)
    sim_exp = torch.exp(sim)

    # negative pairs
    n = torch.mul(sim_exp, 1 - adj).sum(axis=1)
    # positive pairs
    p = torch.mul(sim_exp, adj).sum(axis=1)

    ave = torch.div(p, n)
    loss = -torch.log(ave).mean()

    return loss

# Calculate cosine similarity.
def cosine_sim_tensor(emb):
    M = torch.matmul(emb, emb.T)
    length = torch.norm(emb, p=2, dim=1)
    Norm = torch.matmul(length.reshape((emb.shape[0], 1)), length.reshape((emb.shape[0], 1)).T) + -5e-12      # reshape((emb.shape[0], 1))
    M = torch.div(M, Norm)
    if torch.any(torch.isnan(M)):
        M = torch.where(torch.isnan(M), torch.full_like(M, 0.4868), M)

    return M
BCE_loss = torch.nn.BCEWithLogitsLoss()
KL_loss = torch.nn.KLDivLoss(reduction='batchmean')

def Train_batch(epochs, model,dataloader ,label,args):
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    nmi_result = []
    ari_result = []
    ami_result = []
    optimizer = Adam(model.parameters(), lr=args.lr)
    ari=0
    for epoch in range(epochs):
    # 用于累积所有批次的结果
        all_fea = []
        all_labels = []
        
        for batch_data in dataloader:
            # 提取位置信息和 omic 数据
            location = batch_data[:, -2:].cpu().numpy()  # 转换为 numpy 数组
            omic_data = batch_data[:, :-2].cpu().numpy()  # 转换为 numpy 数组
            
            # 创建 AnnData 对象并构建邻接矩阵
            a = sc.AnnData(omic_data)  # 现在 omic_data 是 numpy 数组
            a.obsm['spatial'] = location
            batch_adj = create_adj(a)
            batch_adj = batch_adj.to(args.device)
            # 将 omic_data 转换回 torch.Tensor 并发送到设备
            omic_data = torch.FloatTensor(omic_data).to(args.device)
            
            # 前向传播
            x_hat, z_hat, adj_hat, z_ae, z_igae, fea = model(omic_data, batch_adj)
            
            # 计算损失
            adj_ = torch.mm(fea, fea.T)
            if adj_.is_sparse:
                adj_ = adj_.to_dense()
            adj_spatial = batch_adj
            if adj_spatial.is_sparse:
                adj_spatial = adj_spatial.to_dense()
            
            loss_adj_1 = BCE_loss(adj_, adj_spatial)
            loss_adj_2 = KL_loss(F.log_softmax(adj_spatial, dim=1) + 1e-8, adj_.softmax(dim=1) + 1e-8)
            loss_NCE = Noise_Cross_Entropy(fea, adj_spatial)
            
            loss_ae = F.mse_loss(x_hat, omic_data)
            loss_w = F.mse_loss(z_hat, torch.spmm(batch_adj, omic_data))
            loss_a = F.mse_loss(adj_hat, batch_adj.to_dense())
            loss_s = F.mse_loss(z_igae, z_ae)
            loss_igae = args.loss_w * loss_w + args.loss_a * loss_adj_1
            
            loss = loss_ae + loss_igae + args.loss_s * loss_s
            if epoch > epochs * 0.1:
                loss = loss_ae + loss_igae + args.loss_s * loss_s + args.loss_n * loss_NCE
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        #     # 累积当前批次的结果
        #     all_fea.append(fea.data.cpu().numpy())
        
        # # 在每个 epoch 结束时，对整个数据集计算指标
        # all_fea = np.concatenate(all_fea, axis=0)  # 将所有批次的特征表示拼接
        # # all_labels = np.concatenate(all_labels, axis=0)  # 将所有批次的标签拼接
        
        # # 使用 KMeans 聚类
        # kmeans = KMeans(n_clusters=args.n_clusters1, n_init=10).fit(all_fea)  # 确保 kmeans 始终被赋值
        # pred_labels = kmeans.labels_
        
        # # 计算指标
        # nmi = normalized_mutual_info_score(label, pred_labels)
        # ari = adjusted_rand_score(label, pred_labels)
        # ami = adjusted_mutual_info_score(label, pred_labels)
        
        # # 记录结果
        # nmi_result.append(nmi)
        # ari_result.append(ari)
        # ami_result.append(ami)
        
        # # 打印结果
        # if epoch % 10 == 9:
        #     print(f'Epoch {epoch + 1}:')
        #     print(f'NMI: {nmi:.4f}, ARI: {ari:.4f}, AMI: {ami:.4f}')
    return model

def Test(model,adata, dataloader, label, args,tool='kmeans'):
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    nmi_result = []
    ari_result = []
    ami_result = []
    homogeneity_result = []
    completeness_result = []
    v_measure_result = []

    with torch.no_grad():
        pred=[]
        Feat=[]
        for batch_data in dataloader:
            
            location = batch_data[:, -2:].cpu().numpy()  # 转换为 numpy 数组
            omic_data = batch_data[:, :-2].cpu().numpy()  # 转换为 numpy 数组
            a=sc.AnnData(omic_data)
            a.obsm['spatial']=location
            batch_adj=create_adj(a)
            batch_adj=batch_adj.to(args.device)
            omic_data = torch.FloatTensor(omic_data).to(args.device)
            
            x_hat, z_hat, adj_hat, z_ae, z_igae,  fea = model(omic_data, batch_adj)
            Feat.append(fea.data.cpu().numpy())
        Feat = np.concatenate(Feat, axis=0)
        adata.obsm['spaMGCN']=Feat
        if tool=='kmeans':
            from sklearn.decomposition import PCA  
            from sklearn.cluster import KMeans  
            kmeans = KMeans(n_clusters=args.n_clusters2, n_init=10).fit(Feat)  
            pred=kmeans.labels_
        elif tool == 'mclust': # mclust, leiden, and louvain
            clustering(adata, key='spaMGCN', add_key='spaMGCN', n_clusters=args.n_clusters, method=tool, use_pca=True)
            pred=adata.obs['mclust']
        elif tool == 'leiden': # mclust, leiden, and louvain
            clustering(adata, key='spaMGCN', add_key='spaMGCN', n_clusters=args.n_clusters, method=tool, use_pca=True)
            pred=adata.obs['leiden']
            print(len(pred))
            print(pred)
        elif tool == 'louvain': # mclust, leiden, and louvain
            clustering(adata, key='spaMGCN', add_key='spaMGCN', n_clusters=args.n_clusters, method=tool, use_pca=True)
            pred=adata.obs['louvain']
        adata.obsm['fea']=Feat
        adata.obs['pred']=pred
        adata.obs['pred'] = pd.Categorical(adata.obs['pred'])  
        nmi, ari, ami, homogeneity, completeness, v_measure = eva(label, adata.obs['pred'])
        nmi_result.append(nmi)
        ari_result.append(ari)
        ami_result.append(ami)
        homogeneity_result.append(homogeneity)
        completeness_result.append(completeness)
        v_measure_result.append(v_measure)
    print('聚类方法为{}'.format(tool))
    print_results(nmi_result, ari_result, ami_result, args,'test')
    return nmi, ari, ami, homogeneity, completeness, v_measure