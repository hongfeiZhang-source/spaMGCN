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
import numpy as np
import random
from train.utils import clustering
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

def Train(epochs, model,adata, data,data1, adj, label, device,args):
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    acc_reuslt = []
    nmi_result = []
    ari_result = []
    ami_result = []
    optimizer = Adam(model.parameters(), lr=args.lr)
    # model.load_state_dict(torch.load(f'save/{args.dataset}/pretrain.pkl', map_location='cpu'))
    # with torch.no_grad():
    #     fea = model.init(data,data1, adj)
    # kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    # cluster_id = kmeans.fit_predict(fea.data.cpu().numpy())
    # model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    ari=0
    for epoch in range(epochs):
        x_hat, z_hat, adj_hat, z_ae, z_igae,x_hat1, z_hat1, adj_hat1, z_ae1, z_igae1,  fea = model(data,data1, adj)

        # tmp_q = q.data
        # p = target_distribution(tmp_q).detach()

        adj_ = torch.mm(fea, fea.T)
        if adj_.is_sparse:  
                adj_ = adj_.to_dense()  
        adj_spatial=adj
        if adj_spatial.is_sparse:  
            adj_spatial = adj_spatial.to_dense()  
        loss_adj_1 = BCE_loss(adj_, adj_spatial)
        loss_adj_2 = KL_loss(F.log_softmax(adj_spatial, dim=1) + 1e-8, adj_.softmax(dim=1) + 1e-8)
        loss_NCE = Noise_Cross_Entropy(fea, adj_spatial)

        loss_ae = F.mse_loss(x_hat, data) +F.mse_loss(x_hat1, data1)
        loss_w = F.mse_loss(z_hat, torch.spmm(adj, data)) +F.mse_loss(z_hat1, torch.spmm(adj, data1))
        loss_a = F.mse_loss(adj_hat, adj.to_dense()) +F.mse_loss(adj_hat1, adj.to_dense())
        loss_s = F.mse_loss(z_igae, z_ae)+F.mse_loss(z_igae1, z_ae1)
        loss_igae = args.loss_w * loss_w + args.loss_a * loss_adj_1
        # loss_kl = F.kl_div((q.log() + q1.log() + q2.log()) / 3, p, reduction='batchmean')
        loss = loss_ae + loss_igae + args.loss_s * loss_s
        if epoch >epochs*0.1:
            loss = loss_ae + loss_igae + args.loss_s * loss_s+args.loss_n*loss_NCE
        # +0.1*(loss_adj_1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 9:
            with torch.no_grad():
                x_hat, z_hat, adj_hat, z_ae, z_igae,x_hat1, z_hat1, adj_hat1, z_ae1, z_igae1,  fea = model(data,data1, adj)
                print('{:3d} loss: {}'.format(epoch, loss))
                # kmeans = KMeans(n_clusters=args.n_clusters1, n_init=10).fit(fea.data.cpu().numpy())
                adata.obsm['spaMGCN']=fea.data.cpu().numpy()
                if args.tool=='mclust':
                    clustering(adata, key='spaMGCN', add_key='spaMGCN', n_clusters=args.n_clusters, method=args.tool, use_pca=True)
                elif args.tool=='kmeans': 
                    kmeans = KMeans(n_clusters=args.n_clusters, n_init=10).fit(fea.data.cpu().numpy())  
                    adata.obs['spaMGCN']=kmeans.labels_
                elif args.tool == 'leiden': # mclust, leiden, and louvain
                    clustering(adata, key='spaMGCN', add_key='spaMGCN', n_clusters=args.n_clusters, method=args.tool, use_pca=True)
                nmi, ari, ami, homogeneity, completeness, v_measure = eva(label, adata.obs['spaMGCN'], epoch)
                nmi_result.append(nmi)
                ari_result.append(ari)
                ami_result.append(ami)
    print_results( nmi_result, ari_result, ami_result, args,'xunlian')
    return model


def Test(model,adata, data,data1, adj, label, device, args,tool='kmeans'):
    acc_reuslt = []
    nmi_result = []
    ari_result = []
    ami_result = []
    homogeneity_result = []
    completeness_result = []
    v_measure_result = []
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    with torch.no_grad():
        x_hat, z_hat, adj_hat, z_ae, z_igae,x_hat1, z_hat1, adj_hat1, z_ae1, z_igae1, fea = model(data,data1, adj)
        if tool=='jicheng':
              

            # Step 1: Obtain features from the model  
            x_hat, z_hat, adj_hat, z_ae, z_igae, x_hat1, z_hat1, adj_hat1, z_ae1, z_igae1, fea = model(data, data1, adj)  

            # Step 2: Define a method to perform K-Means clustering and construct a relation matrix  
            def kmeans_clustering(features, n_clusters):  
                # Perform KMeans clustering  
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)  
                labels = kmeans.fit_predict(features.data.cpu().numpy())  
                # Create a relation matrix  
                n = len(labels)  
                relation_matrix = np.zeros((n, n), dtype=int)  
                for i in range(n):  
                    for j in range(n):  
                        if labels[i] == labels[j]:  
                            relation_matrix[i, j] = 1  
                return relation_matrix  

            # Step 3: Perform clustering on each feature set  
            relation_matrices = []  

            feature_sets = [x_hat, z_hat, z_ae, z_igae, x_hat1, z_hat1, z_ae1, z_igae1, fea]  
            n_clusters = args.n_clusters  # You can adjust the number of clusters as needed  
            for features in feature_sets:  
                relation_matrix = kmeans_clustering(features, n_clusters)  
                relation_matrices.append(relation_matrix)  


            ##策略2
            # Step 4: Integrate multiple relation matrices  
            consensus_matrix = np.mean(relation_matrices, axis=0)  

            # 将共识矩阵转换为二值矩阵（通常取0.5为阈值）  
            consensus_matrix = (consensus_matrix >= 0.5).astype(int)  

            # 步骤3: 使用层次聚类  
            # 进行层次聚类（使用聚类方法，可以选择链接方式，比如‘ward’）  
            Z = linkage(1 - consensus_matrix, method='ward')  # 使用距离矩阵  

            # 決定聚类的数量，假设我们预期聚成3类  
            num_clusters = args.n_clusters 
            pred = fcluster(Z, num_clusters, criterion='maxclust')  

            ##策略1
            # combined_relation_matrix = np.any(relation_matrices, axis=0).astype(int)  

            # # Step 5: Apply Louvain algorithm on the combined relation matrix  
            # # Convert to a graph structure for the Louvain algorithm  
            # import networkx as nx  

            # G = nx.from_numpy_array(combined_relation_matrix)  
            # partition = louvain.best_partition(G)  

            # # Step 6: Output the cluster assignment  
            # pred = pd.Series(partition).value_counts()  

        if tool=='kmeans':
            from sklearn.decomposition import PCA  
            from sklearn.cluster import KMeans  

            # # 假设 fea 是你的特征数据，这里对特征进行降维  
            # n_components = fea.shape[1]-5  # 根据需要设置降维后的维度  
            # pca = PCA(n_components=n_components)  
            # fea_pca = pca.fit_transform(fea.data.cpu().numpy())  

            # 使用降维后的特征进行 KMeans 聚类  
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=10).fit(fea.data.cpu().numpy())  
            pred = kmeans.labels_  
            # kmeans = KMeans(n_clusters=args.n_clusters, n_init=10).fit(fea.data.cpu().numpy())
            # pred=kmeans.labels_
        elif tool=='Spectral':
            from sklearn.cluster import SpectralClustering
            spectral = SpectralClustering(n_clusters=args.n_clusters)
            pred = spectral.fit_predict(fea.data.cpu().numpy())
        elif tool == 'mclust': # mclust, leiden, and louvain
            adata.obsm['MGCN']=fea.data.cpu().numpy()
            clustering(adata, key='MGCN', add_key='SpatialGlue', n_clusters=args.n_clusters, method=tool, use_pca=True)
            pred=adata.obs['mclust']
        elif tool == 'leiden': # mclust, leiden, and louvain
            adata.obsm['MGCN']=fea.data.cpu().numpy()
            clustering(adata, key='MGCN', add_key='SpatialGlue', n_clusters=args.n_clusters, method=tool, use_pca=True)
            pred=adata.obs['leiden']
        elif tool == 'louvain': # mclust, leiden, and louvain
            adata.obsm['MGCN']=fea.data.cpu().numpy()
            clustering(adata, key='MGCN', add_key='SpatialGlue', n_clusters=args.n_clusters, method=tool, use_pca=True)
            pred=adata.obs['leiden']
        adata.obs['pred']=pred
        adata.obs['pred'] = pd.Categorical(adata.obs['pred'])  
        nmi, ari, ami, homogeneity, completeness, v_measure = eva(label, pred)
        nmi_result.append(nmi)
        ari_result.append(ari)
        ami_result.append(ami)
        homogeneity_result.append(homogeneity)
        completeness_result.append(completeness)
        v_measure_result.append(v_measure)
    print('聚类方法为{}'.format(tool))
    print_results(nmi_result, ari_result, ami_result, args,'test')
    return  nmi, ari, ami, homogeneity, completeness, v_measure