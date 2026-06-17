# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from sklearn.cluster import KMeans
# def kmeans_loss(X, G, F):
#     return torch.norm(X - G @ F) ** 2
#
#
# def update_assignment_matrix(X, F):
#     distances = torch.cdist(X, F)
#     assignments = torch.argmin(distances, dim=1)
#     G = torch.zeros((X.size(0), F.size(0)), dtype=torch.float)
#     G[torch.arange(X.size(0)), assignments] = 1
#     return G
#
# def contrastive_loss(X, Y, tau):
#     sim_xy = torch.exp(torch.cosine_similarity(X, Y) / tau)
#     sim_xx_sum = torch.sum(torch.exp(torch.cosine_similarity(X, X) / tau) + torch.exp(torch.cosine_similarity(X, Y) / tau))
#     sim_yy_sum = torch.sum(torch.exp(torch.cosine_similarity(Y, Y) / tau) + torch.exp(torch.cosine_similarity(X, Y) / tau))
#
#     l_i = -torch.log(sim_xy / sim_xx_sum)
#     l_j = -torch.log(sim_xy / sim_yy_sum)
#     return torch.mean(l_i + l_j)
#
#
# def overall_loss(X, G, F,data_point_centers, tau, lambda_reg):
#     L1 = kmeans_loss(X, G, F)
#     L2 = contrastive_loss(X,torch.tensor(data_point_centers,dtype=float), tau)
#     return L1 + lambda_reg * L2
#
#
# class KMeansContrastiveClustering:
#     def __init__(self, n_clusters, lambda_reg=0.1, tau=0.5, lr=0.01, max_iter=100):
#         self.n_clusters = n_clusters
#         self.lambda_reg = lambda_reg
#         self.tau = tau
#         self.lr = lr
#         self.max_iter = max_iter
#
#     def fit(self, X):
#         n_samples, n_features = X.shape
#         X = torch.tensor(X, dtype=torch.float)
#
#         # 初始化聚类中心
#         kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
#         kmeans.fit(X)
#         F = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, requires_grad=True)
#
#         # 初始化分配矩阵
#         G = update_assignment_matrix(X, F)
#
#         optimizer = optim.Adam([F], lr=self.lr)
#
#         for epoch in range(self.max_iter):
#             # labels = F.predict(X)
#             # centers = F.cluster_centers_
#             # data_point_centers = centers[labels]
#
#             optimizer.zero_grad()
#             data_point_centers = F[G.argmax(dim=1)]
#             # 前向传播
#             loss = overall_loss(X, G, F,data_point_centers, self.tau, self.lambda_reg)
#
#             # 反向传播和优化
#             loss.backward()
#             optimizer.step()
#
#             # 更新分配矩阵
#             G = update_assignment_matrix(X, F)
#
#             print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
#
#         self.cluster_centers_ = F.detach().numpy()
#         self.labels_ = torch.argmax(G, dim=1).detach().numpy()
#
#
# # 示例数据生成
# from sklearn.datasets import make_blobs
#
# data, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)
#
# # 训练模型
# model = KMeansContrastiveClustering(n_clusters=3)
# model.fit(data)
#
# print("Final cluster centers:", model.cluster_centers_)
# print("Cluster assignments:", model.labels_)
#

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans

from torch.nn.functional import normalize
from src.FedCPCL.loss import InstanceLoss

class KMeansContrastiveClustering:
    def __init__(self, n_clusters, lambda_reg=0.1, tau=0.5, lr=0.01, max_iter=300):
        self.n_clusters = n_clusters
        self.lambda_reg = lambda_reg
        self.tau = tau
        self.lr = lr
        self.max_iter = max_iter

    def fit(self, X, sample_weight):
        n_samples, n_features = X.shape
        X = torch.tensor(X, dtype=torch.float)

        # 使用 K-means++ 初始化聚类中心
        kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=1, max_iter=1)
        kmeans.fit(X.numpy(),sample_weight=sample_weight)
        F = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, requires_grad=True)

        optimizer = optim.Adam([F], lr=self.lr)

        for epoch in range(self.max_iter):
            # 更新分配矩阵
            G = self.update_assignment_matrix(X, F)

            optimizer.zero_grad()

            # 根据 G 更新 data_point_centers
            data_point_centers = F[G.argmax(dim=1)]
            loss = self.overall_loss(X, G, F, data_point_centers, self.tau, self.lambda_reg)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

        self.cluster_centers_ = F.detach().numpy()
        self.labels_ = torch.argmax(G, dim=1).detach().numpy()

    def kmeans_loss(self,X, G, F):
        return torch.mean(normalize(X, dim=-1) @ normalize(G@F, dim=-1).transpose(-2, -1))


    def update_assignment_matrix(self,X, F):
        distances = torch.cdist(X, F)
        assignments = torch.argmin(distances, dim=1)
        G = torch.zeros((X.size(0), F.size(0)), dtype=torch.float)
        G[torch.arange(X.size(0)), assignments] = 1
        return G


    def contrastive_loss(self,X, Y, tau):
        # sim_xy = torch.exp(torch.cosine_similarity(X.unsqueeze(1), Y.unsqueeze(0), dim=2) / tau)
        # sim_xx = torch.exp(torch.cosine_similarity(X.unsqueeze(1), X.unsqueeze(0), dim=2) / tau)
        # sim_yy = torch.exp(torch.cosine_similarity(Y.unsqueeze(1), Y.unsqueeze(0), dim=2) / tau)
        #
        # sim_xx_sum = torch.sum(sim_xx, dim=1)
        # sim_yy_sum = torch.sum(sim_yy, dim=1)
        # sim_xy_sum = torch.sum(sim_xy, dim=1)
        #
        # l_i = -torch.log(sim_xy.diag() / (sim_xx_sum + sim_xy_sum - sim_xy.diag()))
        # l_j = -torch.log(sim_xy.diag() / (sim_yy_sum + sim_xy_sum - sim_xy.diag()))
        #
        # return torch.mean(l_i + l_j)
        batch_size = X.shape[0]

        conloss = InstanceLoss(batch_size=batch_size,temperature=tau)
        return conloss(X,Y)






    def overall_loss(self, X, G, F, data_point_centers, tau, lambda_reg):
        L1 = self.kmeans_loss(X, G, F)
        L2 = self.contrastive_loss(X, data_point_centers, tau)
        return L1 + lambda_reg * L2



# # 示例数据生成
# from sklearn.datasets import make_blobs
#
# data, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)
#
# # 训练模型
# model = KMeansContrastiveClustering(n_clusters=3)
# model.fit(data)
#
# print("Final cluster centers:", model.cluster_centers_)
# print("Cluster assignments:", model.labels_)
