import copy

import torch
from torch import nn
from tqdm import tqdm

from src.client_base import ClientBase


class ClientDBE(ClientBase):
    def __init__(self, args, client_id, trainset, testset, statistic):
        super().__init__(args, client_id, trainset, testset, statistic)

        self.model = copy.deepcopy(args.model)

        self.kappa = args.client["kappa"]
        self.mu = args.client["mu"]

        self.local_mean = None
        self.global_mean = None  # 一旦确定，不再更新
        self.personalized_mean = torch.zeros(self.args.feature_dim, requires_grad=True, device=self.device)

        self.MSELoss = nn.MSELoss()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)  # 优化器
        self.personalized_mean_optimizer = torch.optim.SGD([self.personalized_mean], lr=self.learning_rate)

    def train(self):
        train_loader = self.get_train_loader()
        self.model.train()

        for epoch in range(self.local_epochs):
            local_iter = tqdm(train_loader, desc=f"client {self.client_id} epoch: {epoch}")
            
            self.local_mean = torch.zeros(self.args.feature_dim).to(self.device)
            for x, y in local_iter:
                x, y = x.to(self.device), y.to(self.device)

                feature = self.model.base(x)

                feature_mean = torch.mean(feature, dim=0).detach()  # 必须 detach 不然两路反向传播报错
                self.local_mean = (1 - self.mu) * self.local_mean + self.mu * feature_mean

                if self.global_mean is not None:
                    output = self.model.head(feature + self.personalized_mean)
                    loss = self.loss(output, y) + self.kappa * self.MSELoss(self.local_mean, self.global_mean)
                else:
                    output = self.model.head(feature)
                    loss = self.loss(output, y)

                self.optimizer.zero_grad()
                self.personalized_mean_optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.personalized_mean_optimizer.step()

                local_iter.set_description(f"client {self.client_id} epoch: {epoch} loss: {loss.item():.4f}")
