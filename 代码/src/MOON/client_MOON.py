import copy

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.MOON.loss import GlobalConLoss
from src.client_base import ClientBase


class ClientMOON(ClientBase):
    def __init__(self, args, client_id, trainset, testset, statistic):
        super().__init__(args, client_id, trainset, testset, statistic)

        self.mu = args.client["mu"]

        self.model = copy.deepcopy(args.model)
        self.server_model = copy.deepcopy(args.model.base)
        self.previous_model = copy.deepcopy(args.model.base)

        self.con_loss = GlobalConLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)  # 优化器

    def train(self):
        train_loader = self.get_train_loader()
        self.model.train()

        for epoch in range(self.local_epochs):
            local_iter = tqdm(train_loader, desc=f"client {self.client_id} epoch: {epoch}")
            for x, y in local_iter:
                x, y = x.to(self.device), y.to(self.device)
                feature = self.model.base(x)
                output = self.model.head(feature)

                loss = self.loss(output, y)
                loss += self.mu * self.con_loss(F.normalize(feature, dim=1),
                                                global_protos=[
                                                    F.normalize(self.server_model(x), dim=1),
                                                    F.normalize(self.previous_model(x), dim=1)
                                                ]
                                                )  # 使用对比学习拉近全局模型，推远上一轮模型的距离

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                local_iter.set_description(f"client {self.client_id} epoch: {epoch} loss: {loss.item():.4f}")
