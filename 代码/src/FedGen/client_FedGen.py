import copy

import numpy as np
import torch
from tqdm import tqdm

from src.client_base import ClientBase


class ClientGen(ClientBase):
    def __init__(self, args, client_id, trainset, testset, statistic):
        super().__init__(args, client_id, trainset, testset, statistic)
        self.generative_model=None

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9
        )

        # 统计每个类别的样本数量
        self.sample_per_class = torch.zeros(args.num_classes)
        for _, y in self.get_train_loader():
            for label in y:
                self.sample_per_class[label.item()] += 1

        self.qualified_labels = []


    def train(self):
        train_loader = self.get_train_loader()
        self.model.train()

        for epoch in range(self.local_epochs):
            local_iter = tqdm(train_loader, desc=f"client {self.client_id} epoch: {epoch}")
            for x, y in local_iter:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                loss = self.loss(self.model(x), y)

                if self.generative_model is not None:
                    labels = np.random.choice(self.qualified_labels, self.batch_size)
                    labels = torch.LongTensor(labels).to(self.device)
                    z = self.generative_model(labels)
                    loss += self.loss(self.model.head(z), labels)

                loss.backward()
                self.optimizer.step()
                local_iter.set_description(f"Gen-Client {self.client_id} epoch: {epoch} loss: {loss.item():.4f}")


