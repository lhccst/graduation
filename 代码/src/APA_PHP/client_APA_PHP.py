import copy

import torch
from tqdm import tqdm

from src.client_base import ClientBase


class ClientAPA_PHP(ClientBase):
    def __init__(self, args, client_id, trainset, testset, statistic):
        super().__init__(args, client_id, trainset, testset, statistic)

        self.model = copy.deepcopy(args.model)
        self.previous_model = copy.deepcopy(args.model)

        self.lamb_da = args.client["lambda"]
        self.proto_loss = torch.nn.MSELoss()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)

    def train(self):
        train_loader = self.get_train_loader()
        self.model.train()
        self.previous_model.eval()

        for epoch in range(self.local_epochs):
            local_iter = tqdm(train_loader, desc=f"client {self.client_id} epoch: {epoch}")
            for x, y in local_iter:
                x, y = x.to(self.device), y.to(self.device)
                feature = self.model.base(x)
                output = self.model.head(feature)
                loss = self.loss(output, y)

                if epoch == 0:
                    previous_feature = self.previous_model.base(x)
                    loss += self.lamb_da * self.proto_loss(previous_feature, feature)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                local_iter.set_description(f"client {self.client_id} epoch: {epoch} loss: {loss.item():.4f}")
