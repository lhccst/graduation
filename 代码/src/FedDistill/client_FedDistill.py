import copy
from collections import defaultdict

import torch
from torch import nn
from tqdm import tqdm

from src.client_base import ClientBase


class ClientFedDistill(ClientBase):
    def __init__(self, args, client_id, trainset, testset, statistic):
        super().__init__(args, client_id, trainset, testset, statistic)

        # self.model = copy.deepcopy(args.model)

        self.lamb_da = args.client["lambda"]

        self.local_logits = {}  # dict
        self.global_logits = None  # list

        self.logit_loss = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)

    def aggregate(self, raw_logits):
        for label in raw_logits.keys():
            logits = torch.stack(raw_logits[label], dim=0)
            self.local_logits[label] = torch.mean(logits, dim=0)

    def collect_logits(self):
        train_loader = self.get_train_loader()
        self.model.eval()

        raw_logits = defaultdict(list)
        with torch.inference_mode():
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)

                for label in torch.unique(y):
                    logits = output[label == y].detach()
                    raw_logits[label.item()].extend(list(torch.unbind(logits, dim=0)))

        return raw_logits

    def train(self):
        train_loader = self.get_train_loader()
        self.model.train()

        for epoch in range(self.local_epochs):
            local_iter = tqdm(train_loader, desc=f"client {self.client_id} epoch: {epoch}")
            for x, y in local_iter:
                x, y = x.to(self.device), y.to(self.device)

                output = self.model(x)
                loss = self.loss(output, y)

                if self.global_logits is not None:
                    global_logits = torch.stack(self.global_logits, dim=0)
                    loss += self.lamb_da * self.logit_loss(output, global_logits[y])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                local_iter.set_description(f"client {self.client_id} epoch: {epoch} loss: {loss.item():.4f}")

        raw_logits = self.collect_logits()
        self.aggregate(raw_logits)
