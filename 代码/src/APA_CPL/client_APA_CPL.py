import copy
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import numpy as np

from src.APA_CPL.loss import GlobalConLoss, SupConLoss
from src.client_base import ClientBase


class ClientAPA_CPL(ClientBase):
    def __init__(self, args, client_id, trainset, testset, statistic):
        super().__init__(args, client_id, trainset, testset, statistic)

        self.model = copy.deepcopy(args.model)

        self.lamb_da = args.client["lambda"]
        self.global_loss = GlobalConLoss()  # contrastive loss
        self.con_loss = SupConLoss()  # contrastive loss

        self.local_protos = {}  # dict
        self.global_protos = None  # list

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)

    def aggregate(self, raw_feats):
        for label, feats in raw_feats.items():
            feats = F.normalize(torch.stack(feats, dim=0), dim=1)
            self.local_protos[label] = torch.mean(feats, dim=0)

    def collect_feats(self):
        train_loader = self.get_train_loader()
        self.model.eval()

        raw_feats = defaultdict(list)
        with torch.inference_mode():
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                feature = self.model.base(x)

                for label in torch.unique(y):
                    protos = feature[label == y].detach()
                    raw_feats[label.item()].extend(list(torch.unbind(protos, dim=0)))

        return raw_feats

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

                normalized_feature = F.normalize(feature, dim=1)
                loss += self.lamb_da * self.con_loss(normalized_feature, labels=y)

                if self.global_protos is not None:
                    loss += self.lamb_da * self.global_loss(normalized_feature,
                                                            labels=y,
                                                            global_protos=self.global_protos)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                local_iter.set_description(f"client {self.client_id} epoch: {epoch} loss: {loss.item():.4f}")

        raw_feats = self.collect_feats()
        self.aggregate(raw_feats)
