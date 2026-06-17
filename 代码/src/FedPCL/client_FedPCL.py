import copy
from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.FedPCL.loss import GlobalConLoss
from src.client_base import ClientBase


class ClientFedPCL(ClientBase):
    def __init__(self, args, client_id, trainset, testset, statistic):
        super().__init__(args, client_id, trainset, testset, statistic)

        self.model = copy.deepcopy(args.model.base)  # 只有特征提取器部分，没有分类器

        self.local_protos = {}
        self.global_protos = None  # list
        self.client_protos_set = None  # list

        self.con_loss = GlobalConLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)

    def collect_feats(self):
        train_loader = self.get_train_loader()
        self.model.eval()

        raw_feats = defaultdict(list)
        with torch.inference_mode():
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                feature = self.model(x)

                for label in torch.unique(y):
                    protos = feature[label == y].detach()
                    raw_feats[label.item()].extend(list(torch.unbind(protos, dim=0)))

        return raw_feats

    def aggregate(self, raw_feats):
        for label, feats in raw_feats.items():
            feats = F.normalize(torch.stack(feats, dim=0))
            self.local_protos[label] = torch.mean(feats, dim=0)

    def train(self):
        train_loader = self.get_train_loader()
        self.model.train()

        if self.global_protos is not None:
            for epoch in range(self.local_epochs):
                local_iter = tqdm(train_loader, desc=f"client {self.client_id} epoch: {epoch}")
                for x, y in local_iter:
                    x, y = x.to(self.device), y.to(self.device)

                    feature = self.model(x)
                    feature = F.normalize(feature, dim=1)

                    global_proto_loss = self.con_loss(feature, labels=y, global_protos=self.global_protos)

                    client_proto_loss = 0
                    for client_protos in self.client_protos_set:
                        client_proto_loss += self.con_loss(feature, labels=y, global_protos=client_protos)
                    client_proto_loss /= len(self.client_protos_set)

                    loss = global_proto_loss + client_proto_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    local_iter.set_description(f"client {self.client_id} epoch: {epoch} loss: {loss.item():.4f}")
        else:
            print(f"client {self.client_id} is collecting Protos...")

        raw_feats = self.collect_feats()
        self.aggregate(raw_feats)

    def evaluate_test(self):
        """ 因为模型没有分类器，因此性能评测函数需要 Overwrite """
        if self.global_protos is None:
            return 0, 0

        global_protos = torch.stack(self.global_protos, dim=0)
        test_loader = self.get_test_loader()

        self.model.eval()
        total_correct = 0
        total_samples = 0
        with torch.inference_mode():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                feature = self.model(x)
                feature = F.normalize(feature, dim=1)

                output = torch.matmul(feature, global_protos.T)
                total_correct += torch.sum(torch.argmax(output, dim=1) == y).item()
                total_samples += len(y)
        return total_correct, total_samples
