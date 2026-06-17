import copy

import seaborn
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from src.FedPAA_GH.client_FedPAA_GH import ClientFedPAA_GH
from src.server_base import ServerBase


class ServerFedPAA_GH(ServerBase):
    def __init__(self, args):
        super().__init__(args)

        self.initialize_clients(ClientFedPAA_GH)

        self.client_embedding = [torch.zeros(self.args.feature_dim)] * self.num_clients  # list
        self.client_protos = []  # list
        self.client_bases = [self.args.model.base.state_dict()] * self.num_clients  # list
        self.aggregateds = [None] * self.num_clients  # list

        self.global_head = copy.deepcopy(args.model.head)

        self.server_loss = nn.CrossEntropyLoss()
        self.server_batch_size = args.server["batch_size"]
        self.server_learning_rate = args.server["learning_rate"]
        self.server_optimizer = torch.optim.SGD(self.global_head.parameters(),
                                                lr=self.server_learning_rate,
                                                momentum=0.9)

    def compute_embedding(self, num_samples, protos):
        weight = torch.tensor(num_samples)
        weight = weight / torch.sum(weight)

        protos = torch.stack(protos, dim=0)
        embedding = torch.sum(protos * weight.reshape(-1, 1), dim=0)
        return embedding

    def dispatch(self):
        for client in self.join_clients:
            client.model.base.load_state_dict(self.aggregateds[client.client_id])
            client.model.head.load_state_dict(self.global_head.state_dict())

    def receive_and_aggregated(self):
        for client in self.join_clients:
            num_samples = []
            protos = []
            for label, proto in client.local_protos.items():
                num_samples.append(client.statistic[str(label)])
                protos.append(proto)
                self.client_protos.append((proto, label))

            self.client_embedding[client.client_id] = self.compute_embedding(num_samples, protos)
            self.client_bases[client.client_id] = client.model.base.state_dict()

        client_embedding = F.normalize(torch.stack(self.client_embedding, dim=0), dim=1)
        similarity = torch.matmul(client_embedding, client_embedding.T)  # shape of (num_clients, num_clients)
        similarity = torch.where(similarity > 0.8, similarity, 1e-5)  # between 0 and 1

        plt.figure()
        seaborn.heatmap(similarity, annot=True, square=True)
        plt.show()

        similarity /= torch.sum(similarity, dim=0)
        for client in self.join_clients:
            client_similarity = similarity[client.client_id]

            aggregated = {}
            for key in self.args.model.base.state_dict().keys():
                param = [client_base[key] for client_base in self.client_bases]
                param = torch.stack(param, dim=0)
                shape = [-1] + [1] * (len(param.shape) - 1)
                aggregated[key] = torch.sum(param * client_similarity.reshape(shape), dim=0)
            self.aggregateds[client.client_id] = aggregated

    def train_global_head(self):
        proto_loader = DataLoader(self.client_protos, self.server_batch_size, shuffle=True, drop_last=False)

        for epoch in range(self.args.client["local_epochs"]):
            print(f"Training global head in Server...  epoch: {epoch} / {self.args.client['local_epochs']}")
            for proto, y in proto_loader:
                y = y.to(self.device)
                output = self.global_head(proto)
                loss = self.server_loss(output, y)

                self.server_optimizer.zero_grad()
                loss.backward()
                self.server_optimizer.step()

    def train(self):
        acc_record = {}
        for i in range(self.global_rounds):
            self.join_clients = self.select_join_clients()
            print(f"============== global round: {i}, number of join clients: {len(self.join_clients)} ==============")

            for client in self.join_clients:
                client.train()

            self.receive_and_aggregated()
            self.train_global_head()
            self.dispatch()

            print("================= evaluate =================")
            acc = self.evaluate()
            acc_record[i] = acc
            print(f"================= global round: {i}, accuracy: {acc} =================")

        return acc_record
