import torch

from src.FedPCL.client_FedPCL import ClientFedPCL
from src.server_base import ServerBase


class ServerFedPCL(ServerBase):
    def __init__(self, args):
        super().__init__(args)

        self.initialize_clients(ClientFedPCL)

        self.global_protos = [None] * self.num_classes
        self.client_protos_set = [None] * self.num_clients

    def dispatch(self):
        for client in self.join_clients:
            client.global_protos = self.global_protos
            client.client_protos_set = self.client_protos_set

    def receive_and_aggregate(self):
        """ 聚合 global_protos """
        for label in range(self.num_classes):
            proto_list = [client.local_protos[label] for client in self.join_clients
                          if label in client.local_protos]
            if len(proto_list) == 0:
                continue
            proto = torch.stack(proto_list, dim=0)
            client_sample = [client.statistic[str(label)] for client in self.join_clients
                             if label in client.local_protos]
            weight = torch.tensor(client_sample).to(self.device)
            weight = weight / torch.sum(weight)
            proto = torch.sum(proto * weight.reshape(-1, 1), dim=0)
            self.global_protos[label] = proto

        """ 获得每个用户的 proto_set，若自身的不含某 label 的样本，则使用全局 proto 代替 """
        for client in self.join_clients:
            self.client_protos_set[client.client_id] = [
                client.local_protos[label] if label in client.local_protos
                else self.global_protos[label]
                for label in range(self.num_classes)
            ]
        for i in range(len(self.client_protos_set)):
            if self.client_protos_set[i] is None:
                self.client_protos_set[i] = self.global_protos

    def train(self):
        acc_record = {}
        for i in range(self.global_rounds):
            self.join_clients = self.select_join_clients()
            print(f"============== global round: {i}, number of join clients: {len(self.join_clients)} ==============")

            for client in self.join_clients:
                client.train()

            self.receive_and_aggregate()
            self.dispatch()

            print("================= evaluate =================")
            acc = self.evaluate()
            acc_record[i] = acc
            print(f"================= global round: {i}, accuracy: {acc} =================")

        return acc_record
