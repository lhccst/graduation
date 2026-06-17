import torch

from src.FedProto.client_FedProto import ClientFedProto
from src.server_base import ServerBase
from mem_utils import MemReporter
import math

class ServerFedProto(ServerBase):
    def __init__(self, args):
        super().__init__(args)

        self.initialize_clients(ClientFedProto)

        self.global_protos = [None] * self.num_classes


    def dispatch(self):
        if None not in self.global_protos:
            for client in self.join_clients:
                self.reporter.track_download(self.global_protos)
                client.global_protos = self.global_protos

    def receive_and_aggregate(self):
        for label in range(self.num_classes):
            proto_list = [client.local_protos[label] for client in self.join_clients
                          if label in client.local_protos]
            if len(proto_list) == 0:
                continue
            proto = torch.stack(proto_list, dim=0)
            self.reporter.track_upload(proto)
            client_sample = [client.statistic[str(label)] for client in self.join_clients
                             if label in client.local_protos]
            weight = torch.tensor(client_sample).to(self.device)
            self.reporter.track_upload(weight)
            weight = weight / torch.sum(weight)
            proto = torch.sum(proto * weight.reshape(-1, 1), dim=0)
            self.global_protos[label] = proto


    def train(self):
        acc_record = {}
        for i in range(self.global_rounds):
            self.args.current_round = i
            self.join_clients = self.select_join_clients()
            print(f"============== global round: {i}, number of join clients: {len(self.join_clients)} ==============")

            for client in self.join_clients:
                client.train()

            self.receive_and_aggregate()
            self.dispatch()

            print("================= evaluate =================")
            acc = self.evaluate(i)
            acc_record[i] = acc
            print(f"================= global round: {i}, accuracy: {acc} =================")

        uploadCom,downloadCom,usedMemory =self.getReport()
        return acc_record,uploadCom,downloadCom,usedMemory

