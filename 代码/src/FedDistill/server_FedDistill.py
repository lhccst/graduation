import torch

from src.FedDistill.client_FedDistill import ClientFedDistill
from src.server_base import ServerBase
from mem_utils import MemReporter

class ServerFedDistill(ServerBase):
    def __init__(self, args):
        super().__init__(args)

        self.initialize_clients(ClientFedDistill)

        self.global_logits = [None] * self.num_classes

    def dispatch(self):
        if None not in self.global_logits:
            for client in self.join_clients:
                self.reporter.track_download(self.global_logits)
                client.global_logits = self.global_logits

    def receive_and_aggregate(self):
        for label in range(self.num_classes):
            logits_list = [client.local_logits[label] for client in self.join_clients
                           if label in client.local_logits]
            if len(logits_list) == 0:
                continue
            logits = torch.stack(logits_list, dim=0)
            self.reporter.track_upload(logits)
            client_sample = [client.statistic[str(label)] for client in self.join_clients
                             if label in client.local_logits]
            weight = torch.tensor(client_sample).to(self.device)
            weight = weight / torch.sum(weight)
            logit = torch.sum(logits * weight.reshape(-1, 1), dim=0)
            self.global_logits[label] = logit

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
            acc = self.evaluate(i)
            acc_record[i] = acc
            print(f"================= global round: {i}, accuracy: {acc} =================")

        uploadCom,downloadCom,usedMemory =self.getReport()
        return acc_record,uploadCom,downloadCom,usedMemory
