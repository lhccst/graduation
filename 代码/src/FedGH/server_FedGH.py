import copy

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.FedGH.client_FedGH import ClientFedGH
from src.server_base import ServerBase
from mem_utils import MemReporter

class ServerFedGH(ServerBase):
    def __init__(self, args):
        super().__init__(args)

        self.initialize_clients(ClientFedGH)

        self.protos = []
        self.global_head = copy.deepcopy(self.args.head).to(self.device)

        self.server_loss = nn.CrossEntropyLoss()
        self.server_batch_size = args.server["batch_size"]
        self.server_learning_rate = args.server["learning_rate"]
        self.server_optimizer = torch.optim.SGD(self.global_head.parameters(),
                                                lr=self.server_learning_rate,
                                                momentum=0.9)

    def dispatch(self):
        for client in self.join_clients:
            client.model.head.load_state_dict(self.global_head.state_dict())
            self.reporter.track_download(self.global_head.state_dict())

    def receive(self):
        self.protos = []
        for client in self.join_clients:
            for label, proto in client.local_protos.items():
                self.protos.append((proto, label))
                self.reporter.track_upload(label)
                self.reporter.track_upload(proto)

    def train_global_head(self):
        proto_loader = DataLoader(self.protos, self.server_batch_size, shuffle=True, drop_last=False)

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

            self.receive()
            self.train_global_head()
            self.dispatch()

            print("================= evaluate =================")
            acc = self.evaluate(i)
            acc_record[i] = acc
            print(f"================= global round: {i}, accuracy: {acc} =================")

        uploadCom,downloadCom,usedMemory =self.getReport()
        return acc_record,uploadCom,downloadCom,usedMemory
