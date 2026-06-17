import random
import time
from src.FedREP.client_FedREP import ClientFedREP
from src.server_base import ServerBase
from mem_utils import MemReporter
import torch

class ServerFedREP(ServerBase):
    def __init__(self, args):
        super().__init__(args)
        self.initialize_clients(ClientFedREP)
        self.aggregated = self.args.model.base.state_dict()
    # 还缺什么？

    def train(self):
        self.reporter = MemReporter()
        acc_record = {}
        for i in range(self.global_rounds):
            self.join_clients = self.select_join_clients()
            print(f"============== global round: {i}, number of join clients: {len(self.join_clients)} ==============")

            for client in self.join_clients:
                client.train()

            self.receive_and_aggregated()
            self.dispatch()

            print("================== evaluate =================")
            acc = self.evaluate()
            acc_record[i] = acc
            print(f"================= global round: {i}, accuracy: {acc} =================")

        uploadCom,downloadCom=self.reporter.print_communication_stats()
        return acc_record,uploadCom,downloadCom

    def receive_and_aggregated(self):
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in self.join_clients:
            tot_samples += client.sample_size
            self.uploaded_weights.append(client.sample_size)
            self.uploaded_models.append(client.model.base.state_dict())
            self.reporter.track_upload(client.model.base.state_dict())
        self.reporter.track_upload(self.uploaded_weights)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

        self.uploaded_weights = torch.tensor(self.uploaded_weights, dtype=torch.float).to(self.device)

        for key in self.aggregated.keys():
            param = [client_model[key] for client_model in self.uploaded_models]
            param = torch.stack(param, dim=0)
            shape = [-1] + [1] * (len(param.shape) - 1)
            self.aggregated[key] = torch.sum(param * self.uploaded_weights.reshape(shape), dim=0)


    def dispatch(self):
        for client in self.join_clients:
            self.reporter.track_download(self.aggregated)
            client.model.base.load_state_dict(self.aggregated)