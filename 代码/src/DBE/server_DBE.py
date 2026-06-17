import torch

from src.DBE.client_DBE import ClientDBE
from src.server_base import ServerBase
from mem_utils import MemReporter

class ServerDBE(ServerBase):
    def __init__(self, args):
        super().__init__(args)

        self.global_means = None
        self.aggregated = self.args.model.base.state_dict()

        self.initialize_clients(ClientDBE)
        self.pre_train()

    def pre_train(self):
        for client in self.clients:
            client.train()

        client_size = torch.tensor(
            [client.sample_size for client in self.clients], dtype=torch.float
        ).to(self.device)
        client_weight = client_size / torch.sum(client_size)

        client_local_means = torch.stack([client.local_mean.detach() for client in self.clients])
        self.global_means = torch.sum(client_local_means * client_weight.reshape(-1, 1), dim=0)

        for client in self.clients:
            client.global_mean = self.global_means

    def dispatch(self):
        for client in self.join_clients:
            client.model.base.load_state_dict(self.aggregated)
            self.reporter.track_download(self.aggregated)

    def receive_and_aggregated(self):
        client_models = []
        client_sample_sizes = []
        for client in self.join_clients:
            client_models.append(client.model.base.state_dict())
            self.reporter.track_upload(client.model.base.state_dict())
            client_sample_sizes.append(client.sample_size)
        size_tensor = torch.tensor(client_sample_sizes, dtype=torch.float).to(self.device)
        self.reporter.track_upload(size_tensor)
        client_weight = size_tensor / torch.sum(size_tensor)

        for key in self.aggregated.keys():
            param = [client_model[key] for client_model in client_models]
            param = torch.stack(param, dim=0)
            shape = [-1] + [1] * (len(param.shape) - 1)
            self.aggregated[key] = torch.sum(param * client_weight.reshape(shape), dim=0)

    def train(self):
        acc_record = {}
        for i in range(self.global_rounds):
            self.join_clients = self.select_join_clients()
            print(f"========== global round: {i}, number of join clients: {len(self.join_clients)} ==========")

            for client in self.join_clients:
                client.train()

            self.receive_and_aggregated()
            self.dispatch()

            print("================= evaluate =================")
            acc = self.evaluate()
            acc_record[i] = acc
            print(f"================= global round: {i}, accuracy: {acc} =================")

        uploadCom,downloadCom,usedMemory =self.getReport()
        return acc_record,uploadCom,downloadCom,usedMemory
