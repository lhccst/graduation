from collections import OrderedDict
import torch

from src.FedPHP.client_FedPHP import ClientFedPHP
from src.server_base import ServerBase

class ServerFedPHP(ServerBase):
    def __init__(self, args):
        super().__init__(args)

        self.aggregated = self.args.model.base.state_dict()
        self.client_weights = None

        self.initialize_clients(ClientFedPHP)

    def dispatch(self):
        for client in self.join_clients:
            client.server_model = self.aggregated
            self.reporter.track_download(self.aggregated)
            client.update_parameter(self.lowest_join_rate, self.global_rounds)

    def receive_and_aggregated(self):
        client_models = [client.model.base.state_dict() for client in self.join_clients]
        for client in self.join_clients:
            # 获取模型的 state_dict，然后调用 track_upload 进行处理
            self.reporter.track_upload(client.model.base.state_dict())

        client_size = torch.tensor(
            [client.sample_size for client in self.join_clients], dtype=torch.float
        ).to(self.device)
        client_weight = client_size / torch.sum(client_size)

        for key in self.aggregated.keys():
            param = [client_model[key] for client_model in client_models]
            param = torch.stack(param, dim=0)
            shape = [-1] + [1] * (len(param.shape) - 1)
            self.aggregated[key] = torch.sum(param * client_weight.reshape(shape), dim=0)

    def train(self):
        acc_record = {}
        for i in range(self.global_rounds):
            print(f"=========== global round: {i} ===========")

            self.join_clients = self.select_join_clients()

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

