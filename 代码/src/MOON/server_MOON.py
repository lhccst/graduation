import torch

from src.MOON.client_MOON import ClientMOON
from src.server_base import ServerBase


class ServerMOON(ServerBase):
    def __init__(self, args):
        super().__init__(args)

        self.aggregated = self.args.model.base.state_dict()

        self.initialize_clients(ClientMOON)

    def dispatch(self):
        for client in self.join_clients:
            client.previous_model.load_state_dict(client.model.base.state_dict())
            client.model.base.load_state_dict(self.aggregated)
            client.server_model.load_state_dict(self.aggregated)

    def receive_and_aggregated(self):
        client_models = []
        client_sample_sizes = []
        for client in self.join_clients:
            client_models.append(client.model.base.state_dict())
            client_sample_sizes.append(client.sample_size)
        size_tensor = torch.tensor(client_sample_sizes, dtype=torch.float).to(self.device)
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
            print(f"============== global round: {i}, number of join clients: {len(self.join_clients)} ==============")

            for client in self.join_clients:
                client.train()

            self.receive_and_aggregated()
            self.dispatch()

            print("================== evaluate =================")
            acc = self.evaluate()
            acc_record[i] = acc
            print(f"================= global round: {i}, accuracy: {acc} =================")

        return acc_record
