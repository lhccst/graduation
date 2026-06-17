import copy

import torch

from src.FedAPAnoSplit.client_FedAPAnoSplit import ClientFedAPAnoSplit
from src.server_base import ServerBase
from mem_utils import MemReporter

class ServerFedAPAnoSplit(ServerBase):
    def __init__(self, args):
        super().__init__(args)

        self.initialize_clients(ClientFedAPAnoSplit)

        initial_param = self.args.model.state_dict()
        self.client_models = [initial_param for _ in range(self.num_clients)]

        self.client_weights = torch.zeros(size=(self.num_clients, self.num_clients),
                                          requires_grad=True,
                                          device=self.device)

        self.server_learning_rate = args.server["learning_rate"]
        self.optimizer = torch.optim.SGD([self.client_weights], self.server_learning_rate, momentum=0.9)

    def aggregate(self, client_id):
        weight = self.client_weights[client_id]
        with torch.no_grad():
            weight[client_id] = 0.5
            weight.data.copy_(weight / torch.sum(weight))

        aggregated = {}
        for key in self.args.model.state_dict().keys():
            param = [client_model[key] for client_model in self.client_models]
            param = torch.stack(param, dim=0)
            shape = [-1] + [1] * (len(param.shape) - 1)
            aggregated[key] = torch.sum(param * weight.reshape(shape), dim=0)
        return aggregated

    def receive(self, client):
        self.reporter.track_upload(client.model.state_dict())
        return copy.deepcopy(client.model.state_dict())

    def dispatch(self, client, aggregated):
        self.reporter.track_download(aggregated)
        client.model.load_state_dict(aggregated)

    def update_delta_theta(self, initial_state, final_state):
        return {
            layer: initial_state[layer] - final_state[layer] for layer in final_state.keys()
        }

    def train(self):
        self.reporter = MemReporter()
        acc_record = {}
        for i in range(self.global_rounds):
            self.join_clients = self.select_join_clients()
            print(f"============== global round: {i}, number of join clients: {len(self.join_clients)} ==============")

            for client in self.join_clients:
                aggregated = self.aggregate(client.client_id)
                self.dispatch(client, aggregated)
                client.train()
                updated = self.receive(client)
                self.client_models[client.client_id] = updated
                delta_theta = self.update_delta_theta(aggregated, updated)

                grad = torch.autograd.grad(
                    outputs=aggregated.values(),
                    inputs=self.client_weights,
                    grad_outputs=delta_theta.values(),
                )

                self.optimizer.zero_grad()
                self.client_weights.grad = grad[0]
                self.optimizer.step()

                with torch.no_grad():
                    self.client_weights.clamp_(0, 1)

            print("================= evaluate =================")
            acc = self.evaluate()
            acc_record[i] = acc
            print(f"================= global round: {i}, accuracy: {acc} =================")

        uploadCom,downloadCom=self.reporter.print_communication_stats()
        return acc_record,uploadCom,downloadCom
