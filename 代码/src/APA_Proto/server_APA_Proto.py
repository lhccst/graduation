import copy

import torch

from src.APA_CPL.client_APA_CCPL import ClientAPA_CCPL
from src.server_base import ServerBase


class ServerAPA_Proto(ServerBase):
    def __init__(self, args):
        super().__init__(args)

        self.initialize_clients(ClientAPA_CCPL)

        """ FedAPA """
        initial_param = self.args.model.base.state_dict()
        self.client_models = [initial_param for _ in range(self.num_clients)]

        self.client_weights = torch.zeros(size=(self.num_clients, self.num_clients),
                                          requires_grad=True,
                                          device=self.device)

        self.server_learning_rate = args.server["learning_rate"]
        self.optimizer = torch.optim.SGD([self.client_weights], self.server_learning_rate, momentum=0.9)

        """ FedProto """
        self.global_protos = [None] * self.num_classes

    def aggregate_parameters(self, client_id):
        weight = self.client_weights[client_id]
        with torch.no_grad():
            weight[client_id] = 0.5
            weight.data.copy_(weight / torch.sum(weight))

        aggregated = {}
        for key in self.args.model.base.state_dict().keys():
            param = [client_model[key] for client_model in self.client_models]
            param = torch.stack(param, dim=0)
            shape = [-1] + [1] * (len(param.shape) - 1)
            aggregated[key] = torch.sum(param * weight.reshape(shape), dim=0)

        return aggregated

    def receive_and_aggregate_protos(self):
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

    def receive_parameters(self, client):
        return copy.deepcopy(client.model.base.state_dict())

    def dispatch(self, client, aggregated, global_protos):
        client.model.base.load_state_dict(aggregated)
        if None not in self.global_protos:
            client.global_protos = global_protos

    def update_delta_theta(self, initial_state, final_state):
        return {
            layer: initial_state[layer] - final_state[layer] for layer in final_state.keys()
        }

    def train(self):
        acc_record = {}
        for i in range(self.global_rounds):
            self.join_clients = self.select_join_clients()
            print(f"============== global round: {i}, number of join clients: {len(self.join_clients)} ==============")

            for client in self.join_clients:
                aggregated = self.aggregate_parameters(client.client_id)
                self.dispatch(client, aggregated, self.global_protos)
                client.train()
                updated = self.receive_parameters(client)
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

            self.receive_and_aggregate_protos()

            print("================= evaluate =================")
            acc = self.evaluate()
            acc_record[i] = acc
            print(f"================= global round: {i}, accuracy: {acc} =================")
        return acc_record
