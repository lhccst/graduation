import copy

import numpy as np
import torch
from sklearn.cluster import KMeans

from src.APA_CCPL.client_APA_CCPL import ClientAPA_CCPL
from src.server_base import ServerBase


class ServerAPA_CCPL(ServerBase):
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
        self.global_protos = None  # list
        self.received_protos = None  # list
        self.sample_weights = None  # list

    def receive_protos(self):
        self.received_protos = []
        sample_size = []
        for client in self.join_clients:
            for key in client.local_protos.keys():
                self.received_protos.append(client.local_protos[key])
                sample_size.append(client.statistic[str(key)])

        self.sample_weights = np.array(sample_size) / np.sum(sample_size)
        self.sample_weights *= len(self.received_protos)

    def aggregate_protos(self):
        proto_tensor = torch.stack(self.received_protos).cpu()
        proto_np_array = proto_tensor.numpy()

        if self.args.server["clustering"] == "kmeans":
            clustering = KMeans(n_clusters=self.num_classes, algorithm="elkan", n_init="auto")
            clustering.fit(proto_np_array, sample_weight=self.sample_weights)

            centroids = clustering.cluster_centers_  # 获取聚类中心

        else:
            raise NotImplementedError

        return [torch.from_numpy(centroid).to(self.device) for centroid in centroids]

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

    def receive_parameters(self, client):
        return copy.deepcopy(client.model.base.state_dict())

    def dispatch(self, client, aggregated, global_protos):
        client.model.base.load_state_dict(aggregated)
        if self.global_protos is not None:
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

            self.receive_protos()
            self.global_protos = self.aggregate_protos()

            print("================= evaluate =================")
            acc = self.evaluate()
            acc_record[i] = acc
            print(f"================= global round: {i}, accuracy: {acc} =================")
        return acc_record
