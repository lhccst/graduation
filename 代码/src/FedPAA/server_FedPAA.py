import seaborn
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from src.FedPAA.client_FedPAA import ClientFedPAA
from src.server_base import ServerBase


class ServerFedPAA(ServerBase):
    def __init__(self, args):
        super().__init__(args)

        self.client_embedding = [torch.zeros(self.args.feature_dim)] * self.num_clients  # list
        self.client_models = [self.args.model.head.state_dict()] * self.num_clients  # list
        self.aggregateds = [None] * self.num_clients  # list

        self.initialize_clients(ClientFedPAA)

    def dispatch(self):
        for client in self.join_clients:
            client.model.head.load_state_dict(self.aggregateds[client.client_id])

    def receive_and_aggregated(self):
        for client in self.join_clients:
            self.client_embedding[client.client_id] = client.embedding
            self.client_models[client.client_id] = client.model.head.state_dict()

        client_embedding = F.normalize(torch.stack(self.client_embedding, dim=0), dim=1)

        similarity = torch.matmul(client_embedding, client_embedding.T)  # shape of (num_clients, num_clients)
        similarity = torch.where(similarity > 0.8, similarity, 1e-5)  # between 0 and 1
        similarity /= torch.sum(similarity, dim=0)

        plt.figure()
        seaborn.heatmap(similarity, annot=True, square=True)
        plt.show()

        for client in self.join_clients:
            client_similarity = similarity[client.client_id]

            aggregated = {}
            for key in self.args.model.head.state_dict().keys():
                param = [client_model[key] for client_model in self.client_models]
                param = torch.stack(param, dim=0)
                shape = [-1] + [1] * (len(param.shape) - 1)
                aggregated[key] = torch.sum(param * client_similarity.reshape(shape), dim=0)
            self.aggregateds[client.client_id] = aggregated

    def train(self):
        acc_record = {}
        for i in range(self.global_rounds):
            self.join_clients = self.select_join_clients()
            print(f"============== global round: {i}, number of join clients: {len(self.join_clients)} ==============")

            for client in self.join_clients:
                client.train()

            self.receive_and_aggregated()
            self.dispatch()

            print("================= evaluate =================")
            acc = self.evaluate()
            acc_record[i] = acc
            print(f"================= global round: {i}, accuracy: {acc} =================")

        return acc_record
