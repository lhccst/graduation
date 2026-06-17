import copy

import torch
from torch import nn

from src.pFedHN.client_pFedHN import ClientFedHN
from src.server_base import ServerBase
from mem_utils import MemReporter

class ServerFedHN(ServerBase):
    def __init__(self, args):
        super().__init__(args)

        self.hyper_network = copy.deepcopy(args.server_model)

        self.initialize_clients(ClientFedHN)

        self.server_learning_rate = args.server["learning_rate"]
        self.optimizer = torch.optim.SGD(self.hyper_network.parameters(), lr=self.server_learning_rate, momentum=0.9)

    def dispatch(self, client, weight):
        client.model.base.load_state_dict(weight)
        self.reporter.track_download(weight)

    def receive(self, client):
        self.reporter.track_upload(client.model.base.state_dict())
        return client.model.base.state_dict()

    def update_delta_theta(self, initial_state, final_state):
        return {
            layer: initial_state[layer] - final_state[layer] for layer in final_state.keys()
        }

    def train(self):
        acc_record = {}
        for i in range(self.global_rounds):
            self.join_clients = self.select_join_clients()
            print(f"============== global round: {i}, number of join clients: {len(self.join_clients)} ==============")

            self.hyper_network.train()
            for client in self.join_clients:
                weight = self.hyper_network(client.client_id)
                self.dispatch(client, weight)
                client.train()
                updated_weight = self.receive(client)
                delta_theta = self.update_delta_theta(weight, updated_weight)

                grads = torch.autograd.grad(
                    outputs=weight.values(),
                    inputs=self.hyper_network.parameters(),
                    grad_outputs=delta_theta.values(),
                    create_graph=True
                )

                self.optimizer.zero_grad()
                # for param, grad in zip(self.hyper_network.parameters(), grads):
                #     param.grad = grad
                with torch.no_grad():  # 确保在 no_grad 模式下进行就地修改
                    for param, grad in zip(self.hyper_network.parameters(), grads):
                        if param.grad is None:
                            param.grad = torch.zeros_like(param)
                        param.grad.copy_(grad)  # 使用 copy_() 进行就地修改

                self.optimizer.step()

            print("================= evaluate =================")
            acc = self.evaluate()
            acc_record[i] = acc
            print(f"================= global round: {i}, accuracy: {acc} =================")

        uploadCom,downloadCom,usedMemory =self.getReport()
        return acc_record,uploadCom,downloadCom,usedMemory


class HyperNetwork(nn.Module):
    def __init__(self,
                 num_clients,
                 embedding_dim,
                 num_hidden,
                 hidden_dim,
                 num_kernels,
                 in_channels,
                 dim,
                 device):
        super().__init__()

        self.device = device

        self.num_kernels = num_kernels
        self.in_channels = in_channels
        self.dim = dim

        self.embeddings = nn.Embedding(num_clients, embedding_dim)

        mlp = [nn.Linear(embedding_dim, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(num_hidden):
            mlp.append(nn.Linear(hidden_dim, hidden_dim))
            mlp.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*mlp)

        self.conv1_weight = nn.Linear(hidden_dim, self.num_kernels * self.in_channels * 5 * 5)
        self.conv1_bias = nn.Linear(hidden_dim, self.num_kernels)
        self.conv2_weight = nn.Linear(hidden_dim, 16 * self.num_kernels * 5 * 5)
        self.conv2_bias = nn.Linear(hidden_dim, 16)
        self.fc1_weight = nn.Linear(hidden_dim, 120 * dim)
        self.fc1_bias = nn.Linear(hidden_dim, 120)
        self.fc2_weight = nn.Linear(hidden_dim, 84 * 120)
        self.fc2_bias = nn.Linear(hidden_dim, 84)

    def forward(self, client_id):
        embed = self.embeddings(torch.tensor(client_id).to(self.device))
        feature = self.mlp(embed)

        weight = {
            "conv1.0.weight": self.conv1_weight(feature).reshape(self.num_kernels, self.in_channels, 5, 5),
            "conv1.0.bias": self.conv1_bias(feature).reshape(-1),
            "conv2.0.weight": self.conv2_weight(feature).reshape(16, self.num_kernels, 5, 5),
            "conv2.0.bias": self.conv2_bias(feature).reshape(-1),
            "fc1.0.weight": self.fc1_weight(feature).reshape(120, self.dim),
            "fc1.0.bias": self.fc1_bias(feature).reshape(-1),
            "fc2.0.weight": self.fc2_weight(feature).reshape(84, 120),
            "fc2.0.bias": self.fc2_bias(feature).reshape(-1),
        }
        return weight


def generate_server_model(args, dim):
    return HyperNetwork(num_clients=args.client["num_clients"],
                        embedding_dim=int(args.client["num_clients"] / 4) + 1,
                        num_hidden=3,
                        hidden_dim=100,
                        num_kernels=6,
                        in_channels=1 if args.dataset in ["mnist", "fmnist"] else 3,
                        dim=dim,
                        device=args.device)
