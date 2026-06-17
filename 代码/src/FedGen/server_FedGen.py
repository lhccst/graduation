import copy
import torch
import numpy as np
from torch import nn

from src.FedGen.client_FedGen import ClientGen
from src.server_base import ServerBase
import torch.nn.functional as F

class ServerFedGen(ServerBase):
    def __init__(self, args):
        super().__init__(args)
        self.global_head = self.args.head.state_dict()
        self.generative_model = Generator(
                                args.noise_dim,
                                args.num_classes,
                                args.hidden_dim,
                                args.feature_dim,
                                self.device
                            ).to(self.device)
        self.generator_learning_rate= args.server['generator_learning_rate']
        self.initialize_clients(ClientGen)
        self.generator_optimizer = torch.optim.Adam(
            self.generative_model.parameters(),
            lr=self.generator_learning_rate
        )

        self.qualified_labels = []
        for client in self.clients:
            for yy in range(self.num_classes):
                self.qualified_labels.extend([yy for _ in range(int(client.sample_per_class[yy].item()))])
        self.reporter.track_upload(self.qualified_labels)
        for client in self.clients:
            client.qualified_labels = self.qualified_labels
            self.reporter.track_download(self.qualified_labels)
        self.server_epochs = args.server['server_epochs']
        self.batch_size = args.server['server_batch_size']
        self.loss = nn.CrossEntropyLoss()


    def train(self):
        acc_record = {}
        for i in range(self.global_rounds):
            self.join_clients = self.select_join_clients()
            print(f"============== global round: {i}, number of join clients: {len(self.join_clients)} ==============")

            for client in self.join_clients:
                client.train()

            # self.receive()
            self.receive_ids()
            self.train_generator()
            self.aggragate()
            self.dispatch()

            print("================== evaluate =================")
            acc = self.evaluate(i)
            acc_record[i] = acc
            print(f"================= global round: {i}, accuracy: {acc} =================")

        uploadCom,downloadCom,usedMemory =self.getReport()
        return acc_record,uploadCom,downloadCom,usedMemory

    def dispatch(self):
        """仅分发全局头部参数"""
        for client in self.join_clients:
            self.reporter.track_download(self.global_head)
            client.model.head.load_state_dict(self.global_head)
            client.generative_model=self.generative_model
            self.reporter.track_download(self.generative_model)


    # def receive(self):
    #     self.client_heads = []
    #     client_sample_sizes = []
    #     for client in self.join_clients:
    #         self.client_heads.append(client.model.head.state_dict())
    #         client_sample_sizes.append(client.sample_size)
    #         self.reporter.track_upload(client.model.head.state_dict())
    #     size_tensor = torch.tensor(client_sample_sizes, dtype=torch.float).to(self.device)
    #     self.reporter.track_upload(size_tensor)
    #     self.client_weight = size_tensor / torch.sum(size_tensor)

    def receive_ids(self):
        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        self.client_heads = []
        tot_samples = 0
        for client in self.join_clients:
            tot_samples += client.sample_size
            self.uploaded_ids.append(client.client_id)
            self.uploaded_weights.append(client.sample_size)
            self.uploaded_models.append(client.model.head)
            self.client_heads.append(client.model.head.state_dict())
            self.reporter.track_upload(client.model.head.state_dict())
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples


    def aggragate(self):
        uploaded_weights= torch.tensor(self.uploaded_weights).to(self.device)
        self.reporter.track_upload(uploaded_weights)
        for key in self.global_head.keys():
            param = [client_head[key] for client_head in self.client_heads]
            param = torch.stack(param, dim=0)
            shape = [-1] + [1] * (len(param.shape) - 1)
            self.global_head[key] = torch.sum(param * uploaded_weights.reshape(shape), dim=0)

    # def aggregate_parameters(self):
    #
    #     for param in head.parameters():
    #         param.data.zero_()
    #
    #     for w, cid in zip(self.uploaded_weights, self.uploaded_ids):
    #         client = self.clients[cid]
    #         client_model = load_item(client.role, 'model', client.save_folder_name).head
    #         for server_param, client_param in zip(head.parameters(), client_model.parameters()):
    #             server_param.data += client_param.data.clone() * w
    #
    #     save_item(head, self.role, 'head', self.save_folder_name)

    def train_generator(self):
        generative_optimizer = torch.optim.Adam(
            params=self.generative_model.parameters(),
            lr=self.generator_learning_rate, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=0, amsgrad=False)
        self.generative_model.train()

        for _ in range(self.server_epochs):
            labels = np.random.choice(self.qualified_labels, self.batch_size)
            labels = torch.LongTensor(labels).to(self.device)
            z = self.generative_model(labels)

            logits = 0
            for w, model in zip(self.uploaded_weights, self.uploaded_models):
                model.eval()
                logits += model(z) * w

            generative_optimizer.zero_grad()
            loss = self.loss(logits, labels)
            loss.backward()
            generative_optimizer.step()

# based on official code https://github.com/zhuangdizhu/FedGen/blob/main/FLAlgorithms/trainmodel/generator.py
class Generator(nn.Module):
    def __init__(self, noise_dim, num_classes, hidden_dim, feature_dim, device) -> None:
        super().__init__()

        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.device = device

        self.fc1 = nn.Sequential(
            nn.Linear(noise_dim + num_classes, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        self.fc = nn.Linear(hidden_dim, feature_dim)

    def forward(self, labels):
        batch_size = labels.shape[0]
        eps = torch.rand((batch_size, self.noise_dim), device=self.device) # sampling from Gaussian

        y_input = F.one_hot(labels, self.num_classes)
        z = torch.cat((eps, y_input), dim=1)

        z = self.fc1(z)
        z = self.fc(z)

        return z