import copy
import random
import time

import torch
from src.FedMRL.client_FedMRL import ClientFedMRL
from src.server_base import ServerBase
from models.models import BaseHeadSplit

class ServerFedMRL(ServerBase):
    def __init__(self, args):
        super().__init__(args)

        self.initialize_clients(ClientFedMRL)
        self.server_global_model = BaseHeadSplit(args, 0, args.sub_feature_dim).to(args.device)


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
            acc = self.evaluate(i)
            acc_record[i] = acc
            print(f"================= global round: {i}, accuracy: {acc} =================")

        uploadCom,downloadCom,usedMemory =self.getReport()
        return acc_record,uploadCom,downloadCom,usedMemory


    def receive_and_aggregated(self):
        client_global_models = []
        client_sample_sizes = []
        for client in self.join_clients:
            client_global_models.append(client.global_model.state_dict())
            client_sample_sizes.append(client.sample_size)
            self.reporter.track_upload(client.global_model.state_dict())
        size_tensor = torch.tensor(client_sample_sizes, dtype=torch.float).to(self.device)
        self.reporter.track_upload(size_tensor)
        client_weight = size_tensor / torch.sum(size_tensor)

        self.server_global_model_dict = self.server_global_model.state_dict()
        for key in self.server_global_model_dict.keys():
            param = [client_global_model[key] for client_global_model in client_global_models]
            param = torch.stack(param, dim=0)
            shape = [-1] + [1] * (len(param.shape) - 1)
            self.server_global_model_dict[key] = torch.sum(param * client_weight.reshape(shape), dim=0)

    def dispatch(self):
        """仅分发全局头部参数"""
        for client in self.join_clients:
            self.reporter.track_download(self.server_global_model_dict)
            client.global_model.load_state_dict(self.server_global_model_dict)

