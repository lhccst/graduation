
import copy

import torch
import numpy as np
from collections import defaultdict
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from models.FedAvgCNN import FedAvgCNN
from models.models import BaseHeadSplit
from src.server_base import ServerBase
from mem_utils import MemReporter
from src.FedRAL.client_FedRAL import ClientFedRAL, ChangeModel, OFT


class ServerFedRAL(ServerBase):
    def __init__(self, args):
        super().__init__(args)
        # self.shared_params = copy.deepcopy(args.head.state_dict())
        # self.aggregatedHead =  copy.deepcopy(args.head.state_dict())
        self.initialize_clients(ClientFedRAL)
        # self.model = ChangeModel(self.model)
        oft = OFT()
        self.aggregated = copy.deepcopy(oft.state_dict())

    def dispatch(self):
        for client in self.join_clients:
            self.reporter.track_download(self.aggregated)
            client.net.oft.load_state_dict(self.aggregated)

    def receive_and_aggregated(self):
        client_models = []
        client_sample_sizes = []
        for client in self.join_clients:
            client_models.append(client.Local_oft)
            client_sample_sizes.append(client.sample_size)
            self.reporter.track_upload(client.Local_oft)
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
