import torch
from collections import defaultdict
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

from src.FedTGP.client_FedTGP import ClientFedTGP
from src.server_base import ServerBase
from src.FedTGP.Trainable_prototypes import Trainable_prototypes
from mem_utils import MemReporter

class ServerFedTGP(ServerBase):
    def __init__(self, args):
        super().__init__(args)

        self.margin_threthold = args.server["margin_threthold"]
        self.server_learning_rate = args.server["server_learning_rate"]
        self.server_epochs = args.server["server_epochs"]
        self.batch_size = args.server["batch_size"]
        self.feature_dim = args.feature_dim
        self.server_hidden_dim = self.feature_dim
        self.initialize_clients(ClientFedTGP)

        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        self.min_gap = None
        self.max_gap = None
        self.global_protos = [None] * self.num_classes
        self.avg_protos = defaultdict(list)
        self.PROTO =  Trainable_prototypes(     # 服务端处训练的原型
                self.num_classes,
                self.server_hidden_dim,
                self.feature_dim,
                self.device
            ).to(self.device)
        self.CEloss = nn.CrossEntropyLoss()



    def dispatch(self):
        if None not in self.global_protos:
            for client in self.join_clients:
                client.global_protos = self.global_protos
                self.reporter.track_download(self.global_protos)

    def receive_and_aggregate(self):
        self.uploaded_protos = []   # 为后续在服务端处训练全局原型做准备
        # proto_list = []
        for label in range(self.num_classes):
            proto_list = []
            for client in self.join_clients:
                if label in client.local_protos:
                    self.uploaded_protos.append((client.local_protos[label], label))
                    proto_list.append(client.local_protos[label])

            # proto_list = [client.local_protos[label] for client in self.join_clients
            #               if label in client.local_protos]
            if len(proto_list) == 0:
                continue
            proto = torch.stack(proto_list, dim=0)
            self.reporter.track_upload(proto)
            proto = torch.mean(proto, dim=0)    # 非加权平均，得到类别簇中心原型
            self.avg_protos[label] = proto

        # calculate class-wise minimum distance
        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        for k1 in self.avg_protos.keys():
            for k2 in self.avg_protos.keys():
                if k1 > k2:
                    dis = torch.norm(self.avg_protos[k1] - self.avg_protos[k2], p=2)
                    self.gap[k1] = torch.min(self.gap[k1], dis)
                    self.gap[k2] = torch.min(self.gap[k2], dis)
        self.min_gap = torch.min(self.gap) # calculate class-wise minimum distance
        for i in range(len(self.gap)):
            if self.gap[i] > torch.tensor(1e8, device=self.device):   # 如果某个类别没有上传原型，则使用最小距离
                self.gap[i] = self.min_gap
        self.max_gap = torch.max(self.gap)
        print('class-wise minimum distance', self.gap)
        print('min_gap', self.min_gap)
        print('max_gap', self.max_gap)

        self.update_Gen()   # 在服务端处更新可训练的全局原型

    def update_Gen(self):
        Gen_opt = torch.optim.SGD(self.PROTO.parameters(), lr=self.server_learning_rate)
        self.PROTO.train()
        for e in range(self.server_epochs):
            proto_loader = DataLoader(self.uploaded_protos, self.batch_size,
                                      drop_last=False, shuffle=True)
            for proto, y in proto_loader:
                y = torch.tensor(y).type(torch.int64).to(self.device)

                proto_gen = self.PROTO(list(range(self.num_classes)))   # forward

                features_square = torch.sum(torch.pow(proto, 2), 1, keepdim=True)
                centers_square = torch.sum(torch.pow(proto_gen, 2), 1, keepdim=True)
                features_into_centers = torch.matmul(proto, proto_gen.T)
                dist = features_square - 2 * features_into_centers + centers_square.T
                dist = torch.sqrt(dist)

                one_hot = F.one_hot(y, self.num_classes).to(self.device)
                gap2 = min(self.max_gap.item(), self.margin_threthold)
                dist = dist + one_hot * gap2
                loss = self.CEloss(-dist, y)

                Gen_opt.zero_grad()
                loss.backward()
                Gen_opt.step()

        print(f'Server loss: {loss.item()}')
        self.uploaded_protos = []

        self.PROTO.eval()
        for class_id in range(self.num_classes):
            self.global_protos[class_id] = self.PROTO(torch.tensor(class_id, device=self.device)).detach()


    def train(self):
        acc_record = {}
        for i in range(self.global_rounds):
            self.join_clients = self.select_join_clients()
            print(f"============== global round: {i}, number of join clients: {len(self.join_clients)} ==============")

            for client in self.join_clients:
                client.train()

            self.receive_and_aggregate()
            self.dispatch()

            print("================= evaluate =================")
            acc = self.evaluate(i)
            acc_record[i] = acc
            print(f"================= global round: {i}, accuracy: {acc} =================")

        uploadCom, downloadCom, usedMemory = self.getReport()
        return acc_record, uploadCom, downloadCom, usedMemory
