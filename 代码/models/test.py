import copy
from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.FedCPCL.loss import SupConLoss, GlobalConLoss
from src.client_base import ClientBase
from models.mlp import simpleMLP

class ClientFedCPCL(ClientBase):
    def __init__(self, args, client_id, trainset, testset, statistic):
        super().__init__(args, client_id, trainset, testset, statistic)

        self.model = copy.deepcopy(args.model)

        self.clustering = None

        self.lamb_da = args.client["lambda"]

        self.global_protos = None  # 服务器收到的全局 proto 用户作为正则项
        self.local_protos = {}  # 所有 batch proto 的平均

        self.con_loss = SupConLoss()
        self.global_loss = GlobalConLoss()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        self.mlp = simpleMLP(512).to(device=self.device)
        self.mlp_optimizer = torch.optim.Adam(self.mlp.parameters(), lr=0.001)

    def aggregate(self, raw_feats):
        for label, feats in raw_feats.items():
            feats = F.normalize(torch.stack(feats, dim=0))
            self.local_protos[label] = torch.mean(feats, dim=0)

    def collect_feats(self):
        train_loader = self.get_train_loader()
        self.model.eval()

        raw_feats = defaultdict(list)
        with torch.inference_mode():
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                feature = self.model.base(x)

                for label in torch.unique(y):
                    protos = feature[label == y].detach()
                    raw_feats[label.item()].extend(list(torch.unbind(protos, dim=0)))

        return raw_feats

    def mlpProcessing(self):
        print("单独处理！")
        # received_protos =torch.tensor(self.received_protos)
        # received_protos = self.mlp(received_protos)
        # self.received_protos=received_protos.tolist()

    #     每个单独处理
        for i in range(len(self.global_protos)):
            self.global_protos[i] = self.mlp(self.global_protos[i])

    def train(self):
        train_loader = self.get_train_loader()
        self.model.train()
        for epoch in range(self.local_epochs):
            local_iter = tqdm(train_loader, desc=f"client {self.client_id} epoch: {epoch}")
            sup = 0
            crossLoss=0
            globalLoss=0
            for x, y in local_iter:
                x, y = x.to(self.device), y.to(self.device)
                feature = self.model.base(x)
                output = self.model.head(feature)
                # lossFun = self.loss.to(self.device)
                # loss = lossFun(output, y)
                # y= y.to('cpu')
                # output = output.to('cpu')
                loss =  2*self.loss(output, y)
                crossLoss += 2*self.loss(output, y)

                normalized_feature = F.normalize(feature, dim=1)
                # con_lossFun =self.con_loss.to(self.device)
                # loss += self.lamb_da * con_lossFun(normalized_feature, labels=y)
                # normalized_feature=normalized_feature.to('cpu')
                loss += 1 * self.con_loss(normalized_feature, labels=y)
                sup +=  1 * self.con_loss(normalized_feature, labels=y)
                # print("SupConloss:",sup)

                if self.global_protos is not None:
                    # self.mlpProcessing()
                    self.processed_global_protos = []
                    for proto in self.global_protos:
                        processed_proto = self.mlp(proto.to(self.device))
                        self.processed_global_protos.append(processed_proto)
                    regulation_protos = [None] * self.num_classes
                    for label, proto in self.local_protos.items():
                        similarity = torch.matmul(proto, torch.stack(self.processed_global_protos, dim=0).T)
                        index = torch.argmax(similarity)
                        regulation_protos[label] = self.processed_global_protos[index]

                    available_index = 0
                    for index, proto in enumerate(self.processed_global_protos):
                        if index in list(self.local_protos.keys()):
                            continue
                        while (available_index < self.num_classes and
                               regulation_protos[available_index] is not None):
                            available_index += 1
                        if available_index >= self.num_classes:
                            break
                        regulation_protos[available_index] = proto


                    # 没用mlp
                if self.global_protos is not None:
                    regulation_protos = [None] * self.num_classes
                    for label, proto in self.local_protos.items():
                        similarity = torch.matmul(proto, torch.stack(self.global_protos, dim=0).T)
                        index = torch.argmax(similarity)
                        regulation_protos[label] = self.global_protos[index]

                    available_index = 0
                    for index, proto in enumerate(self.global_protos):
                        if index in list(self.local_protos.keys()):
                            continue
                        while (available_index < self.num_classes and
                               regulation_protos[available_index] is not None):
                            available_index += 1
                        if available_index >= self.num_classes:
                            break
                        regulation_protos[available_index] = proto

                    loss += 0.1 * self.global_loss(normalized_feature,
                                                            labels=y,
                                                            global_protos=regulation_protos)
                    globalLoss  += 0.1 *self.global_loss(normalized_feature,
                                                            labels=y,
                                                            global_protos=regulation_protos)

                self.optimizer.zero_grad()
                # self.mlp_optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # self.mlp_optimizer.step()

                local_iter.set_description(f"client {self.client_id} epoch: {epoch} loss: {loss.item():.4f}")
            print("sup:",sup)
            print("crossLoss:",crossLoss)
            print("globalLoss:",globalLoss)
        raw_feats = self.collect_feats()
        self.aggregate(raw_feats)

