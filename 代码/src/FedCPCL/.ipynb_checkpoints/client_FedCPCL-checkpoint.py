# import copy
# from collections import defaultdict
#
# import torch
# import torch.nn.functional as F
# from tqdm import tqdm
#
# from src.FedCPCL.loss import SupConLoss, GlobalConLoss
# from src.client_base import ClientBase
#
#
# class ClientFedCPCL(ClientBase):
#     def __init__(self, args, client_id, trainset, testset, statistic):
#         super().__init__(args, client_id, trainset, testset, statistic)
#
#         self.model = copy.deepcopy(args.model)
#
#         self.clustering = None
#
#         self.lamb_da = args.client["lambda"]
#
#         self.global_protos = None  # 服务器收到的全局 proto 用户作为正则项
#         self.local_protos = {}  # 所有 batch proto 的平均
#
#         self.con_loss = SupConLoss()
#         self.global_loss = GlobalConLoss()
#
#         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
#
#     def aggregate(self, raw_feats):
#         for label, feats in raw_feats.items():
#             feats = F.normalize(torch.stack(feats, dim=0))
#             self.local_protos[label] = torch.mean(feats, dim=0)
#
#     def collect_feats(self):
#         train_loader = self.get_train_loader()
#         self.model.eval()
#
#         raw_feats = defaultdict(list)
#         with torch.inference_mode():
#             for x, y in train_loader:
#                 x, y = x.to(self.device), y.to(self.device)
#                 feature = self.model.base(x)
#
#                 for label in torch.unique(y):
#                     protos = feature[label == y].detach()
#                     raw_feats[label.item()].extend(list(torch.unbind(protos, dim=0)))
#
#         return raw_feats
#
#     def train(self):
#         train_loader = self.get_train_loader()
#         self.model.train()
#
#         for epoch in range(self.local_epochs):
#             local_iter = tqdm(train_loader, desc=f"client {self.client_id} epoch: {epoch}")
#             for x, y in local_iter:
#                 x, y = x.to(self.device), y.to(self.device)
#                 feature = self.model.base(x)
#                 output = self.model.head(feature)
#                 loss = self.loss(output, y)
#
#                 normalized_feature = F.normalize(feature, dim=1)
#                 loss += self.lamb_da * self.con_loss(normalized_feature, labels=y)
#
#                 if self.global_protos is not None:
#                     regulation_protos = [None] * self.num_classes
#                     for label, proto in self.local_protos.items():
#                         similarity = torch.matmul(proto, torch.stack(self.global_protos, dim=0).T)
#                         index = torch.argmax(similarity)
#                         regulation_protos[label] = self.global_protos[index]
#
#                     available_index = 0
#                     for index, proto in enumerate(self.global_protos):
#                         if index in list(self.local_protos.keys()):
#                             continue
#                         while (available_index < self.num_classes and
#                                regulation_protos[available_index] is not None):
#                             available_index += 1
#                         if available_index >= self.num_classes:
#                             break
#                         regulation_protos[available_index] = proto
#
#                     loss += self.lamb_da * self.global_loss(normalized_feature,
#                                                             labels=y,
#                                                             global_protos=regulation_protos)
#
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#
#                 local_iter.set_description(f"client {self.client_id} epoch: {epoch} loss: {loss.item():.4f}")
#
#         raw_feats = self.collect_feats()
#         self.aggregate(raw_feats)




import copy
from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.FedCPCL.loss import SupConLoss, GlobalConLoss,DPNLoss
from src.client_base import ClientBase


class ClientFedCPCL(ClientBase):
    def __init__(self, args, client_id, trainset, testset, statistic):
        super().__init__(args, client_id, trainset, testset, statistic)

        self.clustering = None

        self.lamb_da = args.client["lambda"]
        self.belta = args.client["belt"]

        self.global_protos = None  # 服务器收到的全局 proto 用户作为正则项
        self.local_protos = {}  # 所有 batch proto 的平均

        self.con_loss = SupConLoss()
        self.global_loss = GlobalConLoss()
        self.global_loss_emb = DPNLoss()
        self.global_loss_pro = DPNLoss()
        self.proto_weight = {}
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)

    def aggregate(self, raw_feats,raw_scores):
        for label, feats in raw_feats.items():
            feats = F.normalize(torch.stack(feats, dim=0))
            score = torch.tensor(raw_scores[label])
            self.local_protos[label] = torch.mean(feats, dim=0)
            # weight = score / score.sum()
            # local_proto = torch.sum(weight.view(-1, 1).to(self.device) * feats, dim=0)
            # self.local_protos[label] = local_proto
            score_proto = F.softmax(self.model.head(self.local_protos[label].unsqueeze(0)), dim=1)


            # one_hot_label = torch.zeros(self.num_classes)
            # one_hot_label[label] = 1.0
            # kl_div = F.kl_div(score_proto.log().cpu(), one_hot_label, reduction='batchmean').detach()
            # kl_weights = 1.0 / (kl_div + 1e-10)
            # self.proto_weight[label] = kl_weights

            self.proto_weight[label] = score_proto[0, label].cpu().detach()

    def collect_feats(self):
        train_loader = self.get_train_loader()
        self.model.eval()

        raw_feats = defaultdict(list)
        raw_scores = defaultdict(list)
        with torch.inference_mode():
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                feature = self.model.base(x)
                score = F.softmax(self.model.head(feature), dim=1)

                for label in torch.unique(y):
                    protos = feature[label == y].detach()

                    raw_feats[label.item()].extend(list(torch.unbind(protos, dim=0)))
                    raw_scores[label.item()].extend(score[label == y][:, label].detach().cpu().detach())

            # Convert lists to numpy arrays
            # import numpy as np
            # for label in raw_scores:
            #     raw_scores[label] = np.array(raw_scores[label])

        return raw_feats,raw_scores
        # train_loader = self.get_train_loader()
        # self.model.eval()
        #
        # raw_feats = defaultdict(list)
        # with torch.inference_mode():
        #     for x, y in train_loader:
        #         x, y = x.to(self.device), y.to(self.device)
        #         feature = self.model.base(x)
        #
        #         for label in torch.unique(y):
        #             protos = feature[label == y].detach()
        #             raw_feats[label.item()].extend(list(torch.unbind(protos, dim=0)))
        #
        # return raw_feats

    def train(self):
        train_loader = self.get_train_loader()
        self.model.train()

        for epoch in range(self.local_epochs):
            local_iter = tqdm(train_loader, desc=f"client {self.client_id} epoch: {epoch}")
            for x, y in local_iter:
                x, y = x.to(self.device), y.to(self.device)
                feature = self.model.base(x)
                output = self.model.head(feature)
                loss = self.loss(output, y)

                normalized_feature = F.normalize(feature, dim=1)
                loss += self.lamb_da * self.con_loss(normalized_feature, labels=y)

                # print("Only")
                # if self.global_protos is not None:
                #     regulation_protos = [None] * self.num_classes
                #     for label, proto in self.local_protos.items():
                #         similarity = torch.matmul(proto, torch.stack(self.global_protos, dim=0).T)
                #         index = torch.argmax(similarity)
                #         regulation_protos[label] = self.global_protos[index]
                #
                #     available_index = 0
                #     for index, proto in enumerate(self.global_protos):
                #         if index in list(self.local_protos.keys()):
                #             continue
                #         while (available_index < self.num_classes and
                #                regulation_protos[available_index] is not None):
                #             available_index += 1
                #         if available_index >= self.num_classes:
                #             break
                #         regulation_protos[available_index] = proto
                #
                #     loss += self.lamb_da * self.global_loss(normalized_feature,
                #                                             labels=y,
                #                                             global_protos=regulation_protos)

                if self.global_protos is not None:
                    loss += self.belta *self.global_loss_emb(normalized_feature,global_protos=self.global_protos)

                    # loss +=self.belta * self.getDPNProtoLoss(normalized_feature,y)
                    local_proto_list = [v for k, v in self.local_protos.items()]
                    loss +=self.belta * self.global_loss_pro(local_proto_list,global_protos=self.global_protos)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                local_iter.set_description(f"client {self.client_id} epoch: {epoch} loss: {loss.item():.4f}")

        raw_feats,raw_scores = self.collect_feats()
        self.aggregate(raw_feats,raw_scores)
