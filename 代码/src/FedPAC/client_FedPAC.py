# import copy
# import torch
# import torch.nn as nn
# import numpy as np
# import time
# from collections import defaultdict
# from tqdm import tqdm
#
# from src.client_base import ClientBase
#
#
# class ClientFedPAC(ClientBase):
#     def __init__(self, args, client_id, trainset, testset, statistic):
#         super().__init__(args, client_id, trainset, testset, statistic)
#
#         self.model = copy.deepcopy(args.model)
#
#         self.protos = None
#         self.global_protos = None
#         self.loss_mse = nn.MSELoss()
#
#         self.classifier_lr = args.client["classifier_lr"]
#         self.feature_extractor_lr = args.client["feature_extractor_lr"]
#         self.lamda = args.client["lamda"]
#
#     def collect_protos(self):
#         trainloader = self.get_train_loader()
#         self.model.eval()
#
#         protos = defaultdict(list)
#         with torch.no_grad():
#             for i, (x, y) in enumerate(trainloader):
#                 if type(x) == type([]):
#                     x[0] = x[0].to(self.device)
#                 else:
#                     x = x.to(self.device)
#                 y = y.to(self.device)
#                 rep = self.model.base(x)
#
#                 for i, yy in enumerate(y):
#                     y_c = yy.item()
#                     protos[y_c].append(rep[i, :].detach().data)
#
#         self.protos = self.agg_func(protos)  # local protos
#
#     # https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L205
#     def agg_func(self, protos):
#         """
#         Returns the average of the weights.
#         """
#
#         for [label, proto_list] in protos.items():
#             if len(proto_list) > 1:
#                 proto = 0 * proto_list[0].data
#                 for i in proto_list:
#                     proto += i.data
#                 protos[label] = proto / len(proto_list)
#             else:
#                 protos[label] = proto_list[0]
#
#         return protos
#
#     def set_head(self, head):
#         for new_param, old_param in zip(head.parameters(), self.model.head.parameters()):
#             old_param.data = new_param.data.clone()
#
#     def train(self):
#         train_loader = self.get_train_loader()
#         self.model.train()
#
#         # firstly train clssifier
#         for param in self.model.base.parameters():
#             param.requires_grad = False
#         for param in self.model.head.parameters():
#             param.requires_grad = True
#
#
#         self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.classifier_lr, momentum=0.9)
#         local_iter = tqdm(train_loader, desc=f"client {self.client_id} classifier's epoch")
#         for x, y in local_iter:
#             x, y = x.to(self.device), y.to(self.device)
#
#             rep = self.model.base(x)
#             output = self.model.head(rep)
#             loss = self.loss(output, y)
#
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()
#
#         # then train feature extractor
#         for param in self.model.base.parameters():
#             param.requires_grad = True
#         for param in self.model.head.parameters():
#             param.requires_grad = False
#
#         self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.feature_extractor_lr, momentum=0.9)
#         for epoch in range(self.local_epochs):
#             local_iter = tqdm(train_loader, desc=f"client {self.client_id} feature extractor epoch {epoch}" )
#             for x, y in local_iter:
#                 x, y = x.to(self.device), y.to(self.device)
#
#                 rep = self.model.base(x)
#                 output = self.model.head(rep)
#                 loss = self.loss(output, y)
#
#                 if self.global_protos is not None:
#                     proto_new = copy.deepcopy(rep.detach())
#                     for i, yy in enumerate(y):
#                         y_c = yy.item()
#                         if self.global_protos[y_c] is not None:
#                             proto_new[i, :] = self.global_protos[y_c].data
#                     loss += self.loss_mse(proto_new, rep) * self.lamda
#
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#
#                 local_iter.set_description(f"client {self.client_id} epoch: {epoch} loss: {loss.item():.4f}")
#
#         self.collect_protos()
#
#
#     # https://github.com/JianXu95/FedPAC/blob/main/methods/fedpac.py#L126
#     def statistics_extraction(self):
#         model = self.model
#         trainloader = self.get_train_loader()
#         for x, y in trainloader:
#             if type(x) == type([]):
#                 x[0] = x[0].to(self.device)
#             else:
#                 x = x.to(self.device)
#             y = y.to(self.device)
#             with torch.no_grad():
#                 rep = model.base(x).detach()
#             break
#         d = rep.shape[1]
#         feature_dict = {}
#         with torch.no_grad():
#             for x, y in trainloader:
#                 if type(x) == type([]):
#                     x[0] = x[0].to(self.device)
#                 else:
#                     x = x.to(self.device)
#                 y = y.to(self.device)
#                 features = model.base(x)
#                 feat_batch = features.clone().detach()
#                 for i in range(len(y)):
#                     yi = y[i].item()
#                     if yi in feature_dict.keys():
#                         feature_dict[yi].append(feat_batch[i,:])
#                     else:
#                         feature_dict[yi] = [feat_batch[i,:]]
#         for k in feature_dict.keys():
#             feature_dict[k] = torch.stack(feature_dict[k])
#
#         py = torch.zeros(self.num_classes)
#         for x, y in trainloader:
#             for yy in y:
#                 py[yy.item()] += 1
#         py = py / torch.sum(py)
#         py2 = py.mul(py)
#         v = 0
#         h_ref = torch.zeros((self.num_classes, d), device=self.device)
#         for k in range(self.num_classes):
#             if k in feature_dict.keys():
#                 feat_k = feature_dict[k]
#                 num_k = feat_k.shape[0]
#                 feat_k_mu = feat_k.mean(dim=0)
#                 h_ref[k] = py[k]*feat_k_mu
#                 v += (py[k]*torch.trace((torch.mm(torch.t(feat_k), feat_k)/num_k))).item()
#                 v -= (py2[k]*(torch.mul(feat_k_mu, feat_k_mu))).sum().item()
#         v = v/self.sample_size
#
#         return v, h_ref
#


import copy
import torch
import torch.nn as nn
import numpy as np
import time
from collections import defaultdict
from tqdm import tqdm

from src.client_base import ClientBase


class ClientFedPAC(ClientBase):
    def __init__(self, args, client_id, trainset, testset, statistic):
        super().__init__(args, client_id, trainset, testset, statistic)

        self.model = copy.deepcopy(args.model)

        self.protos = None
        self.global_protos = None
        self.loss_mse = nn.MSELoss()

        self.classifier_lr = args.client["classifier_lr"]
        self.feature_extractor_lr = args.client["feature_extractor_lr"]
        self.lamda = args.client["lamda"]

    def collect_protos(self):
        trainloader = self.get_train_loader()
        self.model.eval()

        protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

        self.protos = self.agg_func(protos)  # local protos

    # https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L205
    def agg_func(self, protos):
        """
        Returns the average of the weights.
        """

        for [label, proto_list] in protos.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                protos[label] = proto / len(proto_list)
            else:
                protos[label] = proto_list[0]

        return protos

    def set_head(self, head):
        for new_param, old_param in zip(head.parameters(), self.model.head.parameters()):
            old_param.data = new_param.data.clone()

    def train(self):
        train_loader = self.get_train_loader()
        self.model.train()

        # firstly train clssifier
        for param in self.model.base.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True

        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                         lr=self.classifier_lr, momentum=0.9)
        local_iter = tqdm(train_loader, desc=f"client {self.client_id} classifier's epoch")
        for x, y in local_iter:
            x, y = x.to(self.device), y.to(self.device)

            rep = self.model.base(x)
            output = self.model.head(rep)
            loss = self.loss(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # then train feature extractor
        for param in self.model.base.parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                         lr=self.feature_extractor_lr, momentum=0.9)
        for epoch in range(self.local_epochs):
            local_iter = tqdm(train_loader, desc=f"client {self.client_id} feature extractor epoch {epoch}")
            for x, y in local_iter:
                x, y = x.to(self.device), y.to(self.device)

                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)

                if self.global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if self.global_protos[y_c] is not None:
                            proto_new[i, :] = self.global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                local_iter.set_description(f"client {self.client_id} epoch: {epoch} loss: {loss.item():.4f}")

        self.collect_protos()

    # https://github.com/JianXu95/FedPAC/blob/main/methods/fedpac.py#L126
    def statistics_extraction(self):
        model = self.model
        trainloader = self.get_train_loader()
        for x, y in trainloader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                rep = model.base(x).detach()
            break
        d = rep.shape[1]
        feature_dict = {}
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                features = model.base(x)
                feat_batch = features.clone().detach()
                for i in range(len(y)):
                    yi = y[i].item()
                    if yi in feature_dict.keys():
                        feature_dict[yi].append(feat_batch[i, :])
                    else:
                        feature_dict[yi] = [feat_batch[i, :]]
        for k in feature_dict.keys():
            feature_dict[k] = torch.stack(feature_dict[k])

        py = torch.zeros(self.num_classes)
        for x, y in trainloader:
            for yy in y:
                py[yy.item()] += 1
        py = py / torch.sum(py)
        py2 = py.mul(py)
        v = 0
        h_ref = torch.zeros((self.num_classes, d), device=self.device)
        for k in range(self.num_classes):
            if k in feature_dict.keys():
                feat_k = feature_dict[k]
                num_k = feat_k.shape[0]
                feat_k_mu = feat_k.mean(dim=0)
                h_ref[k] = py[k] * feat_k_mu
                v += (py[k] * torch.trace((torch.mm(torch.t(feat_k), feat_k) / num_k))).item()
                v -= (py2[k] * (torch.mul(feat_k_mu, feat_k_mu))).sum().item()
        v = v / self.sample_size

        return v, h_ref


