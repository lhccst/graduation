# import time
# import numpy as np
# import random
# import torch
# import cvxpy as cvx
# import copy
# from threading import Thread
# from collections import defaultdict
# from mem_utils import MemReporter
# from src.FedPAC.client_FedPAC import ClientFedPAC
# from src.server_base import ServerBase
#
# class ServerFedPAC(ServerBase):
#     def __init__(self, args):
#         super().__init__(args)
#
#         self.model = args.model
#         self.initialize_clients(ClientFedPAC)
#
#         self.aggregated = self.args.model.base.state_dict()
#
#         self.num_classes = args.num_classes
#         self.global_protos = [None for _ in range(args.num_classes)]
#
#         self.Vars = []
#         self.Hs = []
#         self.uploaded_heads = []
#
#     def dispatch(self):
#         for client in self.join_clients:
#             client.model.base.load_state_dict(self.aggregated)
#             client.global_protos = self.global_protos
#
#     def receive_and_aggregate(self):
#         """ receive and aggregate feature extractor parameter and local proto """
#         client_models = []
#         client_sample_sizes = []
#         for client in self.join_clients:
#             client_models.append(client.model.base.state_dict())
#             client_sample_sizes.append(client.sample_size)
#         size_tensor = torch.tensor(client_sample_sizes, dtype=torch.float).to(self.device)
#         client_weight = size_tensor / torch.sum(size_tensor)
#
#         for key in self.aggregated.keys():
#             param = [client_model[key] for client_model in client_models]
#             param = torch.stack(param, dim=0)
#             shape = [-1] + [1] * (len(param.shape) - 1)
#             self.aggregated[key] = torch.sum(param * client_weight.reshape(shape), dim=0)
#
#         for label in range(self.num_classes):
#             proto_list = [client.protos[label] for client in self.join_clients
#                           if label in client.protos]
#             if len(proto_list) == 0:
#                 continue
#             proto = torch.stack(proto_list, dim=0)
#             client_sample = [client.statistic[str(label)] for client in self.join_clients
#                              if label in client.protos]
#             weight = torch.tensor(client_sample).to(self.device)
#             weight = weight / torch.sum(weight)
#             proto = torch.sum(proto * weight.reshape(-1, 1), dim=0)
#             self.global_protos[label] = proto
#
#
#     def train(self):
#         self.reporter = MemReporter()
#         acc_record = {}
#         for i in range(self.global_rounds):
#             self.join_clients = self.select_join_clients()
#             print(f"============== global round: {i}, number of join clients: {len(self.join_clients)} ==============")
#
#             self.Vars = []
#             self.Hs = []
#             for client in self.join_clients:
#                 V, h = client.statistics_extraction()
#                 self.Vars.append(V)
#                 self.Hs.append(h)
#
#             for client in self.join_clients:
#                 client.train()
#
#             # proto and feature extractor parameter
#             self.receive_and_aggregate()
#             self.dispatch()
#
#             # classifier head parameter
#             self.aggregate_and_send_heads()
#
#             print("================= evaluate =================")
#             acc = self.evaluate()
#             acc_record[i] = acc
#             print(f"================= global round: {i}, accuracy: {acc} =================")
#
#         uploadCom,downloadCom=self.reporter.print_communication_stats()
#         return acc_record,uploadCom,downloadCom
#
#     def aggregate_and_send_heads(self):
#         head_weights = solve_quadratic(len(self.join_clients), self.Vars, self.Hs)
#
#         for idx, client in enumerate(self.join_clients):
#             print('(Client {}) Weights of Classifier Head'.format(idx))
#             print(head_weights[idx],'\n')
#
#             # if head_weights[idx] is not None:
#             #     new_head = self.add_heads(head_weights[idx])
#             #     client.set_head(new_head)
#
#             if head_weights[idx] is not None:
#                 new_head = self.add_heads(head_weights[idx])
#             else:
#                 new_head = self.uploaded_heads[client]
#             client.set_head(new_head)
#
#     def add_heads(self, weights):
#         new_head = copy.deepcopy(self.model.head)
#         for param in new_head.parameters():
#             param.data.zero_()
#
#         for client, w in zip(self.join_clients, weights):
#             head = client.model.head
#             for server_param, client_param in zip(new_head.parameters(), head.parameters()):
#                 server_param.data += client_param.data.clone() * w
#
#         # for w, head in zip(weights, self.uploaded_heads):
#         #     for server_param, client_param in zip(new_head.parameters(), head.parameters()):
#         #         server_param.data += client_param.data.clone() * w
#
#         return new_head
#
#
# # https://github.com/JianXu95/FedPAC/blob/main/tools.py#L94
# def solve_quadratic(num_users, Vars, Hs):
#     device = Hs[0][0].device
#     num_cls = Hs[0].shape[0] # number of classes
#     d = Hs[0].shape[1] # dimension of feature representation
#     avg_weight = []
#     for i in range(num_users):
#         # ---------------------------------------------------------------------------
#         # variance ter
#         v = torch.tensor(Vars, device=device)
#         # ---------------------------------------------------------------------------
#         # bias term
#         h_ref = Hs[i]
#         dist = torch.zeros((num_users, num_users), device=device)
#         for j1, j2 in pairwise(tuple(range(num_users))):
#             h_j1 = Hs[j1]
#             h_j2 = Hs[j2]
#             h = torch.zeros((d, d), device=device)
#             for k in range(num_cls):
#                 h += torch.mm((h_ref[k]-h_j1[k]).reshape(d,1), (h_ref[k]-h_j2[k]).reshape(1,d))
#             dj12 = torch.trace(h)
#             dist[j1][j2] = dj12
#             dist[j2][j1] = dj12
#
#         # QP solver
#         p_matrix = torch.diag(v) + dist
#         p_matrix = p_matrix.cpu().numpy()  # coefficient for QP problem
#         evals, evecs = torch.linalg.eig(torch.tensor(p_matrix))
#
#         # for numerical stablity
#         p_matrix_new = 0
#         p_matrix_new = 0
#         for ii in range(num_users):
#             if evals[ii].real >= 0.01:
#                 p_matrix_new += evals[ii].real*torch.mm(evecs[:,ii].reshape(num_users,1), evecs[:,ii].reshape(1, num_users))
#         p_matrix = p_matrix_new.numpy() if not np.all(np.linalg.eigvals(p_matrix)>=0.0) else p_matrix
#
#         # solve QP
#         alpha = 0
#         eps = 1e-3
#         if np.all(np.linalg.eigvals(p_matrix)>=0):
#             alphav = cvx.Variable(num_users)
#             obj = cvx.Minimize(cvx.quad_form(alphav, cvx.psd_wrap(p_matrix)))
#             prob = cvx.Problem(obj, [cvx.sum(alphav) == 1.0, alphav >= 0])
#             prob.solve()
#             alpha = alphav.value
#             alpha = [(i)*(i>eps) for i in alpha] # zero-out small weights (<eps)
#         else:
#             alpha = None # if no solution for the optimization problem, use local classifier only
#
#         avg_weight.append(alpha)
#
#     return avg_weight
#
# # https://github.com/JianXu95/FedPAC/blob/main/tools.py#L10
# def pairwise(data):
#     n = len(data)
#     for i in range(n):
#         for j in range(i, n):
#             yield (data[i], data[j])
#
#

import time
import numpy as np
import random
import torch
import cvxpy as cvx
import copy
from threading import Thread
from collections import defaultdict

from src.FedPAC.client_FedPAC import ClientFedPAC
from src.server_base import ServerBase

from mem_utils import MemReporter
class ServerFedPAC(ServerBase):
    def __init__(self, args):
        super().__init__(args)

        self.model = args.model
        self.initialize_clients(ClientFedPAC)

        self.aggregated = self.args.model.base.state_dict()

        self.num_classes = args.num_classes
        self.global_protos = [None for _ in range(args.num_classes)]

        self.Vars = []
        self.Hs = []
        self.uploaded_heads = []

    def dispatch(self):
        for client in self.join_clients:
            client.model.base.load_state_dict(self.aggregated)
            self.reporter.track_download(self.aggregated)
            client.global_protos = self.global_protos
            self.reporter.track_download(self.global_protos)

    def receive_and_aggregate(self):
        """ receive and aggregate feature extractor parameter and local proto """
        client_models = []
        client_sample_sizes = []
        for client in self.join_clients:
            client_models.append(client.model.base.state_dict())
            self.reporter.track_upload(client.model.base.state_dict())
            client_sample_sizes.append(client.sample_size)
        size_tensor = torch.tensor(client_sample_sizes, dtype=torch.float).to(self.device)
        client_weight = size_tensor / torch.sum(size_tensor)
        self.reporter.track_upload(size_tensor)
        for key in self.aggregated.keys():
            param = [client_model[key] for client_model in client_models]
            param = torch.stack(param, dim=0)
            shape = [-1] + [1] * (len(param.shape) - 1)
            self.aggregated[key] = torch.sum(param * client_weight.reshape(shape), dim=0)

        for label in range(self.num_classes):
            proto_list = [client.protos[label] for client in self.join_clients
                          if label in client.protos]
            if len(proto_list) == 0:
                continue
            proto = torch.stack(proto_list, dim=0)
            self.reporter.track_upload(proto)
            client_sample = [client.statistic[str(label)] for client in self.join_clients
                             if label in client.protos]
            weight = torch.tensor(client_sample).to(self.device)
            weight = weight / torch.sum(weight)
            proto = torch.sum(proto * weight.reshape(-1, 1), dim=0)
            self.global_protos[label] = proto

    def train(self):
        self.reporter = MemReporter()
        acc_record = {}
        for i in range(self.global_rounds):
            self.join_clients = self.select_join_clients()
            print(f"============== global round: {i}, number of join clients: {len(self.join_clients)} ==============")

            self.Vars = []
            self.Hs = []
            for client in self.join_clients:
                V, h = client.statistics_extraction()
                self.Vars.append(V)
                self.Hs.append(h)

            for client in self.join_clients:
                client.train()

            # proto and feature extractor parameter
            self.receive_and_aggregate()
            self.dispatch()

            # classifier head parameter
            self.aggregate_and_send_heads()

            print("================= evaluate =================")
            acc = self.evaluate()
            acc_record[i] = acc
            print(f"================= global round: {i}, accuracy: {acc} =================")

        uploadCom,downloadCom=self.reporter.print_communication_stats()
        return acc_record,uploadCom,downloadCom

    def aggregate_and_send_heads(self):
        head_weights = solve_quadratic(len(self.join_clients), self.Vars, self.Hs)

        for idx, client in enumerate(self.join_clients):
            print('(Client {}) Weights of Classifier Head'.format(idx))
            print(head_weights[idx], '\n')

            if head_weights[idx] is not None:
                new_head = self.add_heads(head_weights[idx])
                client.set_head(new_head)
                self.reporter.track_download(new_head)

    def add_heads(self, weights):
        new_head = copy.deepcopy(self.model.head)
        for param in new_head.parameters():
            param.data.zero_()

        for client, w in zip(self.join_clients, weights):
            head = client.model.head
            self.reporter.track_upload(head)
            for server_param, client_param in zip(new_head.parameters(), head.parameters()):
                server_param.data += client_param.data.clone() * w

        return new_head


# https://github.com/JianXu95/FedPAC/blob/main/tools.py#L94
def solve_quadratic(num_users, Vars, Hs):
    device = Hs[0][0].device
    num_cls = Hs[0].shape[0]  # number of classes
    d = Hs[0].shape[1]  # dimension of feature representation
    avg_weight = []
    for i in range(num_users):
        # ---------------------------------------------------------------------------
        # variance ter
        v = torch.tensor(Vars, device=device)
        # ---------------------------------------------------------------------------
        # bias term
        h_ref = Hs[i]
        dist = torch.zeros((num_users, num_users), device=device)
        for j1, j2 in pairwise(tuple(range(num_users))):
            h_j1 = Hs[j1]
            h_j2 = Hs[j2]
            h = torch.zeros((d, d), device=device)
            for k in range(num_cls):
                h += torch.mm((h_ref[k] - h_j1[k]).reshape(d, 1), (h_ref[k] - h_j2[k]).reshape(1, d))
            dj12 = torch.trace(h)
            dist[j1][j2] = dj12
            dist[j2][j1] = dj12

        # QP solver
        p_matrix = torch.diag(v) + dist
        p_matrix = p_matrix.cpu().numpy()  # coefficient for QP problem
        evals, evecs = torch.linalg.eig(torch.tensor(p_matrix))

        # for numerical stablity
        p_matrix_new = 0
        p_matrix_new = 0
        for ii in range(num_users):
            if evals[ii].real >= 0.01:
                p_matrix_new += evals[ii].real * torch.mm(evecs[:, ii].reshape(num_users, 1),
                                                          evecs[:, ii].reshape(1, num_users))
        p_matrix = p_matrix_new.numpy() if not np.all(np.linalg.eigvals(p_matrix) >= 0.0) else p_matrix

        # solve QP
        alpha = 0
        eps = 1e-3
        if np.all(np.linalg.eigvals(p_matrix) >= 0):
            alphav = cvx.Variable(num_users)
            obj = cvx.Minimize(cvx.quad_form(alphav, cvx.psd_wrap(p_matrix)))
            prob = cvx.Problem(obj, [cvx.sum(alphav) == 1.0, alphav >= 0])
            prob.solve()
            alpha = alphav.value
            alpha = [(i) * (i > eps) for i in alpha]  # zero-out small weights (<eps)
        else:
            alpha = None  # if no solution for the optimization problem, use local classifier only

        avg_weight.append(alpha)

    return avg_weight


# https://github.com/JianXu95/FedPAC/blob/main/tools.py#L10
def pairwise(data):
    n = len(data)
    for i in range(n):
        for j in range(i, n):
            yield (data[i], data[j])

