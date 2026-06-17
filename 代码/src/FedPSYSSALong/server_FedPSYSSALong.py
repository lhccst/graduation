# server_fedssa.py
import copy

import torch
import numpy as np
from collections import defaultdict
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from src.server_base import ServerBase
from mem_utils import MemReporter
from src.FedPSYSSALong.client_FedPSYSSALong import ClientFedPSYSSALong
from src.FedTGP.Trainable_prototypes import Trainable_prototypes
class ServerFedPSYSSALong(ServerBase):
    def __init__(self, args):
        super().__init__(args)

        # 初始化全局共享参数
        self.shared_params = copy.deepcopy(args.head.state_dict())
        self.aggregatedHead =  copy.deepcopy(args.head.state_dict())
        self.initialize_clients(ClientFedPSYSSALong)
        self.class_distribution = defaultdict(list)
        self.class_accuracy = defaultdict(list)
        # self.class_num = defaultdict(list)
        self.sample_weights = [] ##每个客户端的样本数
        self.lambdaWeight = args.server['lambdaWeight']
        # self.aggregatedProjection = self.args.projection.state_dict()

        # 设置超参数
        # self.decay_rounds = 2
        # self.miu_0 = 0.5

        self.collect_class_distribution() ##收集类分布 {类0：客户端[1,2,3],类1：客户端[2,3]}
        # self.global_protos = [None] * self.num_classes
        # self.avg_protos = defaultdict(list)
        # self.server_hidden_dim = args.feature_dim
        # self.PROTO =  Trainable_prototypes(     # 服务端处训练的原型
        #         self.num_classes,
        #         self.server_hidden_dim,
        #         args.feature_dim,
        #         self.device
        #     ).to(self.device)
        # self.server_learning_rate =0.01
        # self.server_epochs= 100
        # self.batch_size = 10
        # self.margin_threthold=100
        # self.CEloss = nn.CrossEntropyLoss()
        # self.collect_class_accuracy()
        # self.collect_class_num()


    # def receive_and_aggregate(self):
    #     for label in range(self.num_classes):
    #         proto_list = [client.local_protos[label] for client in self.join_clients
    #                       if label in client.local_protos]
    #         if len(proto_list) == 0:
    #             continue
    #         proto = torch.stack(proto_list, dim=0)
    #         self.reporter.track_upload(proto)
    #         client_sample = [client.statistic[str(label)] for client in self.join_clients
    #                          if label in client.local_protos]
    #         weight = torch.tensor(client_sample).to(self.device)
    #         self.reporter.track_upload(weight)
    #         weight = weight / torch.sum(weight)
    #         proto = torch.sum(proto * weight.reshape(-1, 1), dim=0)
    #         self.global_protos[label] = proto


    def collect_class_distribution(self):
        """收集各客户端的类别分布"""
        for client in self.clients:
            self.sample_weights.append(client.sample_size)
            for cls in client.owned_classes:
                self.class_distribution[cls].append(client.client_id)
        self.reporter.track_upload(self.sample_weights)
    def collect_class_num(self):
        for client in self.clients:
            self.class_num[client.client_id]=client.statistic

    def collect_class_accuracy(self):
        """收集各客户端的类别准确率"""
        for client in self.clients:
            self.class_accuracy[client.client_id]=client.accuracyClass
        self.reporter.track_upload(self.class_accuracy)
    # def collect_class_accuracy_lhc(self):
    #     """收集各客户端的类别准确率"""
    #     for client in self.clients:
    #         self.class_accuracy[client.client_id]=client.avg_scores
    #     self.reporter.track_upload(self.class_accuracy)

    # def collect_class_distribution(self):
    #     """收集每个客户端上每个类的数量"""
    #     for client in self.clients:
    #         self.sample_weights.append(client.sample_size)
    #         # 遍历每个客户端拥有的类别
    #         for cls in client.owned_classes:
    #             # 如果客户端 ID 没有对应条目，初始化为一个空字典
    #             if client.client_id not in self.class_distribution:
    #                 self.class_distribution[client.client_id] = {}
    #             # 将该类在客户端上的数量记录下来
    #             self.class_distribution[client.client_id][cls] = self.class_distribution[client.client_id].get(cls, 0) + \
    #                                                              client.class_counts[cls]

    def aggregate_shared_params(self):
        # """聚合共享参数"""
        # aggregated = self.args.head.state_dict()
        # # 收集所有客户端的共享参数
        # for client in self.join_clients:
        #     client_params = client.head.state_dict()
        #     for key in client.shared_keys:
        #         for cls in client.owned_classes:
        #             aggregated[key][cls].append(client_params[key][cls])
        #
        # # 加权平均聚合
        # for key in self.aggregatedHead.keys():
        #     for cls in aggregated[key]:
        #         params = torch.stack(aggregated[key][cls])
        #         weights = torch.tensor([
        #             self.sample_weights[cid]
        #             for cid in self.class_distribution[cls]
        #         ]).to(self.device)
        #         weights /= weights.sum()
        #
        #         # 扩展维度进行加权求和
        #         shape = [-1] + [1] * (len(params.shape) - 1)
        #         self.aggregatedHead[key][cls] = torch.sum(
        #             params * weights.view(*shape),
        #             dim=0
        #         )
        """聚合共享参数"""
        aggregated = defaultdict(lambda: defaultdict(list)) ##按类聚合{weight：{类0：tensor1,tensor2},{类1：tensor1,tensor3},bias：...}
        client_models = []
        client_sample_sizes = []
        # 收集所有客户端的共享参数
        for client in self.join_clients:
            client_params = client.model.head.state_dict() ##wx+b w是10*84，提取每个类对应的w
            client_models.append(client.model.head.state_dict())
            client_sample_sizes.append(client.sample_size)
            self.reporter.track_upload(client_params)
            for key in client.shared_keys:
                for cls in client.owned_classes:
                    aggregated[key][cls].append(client_params[key][cls])

        size_tensor = torch.tensor(client_sample_sizes, dtype=torch.float).to(self.device)
        client_weight = size_tensor / torch.sum(size_tensor)
        ##多个客户端的模型参数进行加权平均聚合
        for key in self.aggregatedHead.keys():# 全连接层的weight和bias
            param = [client_model[key] for client_model in client_models]
            param = torch.stack(param, dim=0)
            shape = [-1] + [1] * (len(param.shape) - 1)
            self.aggregatedHead[key] = torch.sum(param * client_weight.reshape(shape), dim=0)
        join_ids = [c.client_id for c in self.join_clients]  # 当前参与客户端的ID列表
        for key in self.shared_params.keys():
            for cls in aggregated[key]: ##cls是 0 按类聚合{weight：{类0：【tensor1,tensor2】},{类1：【tensor1,tensor3】},bias：...}
                # 只获取当前参与客户端中属于该类的client_id
                valid_clients = [
                    cid for cid in self.class_distribution[cls] ##收集类分布 {类0：客户端[1,2,3],类1：客户端[2,3]}
                    if cid in join_ids  # 关键过滤步骤
                ]


                ## 数量分数
                # weights_class = torch.tensor([
                #     self.clients[cid].statistic.get(str(cls),0) ##每个客户端的样本数
                #     for cid in valid_clients
                # ]).to(self.device)
                # weights_class = weights_class.float()
                # weights_class /= weights_class.sum()

                ## 质量分数

                # self.collect_class_accuracy_lhc()
                self.collect_class_accuracy()
                # print([self.class_accuracy[cid][cls] for cid in valid_clients])
                weights_acc = torch.tensor([
                    self.class_accuracy[cid][cls]
                    for cid in valid_clients
                ]).to(self.device)
                #
                weights_acc = weights_acc.float()
                smooth_factor = 0.5
                smooth_acc = weights_acc + smooth_factor
                weights_acc /= smooth_acc.sum()

                #质量分数，数量分数
                # weights = weights_sample
                # weights = weights_class
                weights = weights_acc
                # weights = weights_class + weights_acc

                if weights.sum() == 0:
                    raise ValueError(f"Class {cls} has no valid clients in this round")

                weights /= weights.sum() ## 当weight由多个部分组成生效


                # 加权聚合
                params = torch.stack(aggregated[key][cls])  ##按类聚合{weight：{类0：tensor1,tensor2},{类1：tensor1,tensor3},bias：客户端[2,3]}
                shape = [-1] + [1] * (len(params.shape) - 1)
                # clipped_params = torch.clip(params, min=-1.0, max=1.0)
                self.shared_params[key][cls] = torch.sum(
                    params * weights.view(*shape),
                    dim=0
                )

                # 直接平均
                # params = torch.stack(aggregated[key][cls])
                # # 对每个参数进行求平均
                # self.shared_params[key][cls] = torch.mean(params, dim=0)


    def aggregateHeadSample(self):
        client_models = []
        client_sample_sizes = []
        for client in self.join_clients:
            client_models.append(client.model.head.state_dict())
            client_sample_sizes.append(client.sample_size)
            # self.reporter.track_upload(client.model.state_dict())
        size_tensor = torch.tensor(client_sample_sizes, dtype=torch.float).to(self.device)
        # self.reporter.track_upload(size_tensor)
        client_weight = size_tensor / torch.sum(size_tensor)

        for key in self.aggregatedHead.keys():
            param = [client_model[key] for client_model in client_models]
            param = torch.stack(param, dim=0)
            shape = [-1] + [1] * (len(param.shape) - 1)
            self.aggregatedHead[key] = torch.sum(param * client_weight.reshape(shape), dim=0)


    # def aggregateHeadAvg(self):
    #     client_models = []
    #     client_sample_sizes = []
    #     for client in self.join_clients:
    #         client_models.append(client.model.head.state_dict())
    #         client_sample_sizes.append(client.sample_size)
    #         # self.reporter.track_upload(client.model.state_dict())
    #     size_tensor = torch.tensor(client_sample_sizes, dtype=torch.float).to(self.device)
    #     # self.reporter.track_upload(size_tensor)
    #     client_weight = size_tensor / torch.sum(size_tensor)
    #
    #     for key in self.aggregatedHead.keys():
    #         param = [client_model[key] for client_model in client_models]
    #         param = torch.stack(param, dim=0)
    #         shape = [-1] + [1] * (len(param.shape) - 1)
    #         self.aggregatedHead[key] = torch.sum(param * client_weight.reshape(shape), dim=0)


    def dispatch_parameters(self):
        """分发全局共享参数"""
        for client in self.join_clients:
            client_params = client.model.head.state_dict()
            # for key in self.shared_params.keys():
            #     for cls in client.owned_classes:
            #         client_params[key][cls] = self.shared_params[key][cls]
            for key in self.shared_params.keys():
                client_params[key] = self.lambdaWeight * self.shared_params[key]+(1-self.lambdaWeight)*self.aggregatedHead[key].to('cpu')
                # client_params[key] = client_params[key].to(self.device) + \
                #                      0.9 **(self.global_rounds//10) *\
                #                      (self.lambdaWeight * self.shared_params[key].to(self.device) +(1-self.lambdaWeight)*self.aggregatedHead[key].to(self.device) - client_params[key].to(self.device) )

            client.globalHead.load_state_dict(client_params)
            self.reporter.track_download(client_params)
            # client.model.projection.load_state_dict(self.aggregatedProjection)
            # client.globalHead.load_state_dict(self.aggregatedHead)


    def train(self):
        self.reporter = MemReporter()
        acc_record = {}

        for i in range(self.global_rounds):
            self.args.current_round = i
            self.join_clients = self.select_join_clients()
            print(f"============== global round: {i}, number of join clients: {len(self.join_clients)} ==============")

            for client in self.join_clients:
                client.train()

            # 参数聚合
            self.aggregate_shared_params()
            # self.aggregateHeadSample()
            # self.aggregateHeadAvg()
            # if (i%5==0):
            #     self.aggregateProjection()

            self.dispatch_parameters()
            # self.receive_and_aggregate()
            # self.dispatch()

            print("================== evaluate =================")
            acc = self.evaluate(i) #可能会报错？
            acc_record[i] = acc
            print(f"================= global round: {i}, accuracy: {acc} =================")

        uploadCom,downloadCom,usedMemory =self.getReport()
        return acc_record,uploadCom,downloadCom,usedMemory



# # server_fedssa.py
# import copy
#
# import torch
# import numpy as np
# from collections import defaultdict
# from src.server_base import ServerBase
# from mem_utils import MemReporter
# from src.FedSSA.client_FedSSA import ClientFedSSA
#
# class ServerFedSSA(ServerBase):
#     def __init__(self, args):
#         super().__init__(args)
#
#         # 初始化全局共享参数
#         # self.shared_params = copy.deepcopy(args.head.state_dict())
#         # self.aggregatedHead = self.args.head.state_dict()
#         self.initialize_clients(ClientFedSSA)
#         self.class_distribution = defaultdict(list)
#         self.sample_weights = []
#
#         # 设置超参数
#         self.decay_rounds = 2
#         self.miu_0 = 0.5
#
#         self.collect_class_distribution()
#         self.initializeGlobalHead()
#
#     def initializeGlobalHead(self):
#         Global_class_headers = defaultdict()
#         for key, paras in self.args.head.named_parameters():
#             Global_class_headers[key] = defaultdict()
#             for s in range(self.args.num_classes):
#                 Global_class_headers[key][s] = self.args.head[key][s]
#         self.Global_class_headers=Global_class_headers
#
#     def collect_class_distribution(self):
#         """收集各客户端的类别分布"""
#         for client in self.clients:
#             self.sample_weights.append(client.sample_size)
#             for cls in client.owned_classes:
#                 self.class_distribution[cls].append(client.client_id)
#
#     def aggregate_shared_params(self):
#         # """聚合共享参数"""
#         # aggregated = self.args.head.state_dict()
#         # # 收集所有客户端的共享参数
#         # for client in self.join_clients:
#         #     client_params = client.head.state_dict()
#         #     for key in client.shared_keys:
#         #         for cls in client.owned_classes:
#         #             aggregated[key][cls].append(client_params[key][cls])
#         #
#         # # 加权平均聚合
#         # for key in self.aggregatedHead.keys():
#         #     for cls in aggregated[key]:
#         #         params = torch.stack(aggregated[key][cls])
#         #         weights = torch.tensor([
#         #             self.sample_weights[cid]
#         #             for cid in self.class_distribution[cls]
#         #         ]).to(self.device)
#         #         weights /= weights.sum()
#         #
#         #         # 扩展维度进行加权求和
#         #         shape = [-1] + [1] * (len(params.shape) - 1)
#         #         self.aggregatedHead[key][cls] = torch.sum(
#         #             params * weights.view(*shape),
#         #             dim=0
#         #         )
#         """聚合共享参数"""
#         aggregated = defaultdict(lambda: defaultdict(list))
#
#         # 收集所有客户端的共享参数
#         for client in self.join_clients:
#             client_params = client.model.head.state_dict()
#             for key in self.Global_class_headers.keys():
#                 for cls in client.owned_classes:
#                     aggregated[key][cls].append(client_params[key][cls])
#
#         # 计算权重
#         # client_models = []
#         # client_sample_sizes = []
#         # for client in self.join_clients:
#         #     client_models.append(client.model.head.state_dict())
#         #     client_sample_sizes.append(client.sample_size)
#         #     self.reporter.track_upload(client.model.head.state_dict())
#         # size_tensor = torch.tensor(client_sample_sizes, dtype=torch.float).to(self.device)
#         # self.reporter.track_upload(size_tensor)
#         # client_weight = size_tensor / torch.sum(size_tensor)
#         #
#         # for key in self.Global_class_headers.keys():
#         #     param = [client_model[key] for client_model in client_models]
#         #     param = torch.stack(param, dim=0)
#         #     shape = [-1] + [1] * (len(param.shape) - 1)
#         #     self.aggregated[key] = torch.sum(param * client_weight.reshape(shape), dim=0)
#
#         # 加权平均聚合
#         for key in self.Global_class_headers.keys():
#             for cls in aggregated[key]:
#                 params = torch.stack(aggregated[key][cls])
#                 weights = torch.tensor([
#                     self.sample_weights[cid]
#                     for cid in self.class_distribution[cls]
#                 ]).to(self.device)
#                 # weights /= weights.sum()
#                 weights = weights.float()  # Convert weights to float
#                 weights /= weights.sum()  # Perform the division
#
#                 # 扩展维度进行加权求和
#                 shape = [-1] + [1] * (len(params.shape) - 1)
#                 self.Global_class_headers[key][cls] = torch.sum(
#                     params * weights.view(*shape),
#                     dim=0
#                 )
#
#     def dispatch_parameters(self):
#         """分发全局共享参数"""
#         for client in self.join_clients:
#             client_params = client.model.head.state_dict()
#             # for key in self.aggregatedHead.keys():
#             for key, paras in self.Global_class_headers.items():
#                 for cls in client.owned_classes:
#                     client_params[key][cls] = self.Global_class_headers[key][cls]
#             client.globalHead.load_state_dict(client_params)
#
#
#     def train(self):
#         self.reporter = MemReporter()
#         acc_record = {}
#
#         for i in range(self.global_rounds):
#             self.args.current_round = i
#             self.join_clients = self.select_join_clients()
#             print(f"============== global round: {i}, number of join clients: {len(self.join_clients)} ==============")
#
#             for client in self.join_clients:
#                 client.train()
#
#             # 参数聚合
#             self.aggregate_shared_params()
#
#             self.dispatch_parameters()
#
#             print("================== evaluate =================")
#             acc = self.evaluate()
#             acc_record[i] = acc
#             print(f"================= global round: {i}, accuracy: {acc} =================")
#
#         uploadCom, downloadCom = self.reporter.print_communication_stats()
#         return acc_record, uploadCom, downloadCom