import torch

from src.FedSC.client_FedSC import ClientFedSC
from src.server_base import ServerBase
from mem_utils import MemReporter
import math

class ServerFedSC(ServerBase):
    def __init__(self, args):
        super().__init__(args)

        self.initialize_clients(ClientFedSC)

        self.global_protos = [None] * self.num_classes
        self.preGlobal_protos = [None] * self.num_classes
        self.eta = StepSizeEta(eta0=1.0, decay_strategy='exponential')
        # self.eta = StepSizeEta(eta0=1.0, decay_strategy='cosine')
        # self.miu = args.client["miu"]
    def dispatch(self):
        if None not in self.global_protos:
            for client in self.join_clients:
                self.reporter.track_download(self.global_protos)
                # client.global_protos = self.global_protos
                alpha =  0.0002
                client.global_protos = [proto * alpha for proto in self.global_protos]

                # client.global_protos = [proto * self.miu for proto in self.global_protos]

    def receive_and_aggregate(self):
        for label in range(self.num_classes):
            proto_list = [client.upload_protos[label] for client in self.join_clients
                          if label in client.upload_protos]
            if len(proto_list) == 0:
                continue
            proto = torch.stack(proto_list, dim=0) #10*84
            self.reporter.track_upload(proto)
            # client_sample = [client.statistic[str(label)] for client in self.join_clients
            #                  if label in client.upload_protos]
            client_sample = [1 for client in self.join_clients
                             if label in client.upload_protos]
            weight = torch.tensor(client_sample).to(self.device)
            # self.reporter.track_upload(weight)
            weight = weight / torch.sum(weight)
            proto = torch.sum(proto * weight.reshape(-1, 1), dim=0) #84
            self.global_protos[label] = proto
            # ====余弦衰减融合====

            current_eta = self.eta.get(self.args.current_round)
            # 检查是否已存在前一轮原型
            if self.preGlobal_protos[label] is None:
                # 第一次聚合，直接使用加权原型
                self.global_protos[label] = self.global_protos[label]
            else:
                # 应用余弦衰减融合
                self.global_protos[label] = self.preGlobal_protos[label] + current_eta * self.global_protos[label]

            # 保存当前全局原型
            self.preGlobal_protos[label] = self.global_protos[label].clone()
            # 先筛选出有该类别的客户端===
            valid_clients = [c for c in self.join_clients if label in c.upload_protos]
            num_valid = len(valid_clients)
            self.global_protos[label] = proto / num_valid
            # 记录上轮的全局原型
            self.preGlobal_protos[label] = self.global_protos[label]



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


    def train(self):
        acc_record = {}
        for i in range(self.global_rounds):
            self.args.current_round = i
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

        uploadCom,downloadCom,usedMemory =self.getReport()
        return acc_record,uploadCom,downloadCom,usedMemory


class StepSizeEta:
    """实现图片公式(10)中的 η_g^{(t)} 计算器"""

    def __init__(self, eta0=1.0, decay_strategy='exponential',t = 0):
        self.eta0 = eta0
        self.strategy = decay_strategy
        self.min_eta = 0.5
        self.gamma = 0.99  # 指数衰减系数
        self.T = 50  # 余弦周期
        self.t = t

    def get(self, round_num=None):
        """获取图片公式(10)中的 η_g^{(t)}（预先衰减）"""
        t = self.t if round_num is None else round_num

        if self.strategy == 'exponential':
            # 指数衰减公式：η_g(t) = max(η_0 * γ^t, η_min)
            eta = max(self.eta0 * (self.gamma ** t), self.min_eta)

        elif self.strategy == 'cosine':
            # 余弦衰减公式：η_g(t) = max(η_min, η_0 * 0.5 * (1 + cos(π * t / T)))
            eta = max(self.min_eta, 0.5 * self.eta0 * (1 + math.cos(t * math.pi / self.T)))

        else:
            eta = self.eta0  # 不衰减

        return eta