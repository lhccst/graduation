import copy
from collections import defaultdict

import torch
from torch import nn
from tqdm import tqdm

from src.FedSC.loss import GlobalConLoss, SupConLoss, DPNLoss
from src.client_base import ClientBase
import torch.nn.functional as F


class ClientFedSC(ClientBase):
    def __init__(self, args, client_id, trainset, testset, statistic):
        super().__init__(args, client_id, trainset, testset, statistic)

        # self.model = copy.deepcopy(args.model)

        self.lamb_da = args.client["lambda"]
        self.belta = args.client["belta"]

        self.local_protos = {}  # dict
        self.upload_protos = {}
        self.global_protos = None  # list

        self.proto_loss = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        self.con_loss = SupConLoss()
        self.global_loss = GlobalConLoss()
        self.global_loss_emb = DPNLoss()
        self.global_loss_pro = DPNLoss()

    def aggregate(self, raw_feats):
        for label in raw_feats.keys():
            feats = torch.stack(raw_feats[label], dim=0)
            self.local_protos[label] = torch.mean(feats, dim=0)
            self.upload_protos[label] = self.local_protos[label] * self.statistic[str(label)]

    def collect_feats(self):
        train_loader = self.get_train_loader()
        self.model.eval()

        raw_feats = defaultdict(list)
        with torch.inference_mode():
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                feature = self.model.base(x)

                for label in torch.unique(y):
                    protos = feature[label == y].detach()  # protos: 形状为 [num_samples_of_class_y, feature_dim]的张量
                    raw_feats[label.item()].extend(list(torch.unbind(protos, dim=0)))  # 将特征矩阵拆分成单个特征向量的列表

        return raw_feats

    def alignModels(self, global_protos, proto_batch, y_batch):
        # 获取本地模型的类特定原型
        local_prototypes = [[] for _ in range(self.num_classes)]

        # 根据标签分散原型
        for proto, y in zip(proto_batch, y_batch):
            local_prototypes[y.item()].append(proto)

        local_protos_list = []
        global_protos_list = []
        #=========================================
        for label in range(self.num_classes):
            # 检查本地是否有该类别
            if local_prototypes[label]:  # 非空列表，客户端有这个类型
                # 计算本地平均原型
                stacked_protos = torch.stack(local_prototypes[label])
                local_mean = torch.mean(stacked_protos, dim=0)
                local_protos_list.append(local_mean)
                # 检查全局是否有对应的原型
                if (global_protos[label] is not None) :
                    global_proto = global_protos[label]
                    global_protos_list.append(global_proto)
                else:
                    # 全局没有对应原型，移除刚添加的本地原型
                    local_protos_list.pop()  # 移除最后添加的
        # 如果没有匹配的类别，返回0
        if len(local_protos_list) == 0 or len(global_protos_list) == 0:
            return torch.tensor(0.0, device=self.device)

        mean_prototypes_tensor = torch.stack([proto for proto in local_protos_list if proto is not None])
        global_prototypes_tensor = torch.stack([proto for proto in global_protos_list if proto is not None])
        # 使用KL散度损失进行对齐
        alignment_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
        # alignment_optimizer = torch.optim.SGD(global_head.parameters(), lr=0.01)


        protoLocalLogits = self.model.head(mean_prototypes_tensor)
        # proroGlobalLogits = global_head(mean_prototypes_tensor)
        proroGlobalLogits = self.model.head(global_prototypes_tensor)


        # 计算软最大概率分布之间的KL散度损失
        loss = alignment_loss_fn(
            torch.nn.functional.log_softmax(protoLocalLogits, dim=1),  # 全局
            torch.nn.functional.softmax(proroGlobalLogits, dim=1)  # 本地
        )

        # 清理GPU缓存
        torch.cuda.empty_cache()
        return loss

    def train(self):
        train_loader = self.get_train_loader()
        self.model.train()


        for epoch in range(self.local_epochs):
            local_iter = tqdm(train_loader, desc=f"client {self.client_id} epoch: {epoch}")
            for x, y in local_iter:
                x, y = x.to(self.device), y.to(self.device)
                feature = self.model.base(x)
                # if self.global_protos:  # 原型对齐
                #     loss_align = self.alignModels(self.global_protos,feature, y)
                # else:
                #     loss_align = torch.tensor(0.0, device=self.device)

                output = self.model.head(feature)
                loss = self.loss(output, y)
                # loss += 0.05 * loss_align
                normalized_feature = F.normalize(feature, dim=1)
                loss += self.lamb_da * self.con_loss(normalized_feature, labels=y)


                if self.global_protos is not None:
                    local_proto_list = []
                    proto_labels = []  # 原型标签列表
                    for class_id, proto in self.local_protos.items():
                        local_proto_list.append(proto)
                        proto_labels.append(class_id)  # 用类别ID作为原型标签
                    loss += self.lamb_da * self.global_loss(
                        torch.stack(local_proto_list),  # 转换为张量
                        labels=torch.tensor(proto_labels, device=self.device),  # ✅ 使用原型标签
                        global_protos=self.global_protos
                    )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                local_iter.set_description(f"client {self.client_id} epoch: {epoch} loss: {loss.item():.4f}")

        raw_feats = self.collect_feats()
        self.aggregate(raw_feats)

