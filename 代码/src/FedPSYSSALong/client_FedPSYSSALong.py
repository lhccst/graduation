
import copy
import os
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from src.client_base import ClientBase
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from matplotlib import rcParams
import torch.nn.functional as F
from .loss import SupConLoss

class ClientFedPSYSSALong(ClientBase):
    def __init__(self, args, client_id, trainset, testset, statistic):
        super().__init__(args, client_id, trainset, testset, statistic)

        self.private_keys, self.shared_keys = self.get_parameter_keys()

        # 初始化优化器
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9
            # weight_decay=args.inner_wd
        )

        # 客户端特有属性
        self.owned_classes = self.get_owned_classes()
        # self.alpha = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.alpha.data.fill_(0.5)
        self.globalHead =copy.deepcopy(args.head).to(self.device)
        self.globalEpochRound=0
        self.avg_scores = defaultdict(list)
        # self.decay_rounds=2
        # self.miu_0 = 0.5 # 0.1,0.3,0.5,0.7,0.9,1.0

        # self.local_protos = {}  # dict
        # self.global_protos = None  # list

        # self.proto_loss = nn.MSELoss()
        # self.optimizer_per = torch.optim.SGD(self.model.head.parameters(), lr=self.learning_rate, momentum=0.9)
        self.accuracyClass = [0] * self.num_classes
        ## ==================
        self.cdist = self.dict_to_vector() ## 是一个【1，类别数】的tensor
        self.ala = ALA(self.local_epochs, self.learning_rate, self.device)
        self.ala2 = ALA2(self.local_epochs, self.learning_rate, self.device)
        ## ==================

        # self.class_weights = self._calculate_weights_from_statistic()
        # self.loss = nn.CrossEntropyLoss(weight=self.class_weights)

        # self.loss_fn = nn.CrossEntropyLoss(weight=self.cdist)
        self.con_loss = SupConLoss()
        self.lamb_da = args.client["lambda"]

    def _calculate_weights_from_statistic(self):
        """从 self.statistic 计算类别权重"""
        # 将字典转换为按类别顺序排列的张量
        class_counts = torch.zeros(self.num_classes, device=self.device)
        for class_id, count in self.statistic.items():
            class_counts[int(class_id)] = count

        # 计算权重（样本数越少，权重越大）
        weights = 1.0 / (class_counts + 1e-8)  # 加1e-8防止除零
        # 归一化权重（保持均值不变）
        weights = weights / weights.mean()
        return weights.to(self.device)

    def get_parameter_keys(self):
        """分离共享参数和私有参数"""
        # all_keys = list(self.model.state_dict().keys())
        # private_keys = all_keys[:-2]  # 假设最后两层是分类头
        # shared_keys = all_keys[-2:]
        # all_keys = list(self.model.state_dict().keys())
        private_keys = list(self.model.base.state_dict().keys()) # 假设最后两层是分类头
        shared_keys = list(self.model.head.state_dict().keys())
        return private_keys, shared_keys

    def get_owned_classes(self):
        """获取客户端拥有的类别"""
        owned = set()
        for _, y in self.get_train_loader():
            owned.update(y.unique().tolist())
        return list(owned)

    # def aggregate(self, raw_feats):
    #     for label in raw_feats.keys():
    #         feats = torch.stack(raw_feats[label], dim=0)
    #         self.local_protos[label] = torch.mean(feats, dim=0)

    # def collect_feats(self):
    #     train_loader = self.get_train_loader()
    #     self.model.eval()
    #
    #     raw_feats = defaultdict(list)
    #     with torch.inference_mode():
    #         for x, y in train_loader:
    #             x, y = x.to(self.device), y.to(self.device)
    #             feature = self.model.base(x)
    #
    #             for label in torch.unique(y):
    #                 protos = feature[label == y].detach()
    #                 raw_feats[label.item()].extend(list(torch.unbind(protos, dim=0)))
    #
    #     return raw_feats

    def dict_to_vector(self):
        # 初始化全零向量
        vector = [0] * self.num_classes

        # 遍历字典，填充样本数
        for key, count in self.statistic.items():
            idx = int(key)  # 将字符串键转为整数索引
            if idx < self.num_classes:
                vector[idx] = count
            else:
                # 可选：处理超出范围的类别（如忽略或警告）
                pass
        dist = torch.tensor(vector, device=self.device) ##每个客户端拥有的类的数量[1,0,1137...]类0有1个 类2有0个 类3有1137个

        nonzero_mask = dist > 0
        nonzero_dist = dist[nonzero_mask]
        # rs = max(nonzero_dist)/min(nonzero_dist)  #计算类不平衡程度的指数
        dist = dist/dist.sum()
        nonzero_dist = nonzero_dist/nonzero_dist.sum()


        dist = dist/dist.sum()  ##归一化之后的 ndist

        # entropy = -torch.sum(nonzero_dist * torch.log(nonzero_dist + 1e-9))
        # max_entropy = torch.log(torch.tensor(nonzero_dist.size(0), device=self.device))
        # # print("max_entropy - entropy",max_entropy - entropy)
        # rs_alpha = torch.sigmoid(entropy/max_entropy)+0.25
        # # rs_alpha = torch.exp(-0.5 * entropy)
        # # print(entropy,rs_alpha)
        # print("entropy：",entropy,max_entropy,rs_alpha,rs_alpha+0.25)


        rs_alpha = 0.7
        cdist = dist / dist.max()
        cdist = cdist.to(self.device)
        # cdist = cdist * (max_entropy - entropy) + max_entropy
        cdist = cdist * (1.0 - rs_alpha) + rs_alpha
        # self.cdist = cdist
        cdist = cdist.reshape((1, -1))
        return cdist

    def train(self):
        """本地训练过程"""
        self.globalEpochRound+=1
        self.model.train()
        train_loader = self.get_train_loader()

        # self.fixBase()

        # 动态调整alpha
        # self.adjust_alpha()
        # self.alignModels(self.globalHead)
        # self.fuse_global_paramsWithClassImbalanced()
        # if self.globalHead is not None:
        #     aggregated_head = self.ala2.adaptive_aggregate(self.model, self.globalHead, train_loader)
        #     self.model.head.load_state_dict(aggregated_head)


        # 初始化类别的正确预测数量和总数量
        correct_per_class = [0] * self.num_classes
        total_per_class = [0] * self.num_classes
        # print("before__________>",torch.cuda.memory_allocated() / 1024 ** 2, "MB used")
        for epoch in range(self.local_epochs):
            local_iter = tqdm(train_loader, desc=f"client {self.client_id} epoch: {epoch}")
            for x, y in local_iter:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                #==========
                hs = self.model.base(x) ## 64*84
                ws = self.model.head.weight ##10*84
                hs, ws = hs.to(self.device), ws.to(self.device)


                logits = self.cdist * hs.mm(ws.transpose(0, 1))
                lossA= self.loss(logits, y)
                #==========
                # # 计算模型输出和损失
                feature = self.model.base(x)
                output = self.model.head(feature)
                # output = self.model(x)
                # loss = self.loss_fn(output, y)

                normalized_feature = F.normalize(feature, dim=1)
                #不加cdis
                # loss = self.loss(output, y)
                # loss += self.lamb_da * self.con_loss(normalized_feature, labels=y)
                ##加cdist
                loss = lossA
                loss += self.lamb_da * self.con_loss(normalized_feature, labels=y)


                ##=======
                # loss =  lossA
                # loss =  loss_origin
                ##=======
                loss.backward()
                self.optimizer.step()

                # 更新类别正确预测数量
                _, predicted = torch.max(output, 1)
                for label, pred in zip(y, predicted):
                    total_per_class[label.item()] += 1
                    if label.item() == pred.item():
                        correct_per_class[label.item()] += 1

                # 显示训练进度
                local_iter.set_description(f"client {self.client_id} epoch: {epoch} loss: {loss.item():.4f}")

                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 50)

            # 每个 epoch 结束后打印每个类别的准确率
            for class_id in range(self.num_classes):
                accuracy = correct_per_class[class_id] / total_per_class[class_id] if total_per_class[class_id] > 0 else 0
                self.accuracyClass[class_id] =accuracy
            # print(self.accuracyClass)

        # raw_feats = self.collect_feats()
        # self.aggregate(raw_feats)
        ###########
        # raw_feats, raw_scores = self.collect_feats()

        torch.cuda.empty_cache()
        # print("after__________>",torch.cuda.memory_allocated() / 1024 ** 2, "MB used")

    def collect_feats(self):
        train_loader = self.get_train_loader()
        self.model.eval()

        raw_feats = defaultdict(list)  ##
        raw_scores = defaultdict(list)

        with torch.inference_mode():
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                feature = self.model.base(x)
                score = F.softmax(self.model.head(feature), dim=1)

                for label in torch.unique(y):
                    samples = feature[label == y].detach() ## 一个客户端类别n的特征

                    raw_feats[label.item()].extend(list(torch.unbind(samples, dim=0))) ## 一个客户端类别1-10的特征列表，没有该类则为空
                    raw_scores[label.item()].extend(score[label == y][:, label].detach().cpu().detach()) ## 一个客户端类别1-10的分数列表
        # 计算客户端中每个类别的平均分数
        for label, scores in raw_scores.items():
            if scores:  # 确保列表不为空
                self.avg_scores[label] = sum(score.item() for score in scores) / len(scores)
            else:
                self.avg_scores[label] = 0  # 如果没有分数，则平均分数为0

        return raw_feats,raw_scores

    def adjust_alpha(self):
        """动态调整参数融合系数"""
        if self.args.current_round <= self.decay_rounds:
            decay = torch.cos(torch.tensor(self.args.current_round * torch.pi /
                                           (self.decay_rounds * 2)))
            self.alpha.data.fill_(self.miu_0 * decay.item())
        else:
            self.alpha.data.fill_(0.0)

    def fuse_global_params(self):
        """融合全局共享参数"""
        global_params = self.globalHead.state_dict()
        # local_params = self.model.head.state_dict()
        # global_params = self.server.shared_params
        local_params = self.model.head.state_dict()
        self.alpha = self.alpha.to(self.device)
        with torch.no_grad():
            # for key, param in self.globalHead.named_parameters():
            for key in self.shared_keys:
                for cls in self.owned_classes:
                    # 参数融合公式
                    local_params[key][cls] = (
                            local_params[key][cls] * self.alpha +
                            global_params[key][cls] * (1 - self.alpha)
                    )
        self.model.head.load_state_dict(local_params)

    def fuse_global_paramsWithClassImbalanced(self):
        """融合全局共享参数"""
        global_params = self.globalHead.state_dict()
        # local_params = self.model.head.state_dict()
        # global_params = self.server.shared_params
        local_params = self.model.head.state_dict()
        # self.alpha = self.alpha.to(self.device)
        with torch.no_grad():
            # for key, param in self.globalHead.named_parameters():
            for key in self.shared_keys:
                for cls in self.owned_classes:
                    alpha =self.statistic[str(cls)]/self.sample_size

                    # 参数融合公式
                    local_params[key][cls] = (
                            local_params[key][cls] * alpha +
                            global_params[key][cls] * (1 - alpha)
                    )
        self.model.head.load_state_dict(local_params)


    # def fixBase(self):
    #     for param in self.model.base.parameters():
    #         param.requires_grad = False
    #     for param in self.model.head.parameters():
    #         param.requires_grad = True
    #     train_loader = self.get_train_loader()
    #     for epoch in range(2):
    #         p_local_iter = tqdm(train_loader, desc=f"client {self.client_id} epoch: {epoch}")
    #         for x, y in p_local_iter:
    #             # for i, (x, y) in enumerate(train_loader):
    #             if type(x) == type([]):
    #                 x[0] = x[0].to(self.device)
    #             else:
    #                 x = x.to(self.device)
    #             y = y.to(self.device)
    #             output = self.model(x)
    #             loss = self.loss(output, y)
    #             self.optimizer_per.zero_grad()
    #             loss.backward()
    #             self.optimizer_per.step()
    #             p_local_iter.set_description(f"client {self.client_id} P_local_epoch: {epoch} loss: {loss.item():.4f}")
    #
    #     max_local_epochs = self.local_epochs
    #     for param in self.model.base.parameters():
    #         param.requires_grad = True

    # plan A
    def alignModels(self,global_head):
        # self.plot_confusion_matrix(global_head,"Before Align Confusion Matrix",self.globalEpochRound)
        # Get class-specific prototypes from the local model
        local_prototypes = [[] for _ in range(self.num_classes)]
        train_loader = self.get_train_loader()

        # print(f'client{id}')
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            with torch.no_grad():
                proto_batch = self.model.base(x_batch)
                # proto_batch = self.model.projection(proto_batch)

            # Scatter the prototypes based on their labels
            for proto, y in zip(proto_batch, y_batch):
                local_prototypes[y.item()].append(proto)

        mean_prototypes = []

        # print(f'client{self.id}')
        for class_prototypes in local_prototypes:

            if not class_prototypes == []:
                # Stack the tensors for the current class
                stacked_protos = torch.stack(class_prototypes)

                # Compute the mean tensor for the current class
                mean_proto = torch.mean(stacked_protos, dim=0)
                mean_prototypes.append(mean_proto)
            else:
                mean_prototypes.append(None)

        mean_prototypes_tensor = torch.stack([proto for proto in mean_prototypes if proto is not None])
        # mean_prototypes_tensor = torch.stack(mean_prototypes)
        # mean prototypes经过本地head，得到的
        alignment_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
        # alignment_loss_fn = torch.nn.MSELoss()

        alignment_optimizer = torch.optim.SGD(global_head.parameters(),
                                              lr=0.01)  # Adjust learning rate and optimizer as needed
        # alignment_loss_fn = torch.nn.MSELoss()
        # protoLocalLogits = self.model.head(mean_prototypes_tensor)
        # proroGlobalLogits = global_head(mean_prototypes_tensor)
        #
        # loss = alignment_loss_fn(torch.nn.functional.softmax(protoLocalLogits, dim=1),
        #                          torch.nn.functional.softmax(proroGlobalLogits, dim=1))
        #
        # alignment_optimizer.zero_grad()
        # loss.backward()
        # alignment_optimizer.step()
        for _ in range(1):  # Iterate for 1 epochs; adjust as needed
            protoLocalLogits = self.model.head(mean_prototypes_tensor)
            proroGlobalLogits = global_head(mean_prototypes_tensor)

            loss = alignment_loss_fn(torch.nn.functional.softmax(protoLocalLogits, dim=1),
                                     torch.nn.functional.softmax(proroGlobalLogits, dim=1))


            alignment_optimizer.zero_grad()
            loss.backward()
            alignment_optimizer.step()

        # self.plot_confusion_matrix(global_head,"After Align Confusion Matrix",self.globalEpochRound)
        # Substitute the parameters of the base, enabling personalization
        for new_param, old_param in zip(global_head.parameters(), self.model.head.parameters()):
            old_param.data = new_param.data.clone()
        torch.cuda.empty_cache()
        # print("after__________>",torch.cuda.memory_allocated() / 1024 ** 2, "MB used")

        # self.globalHead = global_head


    # 画混淆矩阵
    # autodl-tmp/FedPSY312/results/kronoDroid/dirichlet
    def plot_confusion_matrix(self,globalHead,title,epoch):
        # rcParams['font.sans-serif'] = ['SimHei']
        # plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
        output_dir = f"/root/autodl-tmp/FedPSY312/results/kronoDroid/hunxiaoKronoDroid/{epoch}"
        output_path = os.path.join(output_dir, f"{self.client_id}-{title}.png")

        # 检查并创建文件夹
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        test_loader = self.get_test_loader()

        true_labels = []
        predicted_labels = []

        globalHead.eval()  # 设置为评估模式
        self.model.eval()  # 如果需要用到特征提取器

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                # 获取特征
                features = self.model.base(x_batch)
                logits = globalHead(features)
                preds = logits.argmax(dim=1)  # 获取预测类别

                true_labels.extend(y_batch.cpu().numpy())
                predicted_labels.extend(preds.cpu().numpy())

        # 绘制混淆矩阵
        cm = confusion_matrix(true_labels, predicted_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(title)
        plt.xlabel("Predict Label")
        plt.ylabel("True Label")
        plt.savefig(output_path)  # 保存图片
        # plt.show()


class ALA:
    def __init__(self, local_epochs, learning_rate, device):
        self.local_epochs = local_epochs
        self.loss = torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.device = device

        self.weights = None  # learnable weight

    def adaptive_aggregate(self, local_model, server_dict, train_loader):
        local_dict = copy.deepcopy(local_model.state_dict())  # 暂存分类模型的参数

        if self.weights is None:
            self.weights = {
                key: torch.ones_like(param, requires_grad=True, device=self.device)
                for key, param in server_dict.items()
            }

        weights_optimizer = torch.optim.SGD(
            [
                {"params": self.weights.values()},
                # {"params": local_model.parameters()}
            ],
            self.learning_rate, momentum=0.9
        )

        model_optimizer = torch.optim.SGD(
            [
                # {"params": self.weights.values()},
                {"params": local_model.parameters()}
            ],
            self.learning_rate, momentum=0.9
        )

        local_model.train()
        # for param in local_model.parameters():
        #     param.requires_grad = False
        for epoch in range(self.local_epochs):
            local_iter = tqdm(train_loader, desc=f"local epoch: {epoch}")
            for x, y in local_iter:
                """ 得到新的参数 """
                aggregated = {}
                for key in server_dict.keys():
                    local_param = local_dict[key]
                    server_param = server_dict[key]
                    weight = self.weights[key]
                    aggregated[key] = local_param + (server_param - local_param) * weight
                local_model.head.load_state_dict(aggregated)

                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)

                model_optimizer.zero_grad()
                loss.backward()
                model_optimizer.step()

                weights_optimizer.zero_grad()
                weights_grads = torch.autograd.grad(
                    outputs=aggregated.values(),
                    inputs=self.weights.values(),
                    grad_outputs=local_model.state_dict().values()
                )

                for weight, grad in zip(self.weights.values(), weights_grads):
                    weight.grad = grad

                weights_optimizer.step()

                with torch.no_grad():
                    for value in self.weights.values():
                        value.clamp_(0, 1)

        with torch.inference_mode():
            aggregated = {}
            for key in server_dict.keys():
                local_param = local_dict[key]
                server_param = server_dict[key]
                weight = self.weights[key]
                aggregated[key] = local_param + (server_param - local_param) * weight
            return aggregated


class ALA2:
    def __init__(self, local_epochs, learning_rate, device):
        self.local_epochs = local_epochs
        self.loss = torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.device = device
        self.weights = None

    def adaptive_aggregate(self, local_model, server_head, train_loader):
        # 1. 保存本地模型参数（只保存头部）
        local_head_dict = copy.deepcopy(local_model.head.state_dict())
        local_base_dict = copy.deepcopy(local_model.base.state_dict())

        server_head_dict = copy.deepcopy(server_head.state_dict())
        # 如果还没有权重，只为头部初始化权重
        if self.weights is None:
            self.weights = {
                # key: torch.ones_like(param, requires_grad=True, device=self.device)
                key: torch.full_like(param, fill_value=0.9, requires_grad=True, device=self.device)
                for key, param in server_head_dict.items()  # 只对头部
            }

        # 3. 关键：冻结模型参数
        # for param in local_model.parameters():
        #     param.requires_grad = False

        # 4. 权重优化器（只优化头部权重）
        weights_optimizer = torch.optim.SGD(
            [{"params": self.weights.values()}],
            self.learning_rate, momentum=0.9
        )

        # 5. 训练头部权重
        local_model.train()
        for epoch in range(1):
            local_iter = tqdm(train_loader, desc=f"local epoch: {epoch}")
            for x, y in local_iter:
                # 5.1 构建完整的模型参数
                aggregated_state_dict = {}

                # 5.1.1 骨干网络保持不变（使用本地骨干）
                aggregated_state_dict.update(local_base_dict)

                # 5.1.2 头部进行自适应聚合
                for key in server_head_dict.keys():
                    local_head_param = local_head_dict[key]
                    server_head_param = server_head_dict[key]
                    weight = self.weights[key]

                    # ˆθ_head = θ_local_head + (θ_server_head - θ_local_head) * W
                    aggregated_head_param = local_head_param + (server_head_param - local_head_param) * weight
                    aggregated_state_dict[key] = aggregated_head_param

                # 5.2 加载完整的模型（骨干+聚合后的头部）
                local_model.load_state_dict(aggregated_state_dict, strict=False)

                # 5.3 前向计算损失
                x, y = x.to(self.device), y.to(self.device)
                output = local_model(x)
                loss = self.loss(output, y)

                # 5.4 手动计算权重梯度
                weights_optimizer.zero_grad()

                # 计算损失对模型参数的梯度
                loss.backward(retain_graph=True)

                # 5.5 计算权重梯度：∇_W L = Δθ ⊙ ∇_{ˆθ}L
                for key in server_head_dict.keys():
                    if key in local_model.state_dict():
                        # 获取对应的模型梯度
                        param = dict(local_model.named_parameters())[key]
                        if param.grad is not None:
                            delta = server_head_dict[key] - local_head_dict[key]
                            model_grad = param.grad

                            # 权重梯度 = 差异 × 模型梯度
                            weight_grad = delta * model_grad

                            # 设置梯度
                            if self.weights[key].grad is None:
                                self.weights[key].grad = weight_grad.detach().clone()
                            else:
                                self.weights[key].grad += weight_grad.detach().clone()

                # 5.6 清空模型梯度
                local_model.zero_grad()

                # 5.7 更新权重
                weights_optimizer.step()

                # 5.8 权重裁剪到 [0, 1]
                with torch.no_grad():
                    for value in self.weights.values():
                        value.clamp_(0, 1)

        # 6. 返回最终聚合的头部参数
        with torch.inference_mode():
            aggregated_head_dict = {}
            for key in server_head_dict.keys():
                local_head_param = local_head_dict[key]
                server_head_param = server_head_dict[key]
                weight = self.weights[key]
                aggregated_head_dict[key] = local_head_param + (server_head_param - local_head_param) * weight

            return aggregated_head_dict