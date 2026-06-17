import copy
import math

import torch
from tqdm import tqdm

from src.client_base import ClientBase
from collections import defaultdict
import torch.nn.functional as F

class ClientFedASAPA(ClientBase):
    def __init__(self, args, client_id, trainset, testset, statistic):
        super().__init__(args, client_id, trainset, testset, statistic)

        self.model = copy.deepcopy(args.model)
        self.global_model = copy.deepcopy(args.model.base)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)

        self.local_protos = {}
    def train(self):
        train_loader = self.get_train_loader()
        self.model.train()
        self.alignModels(self.global_model)
        for epoch in range(self.local_epochs):
            local_iter = tqdm(train_loader, desc=f"client {self.client_id} epoch: {epoch}")
            for x, y in local_iter:
                x, y = x.to(self.device), y.to(self.device)
                hs= self.model.base(x)
                ws = self.model.head.weight
                hs, ws = hs.to(self.device), ws.to(self.device)

                cdist = dist / dist.max()
                cdist = cdist.to(self.device)
                cdist = cdist * (1.0 - args.rs_alpha) + args.rs_alpha
                cdist = cdist.reshape((1, -1))
                logits = cdist * hs.mm(ws.transpose(0, 1))
                loss1 = self.criterion(logits, labels)






                output = self.model(x)
                loss = self.loss(output, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                local_iter.set_description(f"client {self.client_id} epoch: {epoch} loss: {loss.item():.4f}")



    # def alignModels(self,global_model):
    #     # Get class-specific prototypes from the local model
    #     local_prototypes = [[] for _ in range(self.num_classes)]
    #     train_loader = self.get_train_loader()
    #
    #     # print(f'client{id}')
    #     for x_batch, y_batch in train_loader:
    #         x_batch = x_batch.to(self.device)
    #         y_batch = y_batch.to(self.device)
    #
    #         with torch.no_grad():
    #             proto_batch = self.model.base(x_batch)
    #
    #         # Scatter the prototypes based on their labels
    #         for proto, y in zip(proto_batch, y_batch):
    #             local_prototypes[y.item()].append(proto)
    #
    #     mean_prototypes = []
    #
    #     # print(f'client{self.id}')
    #     for class_prototypes in local_prototypes:
    #
    #         if not class_prototypes == []:
    #             # Stack the tensors for the current class
    #             stacked_protos = torch.stack(class_prototypes)
    #
    #             # Compute the mean tensor for the current class
    #             mean_proto = torch.mean(stacked_protos, dim=0)
    #             mean_prototypes.append(mean_proto)
    #         else:
    #             mean_prototypes.append(None)
    #
    #     alignment_optimizer = torch.optim.SGD(global_model.parameters(),
    #                                           lr=0.01)  # Adjust learning rate and optimizer as needed
    #     alignment_loss_fn = torch.nn.MSELoss()
    #
    #     for _ in range(1):  # Iterate for 1 epochs; adjust as needed
    #         for x_batch, y_batch in train_loader:
    #             x_batch = x_batch.to(self.device)
    #             y_batch = y_batch.to(self.device)
    #             global_proto_batch = global_model(x_batch)
    #             loss = 0
    #             for label in y_batch.unique():
    #                 if mean_prototypes[label.item()] is not None:
    #                     loss += alignment_loss_fn(global_proto_batch[y_batch == label], mean_prototypes[label.item()])
    #             # print("align loss",loss)
    #             alignment_optimizer.zero_grad()
    #             loss.backward()
    #             alignment_optimizer.step()
    #
    #     # Substitute the parameters of the base, enabling personalization
    #     for new_param, old_param in zip(global_model.parameters(), self.model.base.parameters()):
    #         old_param.data = new_param.data.clone()

    # # 类别权重加权
    def alignModels(self,global_model):
        # Get class-specific prototypes from the local model
        local_prototypes = [[] for _ in range(self.num_classes)]
        train_loader = self.get_train_loader()

        # print(f'client{id}')
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            with torch.no_grad():
                proto_batch = self.model.base(x_batch)

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

        alignment_optimizer = torch.optim.SGD(global_model.parameters(),
                                              lr=0.01)  # Adjust learning rate and optimizer as needed
        alignment_loss_fn = torch.nn.MSELoss()

        # 初始化每个类别的准确率统计
        num_classes = len(mean_prototypes)  # 假设 mean_prototypes 是一个列表，长度为类别数量
        class_accuracies = [1.0] * num_classes  # 初始化每个类别的准确率为 1.0
        class_counts = [0] * num_classes  # 记录每个类别的样本数

        for _ in range(1):  # Iterate for 1 epochs; adjust as needed
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                global_proto_batch = global_model(x_batch)

                # 计算每个类别的预测结果
                outputs = self.model.head(global_proto_batch)  # 假设 global_proto_batch 是模型的输出 logits
                preds = torch.argmax(outputs, dim=1)
                correct = (preds == y_batch).float()  # 正确预测的样本

                # 更新每个类别的准确率
                for label in y_batch.unique():
                    label = label.item()
                    mask = (y_batch == label)
                    class_counts[label] += mask.sum().item()
                    class_accuracies[label] = correct[mask].sum().item() / mask.sum().item()  # 更新准确率

                # print("class_accuracies",class_accuracies)
                # 计算加权损失
                loss = 0
                for label in y_batch.unique():
                    label = label.item()
                    if mean_prototypes[label] is not None:
                        # 使用准确率作为权重（准确率越低，权重越大）
                        # weight =math.exp(1.0-class_accuracies[label])
                        weight =1.0-class_accuracies[label]
                        # print("weight:",weight)# 准确率越低，权重越大
                        loss += weight * alignment_loss_fn(global_proto_batch[y_batch == label], mean_prototypes[label])

                # 反向传播和优化
                alignment_optimizer.zero_grad()
                loss.backward()
                alignment_optimizer.step()

        # Substitute the parameters of the base, enabling personalization
        for new_param, old_param in zip(global_model.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()

    # 平滑
    # def alignModels(self, global_model):
    #     # Get class-specific prototypes from the local model
    #     local_prototypes = [[] for _ in range(self.num_classes)]
    #     train_loader = self.get_train_loader()
    #
    #     # print(f'client{id}')
    #     for x_batch, y_batch in train_loader:
    #         x_batch = x_batch.to(self.device)
    #         y_batch = y_batch.to(self.device)
    #
    #         with torch.no_grad():
    #             proto_batch = self.model.base(x_batch)
    #
    #         # Scatter the prototypes based on their labels
    #         for proto, y in zip(proto_batch, y_batch):
    #             local_prototypes[y.item()].append(proto)
    #
    #     mean_prototypes = []
    #
    #     # print(f'client{self.id}')
    #     for class_prototypes in local_prototypes:
    #
    #         if not class_prototypes == []:
    #             # Stack the tensors for the current class
    #             stacked_protos = torch.stack(class_prototypes)
    #
    #             # Compute the mean tensor for the current class
    #             mean_proto = torch.mean(stacked_protos, dim=0)
    #             mean_prototypes.append(mean_proto)
    #         else:
    #             mean_prototypes.append(None)
    #
    #     alignment_optimizer = torch.optim.SGD(global_model.parameters(),
    #                                           lr=0.01)  # Adjust learning rate and optimizer as needed
    #     alignment_loss_fn = torch.nn.MSELoss()
    #
    #     # 初始化每个类别的准确率统计
    #     num_classes = len(mean_prototypes)  # 假设 mean_prototypes 是一个列表，长度为类别数量
    #     class_accuracies = [1.0] * num_classes  # 初始化每个类别的准确率为 1.0
    #     smoothed_accuracies = [1.0] * num_classes  # 初始化平滑后的准确率
    #     alpha = 0.1  # 平滑系数
    #
    #     for _ in range(1):  # Iterate for 1 epochs; adjust as needed
    #         for x_batch, y_batch in train_loader:
    #             x_batch = x_batch.to(self.device)
    #             y_batch = y_batch.to(self.device)
    #             global_proto_batch = global_model(x_batch)
    #
    #             # 计算每个类别的预测结果
    #             preds = torch.argmax(global_proto_batch, dim=1)  # 假设 global_proto_batch 是模型的输出 logits
    #             correct = (preds == y_batch).float()  # 正确预测的样本
    #
    #             # 更新每个类别的准确率
    #             for label in y_batch.unique():
    #                 label = label.item()
    #                 mask = (y_batch == label)
    #                 if mask.sum().item() > 0:  # 确保有样本
    #                     current_accuracy = correct[mask].sum().item() / mask.sum().item()
    #                     # 使用指数滑动平均平滑准确率
    #                     smoothed_accuracies[label] = alpha * current_accuracy + (1 - alpha) * smoothed_accuracies[label]
    #
    #             # 计算加权损失
    #             loss = 0
    #             for label in y_batch.unique():
    #                 label = label.item()
    #                 if mean_prototypes[label] is not None:
    #                     # 使用平滑后的准确率作为权重（准确率越低，权重越大）
    #                     weight = 1.0 - smoothed_accuracies[label]  # 准确率越低，权重越大
    #                     loss += weight * alignment_loss_fn(global_proto_batch[y_batch == label], mean_prototypes[label])
    #
    #             # 反向传播和优化
    #             alignment_optimizer.zero_grad()
    #             loss.backward()
    #             alignment_optimizer.step()
    #     # Substitute the parameters of the base, enabling personalization
    #     for new_param, old_param in zip(global_model.parameters(), self.model.base.parameters()):
    #         old_param.data = new_param.data.clone()

    # def alignModels(self,global_model):
    #     # Get class-specific prototypes from the local model
    #     local_prototypes = [[] for _ in range(self.num_classes)]
    #     global_prototypes = [[] for _ in range(self.num_classes)]
    #     train_loader = self.get_train_loader()
    #
    #     # print(f'client{id}')
    #     for x_batch, y_batch in train_loader:
    #         x_batch = x_batch.to(self.device)
    #         y_batch = y_batch.to(self.device)
    #
    #         with torch.no_grad():
    #             local_proto_batch = self.model.base(x_batch)
    #             global_proto_batch = global_model(x_batch)
    #
    #         # Scatter the prototypes based on their labels
    #         for proto, y in zip(local_proto_batch, y_batch):
    #             local_prototypes[y.item()].append(proto)
    #
    #         for proto, y in zip(global_proto_batch, y_batch):
    #             global_prototypes[y.item()].append(proto)
    #
    #     local_mean_prototypes = []
    #     global_mean_prototypes = []
    #
    #     # print(f'client{self.id}')
    #     for class_prototypes in local_prototypes:
    #
    #         if not class_prototypes == []:
    #             # Stack the tensors for the current class
    #             stacked_protos = torch.stack(class_prototypes)
    #
    #             # Compute the mean tensor for the current class
    #             mean_proto = torch.mean(stacked_protos, dim=0)
    #             local_mean_prototypes.append(mean_proto)
    #         else:
    #             local_mean_prototypes.append(None)
    #
    #     for class_prototypes in global_prototypes:
    #
    #         if not class_prototypes == []:
    #             # Stack the tensors for the current class
    #             stacked_protos = torch.stack(class_prototypes)
    #
    #             # Compute the mean tensor for the current class
    #             mean_proto = torch.mean(stacked_protos, dim=0)
    #             global_mean_prototypes.append(mean_proto)
    #         else:
    #             global_mean_prototypes.append(None)
    #
    #     alignment_optimizer = torch.optim.SGD(global_model.parameters(),
    #                                           lr=0.01)  # Adjust learning rate and optimizer as needed
    #     alignment_loss_fn = torch.nn.MSELoss()
    #
    #     for _ in range(1):  # Iterate for 1 epochs; adjust as needed
    #         loss = 0
    #         # if local_mean_prototypes is not None:
    #         #     loss += alignment_loss_fn(local_mean_prototypes, global_mean_prototypes)
    #         #     # print("align loss",loss)
    #         #     alignment_optimizer.zero_grad()
    #         #     loss.backward()
    #         #     alignment_optimizer.step()
    #         # 确保 valid_local_prototypes 和 valid_global_prototypes 都是张量
    #         valid_local_prototypes = [proto for proto in local_mean_prototypes if proto is not None]
    #         valid_global_prototypes = [proto for proto in global_mean_prototypes if proto is not None]
    #
    #         # 如果 valid_local_prototypes 和 valid_global_prototypes 都不为空
    #         if valid_local_prototypes and valid_global_prototypes:
    #             local_tensor = torch.stack(valid_local_prototypes, dim=0).requires_grad_()  # 设置 requires_grad=True
    #             global_tensor = torch.stack(valid_global_prototypes, dim=0).requires_grad_()  # 设置 requires_grad=True
    #
    #             # 计算 MSE 损失
    #             loss = alignment_loss_fn(local_tensor, global_tensor)
    #             alignment_optimizer.zero_grad()
    #             loss.backward()
    #             alignment_optimizer.step()
    #
    #     # Substitute the parameters of the base, enabling personalization
    #     for new_param, old_param in zip(global_model.parameters(), self.model.base.parameters()):
    #         old_param.data = new_param.data.clone()

    # def alignModels(self,global_model):
    #     # Get class-specific prototypes from the local model
    #     local_prototypes = [[] for _ in range(self.num_classes)]
    #     global_prototypes = [[] for _ in range(self.num_classes)]
    #     train_loader = self.get_train_loader()
    #
    #     # print(f'client{id}')
    #     for x_batch, y_batch in train_loader:
    #         x_batch = x_batch.to(self.device)
    #         y_batch = y_batch.to(self.device)
    #
    #         with torch.no_grad():
    #             local_proto_batch = self.model.base(x_batch)
    #             global_proto_batch = global_model(x_batch)
    #
    #         # Scatter the prototypes based on their labels
    #         for proto, y in zip(local_proto_batch, y_batch):
    #             local_prototypes[y.item()].append(proto)
    #
    #         for proto, y in zip(global_proto_batch, y_batch):
    #             global_prototypes[y.item()].append(proto)
    #
    #     local_mean_prototypes = []
    #     global_mean_prototypes = []
    #
    #     # print(f'client{self.id}')
    #     for class_prototypes in local_prototypes:
    #
    #         if not class_prototypes == []:
    #             # Stack the tensors for the current class
    #             stacked_protos = torch.stack(class_prototypes)
    #
    #             # Compute the mean tensor for the current class
    #             mean_proto = torch.mean(stacked_protos, dim=0)
    #             local_mean_prototypes.append(mean_proto)
    #         else:
    #             local_mean_prototypes.append(None)
    #
    #     for class_prototypes in global_prototypes:
    #
    #         if not class_prototypes == []:
    #             # Stack the tensors for the current class
    #             stacked_protos = torch.stack(class_prototypes)
    #
    #             # Compute the mean tensor for the current class
    #             mean_proto = torch.mean(stacked_protos, dim=0)
    #             global_mean_prototypes.append(mean_proto)
    #         else:
    #             global_mean_prototypes.append(None)
    #
    #     alignment_optimizer = torch.optim.SGD(global_model.parameters(),
    #                                           lr=0.01)  # Adjust learning rate and optimizer as needed
    #     alignment_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
    #
    #     for _ in range(1):  # Iterate for 1 epochs; adjust as needed
    #         loss = 0
    #         # if local_mean_prototypes is not None:
    #         #     loss += alignment_loss_fn(local_mean_prototypes, global_mean_prototypes)
    #         #     # print("align loss",loss)
    #         #     alignment_optimizer.zero_grad()
    #         #     loss.backward()
    #         #     alignment_optimizer.step()
    #         # 确保 valid_local_prototypes 和 valid_global_prototypes 都是张量
    #         valid_local_prototypes = [proto for proto in local_mean_prototypes if proto is not None]
    #         valid_global_prototypes = [proto for proto in global_mean_prototypes if proto is not None]
    #
    #         # 如果 valid_local_prototypes 和 valid_global_prototypes 都不为空
    #         if valid_local_prototypes and valid_global_prototypes:
    #             local_tensor = torch.stack(valid_local_prototypes, dim=0).requires_grad_()  # 设置 requires_grad=True
    #             global_tensor = torch.stack(valid_global_prototypes, dim=0).requires_grad_()  # 设置 requires_grad=True
    #
    #             # 计算 MSE 损失
    #             loss += alignment_loss_fn(F.log_softmax(global_tensor, dim=1),
    #                                       F.log_softmax(local_tensor, dim=1),)
    #
    #             alignment_optimizer.zero_grad()
    #             alignment_optimizer.zero_grad()
    #             loss.backward()
    #             alignment_optimizer.step()
    #
    #     # Substitute the parameters of the base, enabling personalization
    #     for new_param, old_param in zip(global_model.parameters(), self.model.base.parameters()):
    #         old_param.data = new_param.data.clone()


    # def alignModels(self,global_model):
    #     # Get class-specific prototypes from the local model
    #     local_prototypes = [[] for _ in range(self.num_classes)]
    #     global_prototypes = [[] for _ in range(self.num_classes)]
    #     train_loader = self.get_train_loader()
    #
    #     alignment_optimizer = torch.optim.SGD(global_model.parameters(),
    #                                           lr=0.01)  # Adjust learning rate and optimizer as needed
    #     alignment_loss_fn = torch.nn.MSELoss()
    #
    #     for _ in range(1):  # Iterate for 1 epochs; adjust as needed
    #         # print(f'client{id}')
    #         for x_batch, y_batch in train_loader:
    #             x_batch = x_batch.to(self.device)
    #             y_batch = y_batch.to(self.device)
    #
    #             with torch.no_grad():
    #                 local_proto_batch = self.model.base(x_batch)
    #                 global_proto_batch = global_model(x_batch)
    #
    #             # Scatter the prototypes based on their labels
    #             for proto, y in zip(local_proto_batch, y_batch):
    #                 local_prototypes[y.item()].append(proto)
    #
    #             for proto, y in zip(global_proto_batch, y_batch):
    #                 global_prototypes[y.item()].append(proto)
    #
    #         local_mean_prototypes = []
    #         global_mean_prototypes = []
    #
    #         # print(f'client{self.id}')
    #         for class_prototypes in local_prototypes:
    #
    #             if not class_prototypes == []:
    #                 # Stack the tensors for the current class
    #                 stacked_protos = torch.stack(class_prototypes)
    #
    #                 # Compute the mean tensor for the current class
    #                 mean_proto = torch.mean(stacked_protos, dim=0)
    #                 local_mean_prototypes.append(mean_proto)
    #             else:
    #                 local_mean_prototypes.append(None)
    #
    #         for class_prototypes in global_prototypes:
    #
    #             if not class_prototypes == []:
    #                 # Stack the tensors for the current class
    #                 stacked_protos = torch.stack(class_prototypes)
    #
    #                 # Compute the mean tensor for the current class
    #                 mean_proto = torch.mean(stacked_protos, dim=0)
    #                 global_mean_prototypes.append(mean_proto)
    #             else:
    #                 global_mean_prototypes.append(None)
    #         valid_local_prototypes = [proto for proto in local_mean_prototypes if proto is not None]
    #         valid_global_prototypes = [proto for proto in global_mean_prototypes if proto is not None]
    #
    #         # 如果 valid_local_prototypes 和 valid_global_prototypes 都不为空
    #         if valid_local_prototypes and valid_global_prototypes:
    #             local_tensor = torch.stack(valid_local_prototypes, dim=0).requires_grad_()  # 设置 requires_grad=True
    #             global_tensor = torch.stack(valid_global_prototypes, dim=0).requires_grad_()  # 设置 requires_grad=True
    #
    #             # 计算 MSE 损失
    #             loss = alignment_loss_fn(local_tensor, global_tensor)
    #             alignment_optimizer.zero_grad()
    #             loss.backward()
    #             alignment_optimizer.step()
    #
    #     # Substitute the parameters of the base, enabling personalization
    #     for new_param, old_param in zip(global_model.parameters(), self.model.base.parameters()):
    #         old_param.data = new_param.data.clone()

    # def alignModels(self,global_model):
    #     # Get class-specific prototypes from the local model
    #     local_prototypes = [[] for _ in range(self.num_classes)]
    #     global_prototypes = [[] for _ in range(self.num_classes)]
    #     train_loader = self.get_train_loader()
    #     for x_batch, y_batch in train_loader:
    #         x_batch = x_batch.to(self.device)
    #         y_batch = y_batch.to(self.device)
    #
    #         with torch.no_grad():
    #             local_proto_batch = self.model.base(x_batch)
    #         # Scatter the prototypes based on their labels
    #         for proto, y in zip(local_proto_batch, y_batch):
    #             local_prototypes[y.item()].append(proto)
    #
    #     local_mean_prototypes = []
    #
    #     for class_prototypes in local_prototypes:
    #
    #         if not class_prototypes == []:
    #             # Stack the tensors for the current class
    #             stacked_protos = torch.stack(class_prototypes)
    #
    #             # Compute the mean tensor for the current class
    #             mean_proto = torch.mean(stacked_protos, dim=0)
    #             local_mean_prototypes.append(mean_proto)
    #         else:
    #             local_mean_prototypes.append(None)
    #     alignment_optimizer = torch.optim.SGD(global_model.parameters(),
    #                                           lr=0.01)  # Adjust learning rate and optimizer as needed
    #     alignment_loss_fn = torch.nn.MSELoss()
    #
    #     for _ in range(2):  # Iterate for 1 epochs; adjust as needed
    #         # print(f'client{id}')
    #         train_loader = self.get_train_loader()
    #         for x_batch, y_batch in train_loader:
    #             x_batch = x_batch.to(self.device)
    #             y_batch = y_batch.to(self.device)
    #
    #             with torch.no_grad():
    #                 global_proto_batch = global_model(x_batch)
    #
    #             for proto, y in zip(global_proto_batch, y_batch):
    #                 global_prototypes[y.item()].append(proto)
    #
    #         global_mean_prototypes = []
    #
    #         for class_prototypes in global_prototypes:
    #
    #             if not class_prototypes == []:
    #                 # Stack the tensors for the current class
    #                 stacked_protos = torch.stack(class_prototypes)
    #
    #                 # Compute the mean tensor for the current class
    #                 mean_proto = torch.mean(stacked_protos, dim=0)
    #                 global_mean_prototypes.append(mean_proto)
    #             else:
    #                 global_mean_prototypes.append(None)
    #         valid_local_prototypes = [proto for proto in local_mean_prototypes if proto is not None]
    #         valid_global_prototypes = [proto for proto in global_mean_prototypes if proto is not None]
    #
    #         # 如果 valid_local_prototypes 和 valid_global_prototypes 都不为空
    #         if valid_local_prototypes and valid_global_prototypes:
    #             local_tensor = torch.stack(valid_local_prototypes, dim=0).requires_grad_()  # 设置 requires_grad=True
    #             global_tensor = torch.stack(valid_global_prototypes, dim=0).requires_grad_()  # 设置 requires_grad=True
    #
    #             # 计算 MSE 损失
    #             loss = alignment_loss_fn(local_tensor, global_tensor)
    #             alignment_optimizer.zero_grad()
    #             loss.backward()
    #             alignment_optimizer.step()
    #
    #     # Substitute the parameters of the base, enabling personalization
    #     for new_param, old_param in zip(global_model.parameters(), self.model.base.parameters()):
    #         old_param.data = new_param.data.clone()
