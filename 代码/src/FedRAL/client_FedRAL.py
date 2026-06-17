import numpy as np
import torch
from tqdm import tqdm

from models.models import BaseHeadSplit
from src.client_base import ClientBase
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from matplotlib import rcParams
import torch.nn.functional as F
import argparse
import copy
import csv
import json
import logging
import random
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from tqdm import trange
from models.FedAvgCNN import FedAvgCNN

class ClientFedRAL(ClientBase):
    def __init__(self, args, client_id, trainset, testset, statistic):
        super().__init__(args, client_id, trainset, testset, statistic)


        # self.base.fc = nn.AdaptiveAvgPool1d(args.feature_dim)
        # self.GM_small = copy.deepcopy(self.net_small.state_dict())

        # model_count = len(args.models)
        self.model = ChangeModel(self.model).to(self.device)
        self.PM = copy.deepcopy(self.model.state_dict())
        # self.net = self.model.to(self.device)
        self.net = copy.deepcopy(self.model).to(self.device)
        wd = 1e-3
        lr_net = 0.01
        self.optimizer_net = torch.optim.SGD(params=self.net.parameters(), lr=lr_net, momentum=0.9, weight_decay=wd)

        self.criteria = torch.nn.CrossEntropyLoss()
        self.Local_oft = copy.deepcopy(self.net.oft.state_dict())# local generators
        self.Local_mask = torch.zeros_like(self.net.oft.R.data)# local generators
        self.block_num = 1
        self.R_dim = 84
        # self.R_dim = args.feature_dim

    def test_acc(self,net, testloader, criteria):
        net.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0.0

        with torch.no_grad():
            for batch in testloader:
                img, label = tuple(t.to(self.device) for t in batch)
                pred = net(img)

                # 累加损失（加权）
                batch_loss = criteria(pred, label)
                total_loss += batch_loss.item() * len(label)  # 重要：乘以批次大小

                # 累加正确预测数
                pred_labels = pred.argmax(dim=1)
                correct = pred_labels.eq(label).sum().item()
                total_correct += correct
                total_samples += len(label)

        # 计算总体指标
        mean_loss = total_loss / total_samples  # 加权平均损失
        accuracy = total_correct / total_samples  # 总体准确率

        return mean_loss, accuracy

    def evaluate_test(self):
        self.net.eval()
        # net.eval()
        test_loader = self.get_test_loader()
        class_correct = [0] * self.num_classes
        class_total = [0] * self.num_classes
        trained_loss, trained_acc = self.test_acc(self.net, test_loader, self.criteria)
        print(f'client: {self.client_id} trained acc: {trained_acc}')

        with torch.no_grad():
            test_acc = 0
            num_batch = 0
            for batch in test_loader:
                num_batch += 1
                # batch = next(iter(testloader))

                img, label = tuple(t.to(self.device) for t in batch)
                # pred,_ = net(img,torch.zeros(500).to(device),torch.ones(500).to(device)) # img, homo_rep, alpha_vector
                pred = self.net(img)

                preds = torch.argmax(pred, dim=1)
                for i in range(self.num_classes):
                    class_mask = (label == i)
                    class_total[i] += torch.sum(class_mask).item()
                    class_correct[i] += torch.sum((preds == label) & class_mask).item()

            total_correct = sum(class_correct)
            total_samples = sum(class_total)
        return total_correct, total_samples, class_correct, class_total

    def train(self) -> None:

        # net_large = net_set[node_id % 5]
        self.net.load_state_dict(self.PM)
        train_loader = self.get_train_loader()
        test_loader = self.get_test_loader()
        self.net.train()
        for epoch in range(self.local_epochs):
            local_iter = tqdm(train_loader, desc=f"client {self.client_id} epoch: {epoch}")
            for x, y in local_iter:
                img, label = x.to(self.device), y.to(self.device)

                self.optimizer_net.zero_grad()

                # pred = self.net(img)
                pred = self.net(img)  ## 64*84
                loss = self.criteria(pred, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 50)
                self.optimizer_net.step()
                # 显示训练进度
                local_iter.set_description(f"client {self.client_id} epoch: {epoch} loss: {loss.item():.4f}")

        # collect local NN parameters
        self.PM = copy.deepcopy(self.net.state_dict())
        self.Local_oft = copy.deepcopy(self.net.oft.state_dict())

        if self.block_num > 1:
            # compress the pure oft matrix for efficient communicatin
            pure_oft = copy.deepcopy(self.net.oft.R.data)
            masked_pure_oft = torch.zeros_like(pure_oft)
            mask = torch.zeros_like(pure_oft)

            block_size = int(self.R_dim / self.block_num)
            for i in range(self.block_num):
                start = i * block_size
                end = (i + 1) * block_size
                masked_pure_oft[start:end, start:end] = pure_oft[start:end, start:end]
                mask[start:end, start:end] = 1
            self.net.oft.R.data = masked_pure_oft
            self.Local_oft = copy.deepcopy(self.net.oft.state_dict())
            self.Local_mask = mask

        # print(f'alpha:{torch.mean(alpha)}')

        # mix acc test
        trained_loss, trained_acc = self.test_acc(self.net, test_loader,self.criteria)

        print(f'trained loss: {trained_loss}')
        print(f'trained acc: {trained_acc}')




class ChangeModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.base = model.base
        self.head = model.head
        self.oft = OFT()

    def forward(self, x):
        rep = self.base(x)
        rep_oft = self.oft(rep)
        # rep_total = rep + 0.1*rep_oft
        out = self.head(rep)
        return out

class OFT(nn.Module):
    def __init__(self):
        super(OFT, self).__init__()
        self.R = nn.Parameter(torch.randn(84, 84))  #  self.fc.weight.data.t()


    def forward(self,x):
        o = torch.matmul(x, self.R)
        return o

