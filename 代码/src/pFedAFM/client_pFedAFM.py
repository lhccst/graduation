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

class ClientpFedAFM(ClientBase):
    def __init__(self, args, client_id, trainset, testset, statistic):
        super().__init__(args, client_id, trainset, testset, statistic)
        self.alpha_model = vector_alpha()
        self.net_small = copy.deepcopy(BaseHeadSplit(args, 0)).to(self.device)

        # self.base.fc = nn.AdaptiveAvgPool1d(args.feature_dim)
        # self.GM_small = copy.deepcopy(self.net_small.state_dict())

        # model_count = len(args.models)
        self.PM_large = copy.deepcopy(self.model).state_dict()
        self.net_large = copy.deepcopy(self.model)
        self.ALPHA = torch.ones(args.feature_dim)
        self.alpha_model = self.alpha_model.to(self.device)

    def test_acc(self,net_large, net_small, net_alpha, testloader, criteria):
        # net.eval()
        self.net_large.eval()
        self.alpha_model.eval()
        self.net_small.eval()
        with torch.no_grad():
            test_acc = 0
            num_batch = 0

            for batch in testloader:
                num_batch += 1
                # batch = next(iter(testloader))

                img, label = tuple(t.to(self.device) for t in batch)
                # pred,_ = net(img,torch.zeros(500).to(device),torch.ones(500).to(device)) # img, homo_rep, alpha_vector
                small_rep = self.net_small.base(img)

                alpha = net_alpha.alpha

                small_rep = net_alpha(small_rep)
                o1 = self.net_large.base(img)  ## 64*84
                o = o1 * alpha.to(small_rep.device) + small_rep
                large_pred = self.net_large.head(o)
                # large_pred, mix_rep = net_large(img, small_rep, alpha)

                test_loss = criteria(large_pred, label)
                test_acc += large_pred.argmax(1).eq(label).sum().item() / len(label)
            mean_test_loss = test_loss / num_batch
            mean_test_acc = test_acc / num_batch
        return mean_test_loss, mean_test_acc

    def evaluate_test(self):
        self.net_large.eval()
        self.alpha_model.eval()
        self.net_small.eval()
        # net.eval()
        test_loader = self.get_test_loader()
        class_correct = [0] * self.num_classes
        class_total = [0] * self.num_classes
        with torch.no_grad():
            test_acc = 0
            num_batch = 0
            for batch in test_loader:
                num_batch += 1
                # batch = next(iter(testloader))

                img, label = tuple(t.to(self.device) for t in batch)
                # pred,_ = net(img,torch.zeros(500).to(device),torch.ones(500).to(device)) # img, homo_rep, alpha_vector
                small_rep = self.net_small.base(img)

                alpha = self.alpha_model.alpha

                small_rep = self.alpha_model(small_rep)
                o1 = self.net_large.base(img)  ## 64*84
                o = o1 * alpha.to(small_rep.device) + small_rep
                large_pred = self.net_large.head(o)
                # large_pred, mix_rep = net_large(img, small_rep, alpha)

                preds = torch.argmax(large_pred, dim=1)
                for i in range(self.num_classes):
                    class_mask = (label == i)
                    class_total[i] += torch.sum(class_mask).item()
                    class_correct[i] += torch.sum((preds == label) & class_mask).item()

            total_correct = sum(class_correct)
            total_samples = sum(class_total)
        return total_correct, total_samples, class_correct, class_total

    def train(self) -> None:

        ###############################
        # init nodes, hnet, local net #
        ###############################
        # nodes = BaseNodes(data_name, data_path, num_nodes, classes_per_node=classes_per_node,
        #                   LowProb=LowProb, batch_size=bs)

        # -------compute aggregation weights-------------#
        # train_sample_count = nodes.train_sample_count
        # eval_sample_count = nodes.eval_sample_count
        # test_sample_count = nodes.test_sample_count

        # client_sample_count = [train_sample_count[i] + eval_sample_count[i] + test_sample_count[i] for i in
        #                        range(len(train_sample_count))]
        # client_sample_count = [sum(client_stat.values()) for client_stat in self.statistics]
        # -----------------------------------------------#

        # if data_name == "cifar10":
        #     net_1 = CNN_1_hetero_AFM(n_kernels=n_kernels)
        #     net_2 = CNN_2_hetero_AFM(n_kernels=n_kernels)
        #     net_3 = CNN_3_hetero_AFM(n_kernels=n_kernels)
        #     net_4 = CNN_4_hetero_AFM(n_kernels=n_kernels)
        #     net_5 = CNN_5_hetero_AFM(n_kernels=n_kernels)
        #
        #     net_small = CNN_5_homo_AFM(n_kernels=n_kernels)
        # elif data_name == "cifar100":
        #     net_1 = CNN_1_hetero_AFM(n_kernels=n_kernels, out_dim=100)
        #     net_2 = CNN_2_hetero_AFM(n_kernels=n_kernels, out_dim=100)
        #     net_3 = CNN_3_hetero_AFM(n_kernels=n_kernels, out_dim=100)
        #     net_4 = CNN_4_hetero_AFM(n_kernels=n_kernels, out_dim=100)
        #     net_5 = CNN_5_hetero_AFM(n_kernels=n_kernels, out_dim=100)
        #
        #     net_small = CNN_5_homo_AFM(n_kernels=n_kernels, out_dim=100)
        # elif data_name == "mnist":
        #     net = CNN_1_hetero_AFM(n_kernels=n_kernels)
        # else:
        #     raise ValueError("choose data_name from ['cifar10', 'cifar100']")
        #
        # net_1 = net_1.to(device)
        # net_2 = net_2.to(device)
        # net_3 = net_3.to(device)
        # net_4 = net_4.to(device)
        # net_5 = net_5.to(device)
        # net_set = [net_1, net_2, net_3, net_4, net_5]

        # net_small = net_small.to(device)

        # alpha_model = self.alpha_model.to(device)

        ##################
        # init optimizer #
        ##################
        lr = 1e-2
        lr_alpha = 1e-2
        wd = 1e-3

        optimizer_small = torch.optim.SGD(params=self.net_small.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        # optimizer_large = torch.optim.SGD(params=net_large.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        optimizer_alpha = torch.optim.SGD(params=self.alpha_model.parameters(), lr=lr_alpha, momentum=0.9, weight_decay=wd)
        # optimizer_alpha = torch.optim.Adam(alpha_model.parameters(), lr=lr_alpha)

        # times = 2
        # interval = 5
        # scheduler_lambda = lambda epoch: times ** (epoch // interval)

        # scheduler = LambdaLR(optimizer_alpha, lr_lambda=scheduler_lambda)

        criteria = torch.nn.CrossEntropyLoss()


        ################
        # init metrics #
        ################
        # step_iter = trange(steps)

        # GM_small = copy.deepcopy(net_small.state_dict())
        #
        # PM_large_acc = defaultdict()
        # PM_small = defaultdict()
        # PM_large = defaultdict()
        # PM_gate = defaultdict()
        # PM_mix_acc = defaultdict()
        # ALPHA = defaultdict()
        # Alpha_mean = defaultdict()
        #
        # for i in range(num_nodes):
        #     PM_large_acc[i] = 0
        #     PM_mix_acc[i] = 0
        #     PM_small[i] = GM_small
        #     PM_large[i] = copy.deepcopy(net_set[i % 5].state_dict())
        #     ALPHA[i] = torch.ones(500)
        #     Alpha_mean[i] = torch.mean(ALPHA[i])
        #
        # save_path = Path(save_path)
        # save_path.mkdir(parents=True, exist_ok=True)
        small_local_trained_loss = []
        small_local_trained_acc = []
        small_global_loss = []
        small_global_acc = []
        large_local_trained_loss = []
        large_local_trained_acc = []
        results = []

        clear_local_large_loss = []
        clear_local_large_acc = []

        LNs = defaultdict()  # colloect small_local_model

        # logging.info(f'#----Round:{step}----#')
        # net_small.load_state_dict(GM_small)

        # net_large = net_set[node_id % 5]
        self.net_large.load_state_dict(self.PM_large)
        optimizer_large = torch.optim.SGD(params=self.net_large.parameters(), lr=lr, momentum=0.9,
                                          weight_decay=wd)

        self.alpha_model.alpha.data = torch.tensor(self.ALPHA)  # use alpha in the last round
        train_loader = self.get_train_loader()
        test_loader = self.get_test_loader()
        # evlaute GM
        # global_loss,  global_acc = test_acc(net_small, nodes.test_loaders[node_id], criteria)
        # global_loss, global_acc = self.test_acc_global(self.net_large, self.net_small, test_loader,
        #                                                criteria)
        # small_global_loss.append(global_loss.cpu().item())
        # small_global_acc.append(global_acc)

        # step1: freeze global extractor, train local NN
        for epoch in range(self.local_epochs):
            self.net_large.train()
            self.alpha_model.train()
            local_iter = tqdm(train_loader, desc=f"client {self.client_id} epoch: {epoch}")
            for x, y in local_iter:
                img, label = x.to(self.device), y.to(self.device)

                small_rep = self.net_small.base(img)

                alpha = self.alpha_model.alpha

                small_rep = self.alpha_model(small_rep)

                o1 = self.net_large.base(img)  ## 64*84
                o = o1 * alpha.to(small_rep.device) + small_rep
                large_pred = self.net_large.head(o)

                loss = criteria(large_pred, label)

                optimizer_large.zero_grad()
                # optimizer_small.zero_grad()
                optimizer_alpha.zero_grad()

                loss.backward(retain_graph=True)

                torch.nn.utils.clip_grad_norm_(self.net_large.parameters(), 50)
                # torch.nn.utils.clip_grad_norm_(net_small.parameters(), 50)

                # torch.nn.utils.clip_grad_norm_(alpha_model.parameters(), 1)

                self.alpha_model.alpha.data = torch.clamp(self.alpha_model.alpha.data, 0, 1)

                optimizer_large.step()
                # optimizer_small.step()
                optimizer_alpha.step()
                self.alpha_model.alpha.data = torch.clamp(self.alpha_model.alpha.data, 0, 1)
                # 显示训练进度
                local_iter.set_description(f"client {self.client_id} epoch: {epoch} loss: {loss.item():.4f}")

        # collect local NN parameters
        self.PM_large = copy.deepcopy(self.net_large.state_dict())
        self.ALPHA = alpha
        Alpha_mean = torch.mean(alpha)
        # print(f'alpha:{torch.mean(alpha)}')

        # mix acc test
        trained_loss, trained_acc = self.test_acc(self.net_large, self.net_small, self.alpha_model, test_loader,
                                                  criteria)
        large_local_trained_loss.append(trained_loss.cpu().item())
        large_local_trained_acc.append(trained_acc)
        PM_large_acc = trained_acc
        #
        # # local large acc test
        # trained_loss1, trained_acc1 = self.test_acc(net_large, net_small, alpha_model,
        #                                             test_loader, criteria)
        # clear_local_large_loss.append(trained_loss1.cpu().item())
        # clear_local_large_acc.append(trained_acc1)

        # step2: freeze local header, train global extractor
        for i in range(self.local_epochs):
            self.net_small.train()

            for j, batch in enumerate(train_loader):
                img, label = tuple(t.to(self.device) for t in batch)

                small_pred = self.net_small(img)

                # alpha = alpha_model.alpha.item()
                # small_rep = alpha_model(small_rep)
                # large_pred = self.net_large(img)

                loss = criteria(small_pred, label)

                optimizer_small.zero_grad()

                loss.backward(retain_graph=True)

                torch.nn.utils.clip_grad_norm_(self.net_small.parameters(), 50)

                optimizer_small.step()

        # LNs[node_id] = net_small.state_dict()

        # evaluate trained local model
        # trained_loss, trained_acc = test_acc(net_small, nodes.test_loaders[node_id], criteria)
        # small_local_trained_loss.append(trained_loss.cpu().item())
        # small_local_trained_acc.append(trained_acc)

        # scheduler.step()
        # lr_alpha = optimizer_alpha.param_groups[0]['lr']
        # print(f'current alpha learning rate:{lr_alpha}')
        #
        # if lr_alpha > 0.1:
        #     for param_group in optimizer_alpha.param_groups:
        #         param_group['lr'] = 0.1

        # scheduler_small.step()
        # scheduler_large.step()
        # scheduler_gate.step()
        # LR_gate[node_id] = scheduler_gate.get_last_lr()scheduler_gate
        # print('\t last_lr:', scheduler_gate.get_last_lr())

        mean_small_global_loss = round(np.mean(small_global_loss), 4)
        mean_small_global_acc = round(np.mean(small_global_acc), 4)
        # mean_small_trained_loss = round(np.mean(small_local_trained_loss), 4)
        # mean_small_trained_acc = round(np.mean(small_local_trained_acc), 4)
        mean_large_trained_loss = round(np.mean(large_local_trained_loss), 4)
        mean_large_trained_acc = round(np.mean(large_local_trained_acc), 4)

        mean_clear_local_large_loss = round(np.mean(clear_local_large_loss), 4)
        mean_large_clear_local_large_acc = round(np.mean(clear_local_large_acc), 4)

        # results = [mean_small_global_loss, mean_small_global_acc, mean_small_trained_loss, mean_small_trained_acc, mean_large_trained_loss, mean_large_trained_acc] + [round(i) for i in PM_large_acc.values()]
        # results = [mean_small_global_loss, mean_small_global_acc, mean_large_trained_loss,
        #            mean_large_trained_acc, mean_clear_local_large_loss, mean_large_clear_local_large_acc] + [
        #               round(i, 4) for i in PM_large_acc.values()] + [i.data.item() for i in
        #                                                              Alpha_mean.values()]  # + [i for i in LR_gate.values()]
        print(f'Mean Alpha: {Alpha_mean}')
        print(f'Mean Large Acc: {mean_large_trained_acc}')
        print(f'trained acc: {trained_acc}')

        # logging.info(
        #     f'Round:{step} | small_GM_loss:{mean_small_global_loss} | small_PM_loss:{mean_small_trained_loss} | large_PM_loss:{mean_large_trained_loss}')
        # logging.info(
        #     f'Round:{step} | small_GM_acc:{mean_small_global_acc} | small_PM_acc:{mean_small_trained_acc} | large_PM_acc:{mean_large_trained_acc}')

        # client_agg_weights = OrderedDict()
        # select_nodes_sample_count = OrderedDict()
        # for i in range(len(select_nodes)):
        #     select_nodes_sample_count[select_nodes[i]] = client_sample_count[select_nodes[i]]
        # for i in range(len(select_nodes)):
        #     client_agg_weights[select_nodes[i]] = select_nodes_sample_count[select_nodes[i]] / sum(
        #         select_nodes_sample_count.values())
        #
        # weight_keys = list(net_small.state_dict().keys())
        # for key in weight_keys:
        #     key_sum = 0
        #     for id, model in LNs.items():
        #         key_sum += client_agg_weights[id] * model[key]
        #     GM_small[key] = key_sum
        # logging.info(f'Global model is updated after aggregation')


class vector_alpha(nn.Module):
    def __init__(self):
        super(vector_alpha, self).__init__()

        self.alpha = nn.Parameter(torch.ones(84))

    def forward(self, small_input):
        output = (1-self.alpha.to(small_input.device)) * small_input
        return output
