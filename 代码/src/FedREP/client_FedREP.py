# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import torch
import numpy as np
import time
from src.client_base import ClientBase
from tqdm import tqdm
import copy

class ClientFedREP(ClientBase):
    def __init__(self, args, client_id, trainset, testset, statistic):
        super().__init__(args, client_id, trainset, testset, statistic)
        self.model = copy.deepcopy(args.model)
        self.optimizer = torch.optim.SGD(self.model.base.parameters(), lr=self.learning_rate, momentum=0.9)
        # self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer=self.optimizer,
        #     gamma=args.learning_rate_decay_gamma
        # )
        self.optimizer_per = torch.optim.SGD(self.model.head.parameters(), lr=self.learning_rate, momentum=0.9)
        # self.learning_rate_scheduler_per = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer=self.optimizer_per,
        #     gamma=args.learning_rate_decay_gamma
        # )

        self.plocal_epochs = args.client["local_rep_ep"]  # FedREP单独需要的参数

    def train(self):
        train_loader = self.get_train_loader()
        self.model.train()

        for param in self.model.base.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True

        for epoch in range(self.plocal_epochs):
            p_local_iter = tqdm(train_loader, desc=f"client {self.client_id} epoch: {epoch}")
            for x, y in p_local_iter:
            # for i, (x, y) in enumerate(train_loader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer_per.zero_grad()
                loss.backward()
                self.optimizer_per.step()
                p_local_iter.set_description(f"client {self.client_id} P_local_epoch: {epoch} loss: {loss.item():.4f}")
                
        max_local_epochs = self.local_epochs
        for param in self.model.base.parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = False

        for epoch in range(max_local_epochs):
            local_iter = tqdm(train_loader, desc=f"client {self.client_id} epoch: {epoch}")
            for x, y in local_iter:
            # for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                local_iter.set_description(f"client {self.client_id} epoch: {epoch} loss: {loss.item():.4f}")
