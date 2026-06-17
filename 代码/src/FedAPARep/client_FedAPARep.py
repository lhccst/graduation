import copy

import torch
from tqdm import tqdm

from src.client_base import ClientBase


class ClientFedAPARep(ClientBase):
    def __init__(self, args, client_id, trainset, testset, statistic):
        super().__init__(args, client_id, trainset, testset, statistic)

        self.model = copy.deepcopy(args.model)
        self.plocal_epochs = args.client["local_rep_ep"]  # FedREP单独需要的参数
        self.optimizer = torch.optim.SGD(self.model.base.parameters(), lr=self.learning_rate, momentum=0.9)

        self.optimizer_per = torch.optim.SGD(self.model.head.parameters(), lr=self.learning_rate, momentum=0.9)
    def train(self):
        train_loader = self.get_train_loader()
        self.model.train()

        # for epoch in range(self.local_epochs):
        #     local_iter = tqdm(train_loader, desc=f"client {self.client_id} epoch: {epoch}")
        #     for x, y in local_iter:
        #         x, y = x.to(self.device), y.to(self.device)
        #         output = self.model(x)
        #         loss = self.loss(output, y)
        #
        #         self.optimizer.zero_grad()
        #         loss.backward()
        #         self.optimizer.step()
        #
        #         local_iter.set_description(f"client {self.client_id} epoch: {epoch} loss: {loss.item():.4f}")

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

        # max_local_epochs = self.local_epochs
#         for param in self.model.base.parameters():
#             param.requires_grad = True
#         for param in self.model.head.parameters():
#             param.requires_grad = False

#         for epoch in range(max_local_epochs):
#             local_iter = tqdm(train_loader, desc=f"client {self.client_id} epoch: {epoch}")
#             for x, y in local_iter:
#                 # for i, (x, y) in enumerate(trainloader):
#                 if type(x) == type([]):
#                     x[0] = x[0].to(self.device)
#                 else:
#                     x = x.to(self.device)
#                 y = y.to(self.device)
#                 output = self.model(x)
#                 loss = self.loss(output, y)
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#                 local_iter.set_description(f"client {self.client_id} epoch: {epoch} loss: {loss.item():.4f}")
