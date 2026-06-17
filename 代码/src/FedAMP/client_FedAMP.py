import copy

import torch
from tqdm import tqdm

from src.client_base import ClientBase


class ClientFedAMP(ClientBase):
    def __init__(self, args, client_id, trainset, testset, statistic):
        super().__init__(args, client_id, trainset, testset, statistic)

        self.model = copy.deepcopy(args.model)

        self.alpha = args.client["alpha"]
        self.lamb_da = args.client["lambda"]
        self.global_model = None

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)

    def regulation_term(self, wi, wj):
        wi = torch.cat([torch.flatten(param) for param in wi.values()])
        wj = torch.cat([torch.flatten(param) for param in wj.values()])
        norm = torch.norm(wi - wj)
        return 0.5 * self.lamb_da / self.alpha * (norm ** 2)

    def train(self):
        train_loader = self.get_train_loader()
        self.model.train()

        for epoch in range(self.local_epochs):
            local_iter = tqdm(train_loader, desc=f"client {self.client_id} epoch: {epoch}")
            for x, y in local_iter:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                loss = self.loss(self.model(x), y)

                if self.global_model is not None:
                    loss += self.regulation_term(self.model.state_dict(), self.global_model)

                loss.backward()
                self.optimizer.step()

                local_iter.set_description(f"client {self.client_id} epoch: {epoch} loss: {loss.item():.4f}")
