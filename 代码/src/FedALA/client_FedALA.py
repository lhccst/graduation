import copy

import torch
from tqdm import tqdm

from src.client_base import ClientBase


class ClientFedALA(ClientBase):
    def __init__(self, args, client_id, trainset, testset, statistic):
        super().__init__(args, client_id, trainset, testset, statistic)

        self.model = copy.deepcopy(args.model)
        self.server_model = None

        self.ala = ALA(self.local_epochs, self.learning_rate, self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)

    def train(self):
        train_loader = self.get_train_loader()

        if self.server_model is not None:
            aggregated_base = self.ala.adaptive_aggregate(self.model, self.server_model, train_loader)
            self.model.base.load_state_dict(aggregated_base)

        self.model.train()
        for epoch in range(self.local_epochs):
            local_iter = tqdm(train_loader, desc=f"client {self.client_id} epoch: {epoch}")
            for x, y in local_iter:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                loss = self.loss(self.model(x), y)
                loss.backward()
                self.optimizer.step()

                local_iter.set_description(f"client {self.client_id} epoch: {epoch} loss: {loss.item():.4f}")


class ALA:
    def __init__(self, local_epochs, learning_rate, device):
        self.local_epochs = local_epochs
        self.loss = torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.device = device

        self.weights = None  # learnable weight

    def adaptive_aggregate(self, local_model, server_dict, train_loader):
        local_dict = copy.deepcopy(local_model.base.state_dict())  # 暂存分类模型的参数

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
                local_model.base.load_state_dict(aggregated)

                x, y = x.to(self.device), y.to(self.device)
                output = local_model(x)
                loss = self.loss(output, y)

                # model_optimizer.zero_grad()
                # loss.backward()
                # model_optimizer.step()

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
