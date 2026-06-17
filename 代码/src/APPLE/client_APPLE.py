import copy

import torch
from torch import nn
from tqdm import tqdm

from src.client_base import ClientBase


class ClientAPPLE(ClientBase):
    def __init__(self, args, client_id, trainset, testset, statistic):
        super().__init__(args, client_id, trainset, testset, statistic)

        self.model = copy.deepcopy(args.model)  # 实际用户用于分类模型
        self.agg_model = AggregateModel(args=self.args,
                                        client_id=self.client_id,
                                        num_clients=args.client["num_clients"],
                                        device=args.device).to(args.device)  # 参数聚合模型

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)  # 优化器
        self.agg_optimizer = torch.optim.SGD(
            [
                {"params": self.agg_model.dr},
                {"params": list(self.agg_model.client_models[self.client_id].values())},
            ],
            lr=self.learning_rate,
        )

    def train(self):
        train_loader = self.get_train_loader()

        self.model.train()
        self.agg_model.train()

        for epoch in range(self.local_epochs):
            local_iter = tqdm(train_loader, desc=f"client {self.client_id} epoch: {epoch}")
            for x, y in local_iter:

                aggregated = self.agg_model()  # 就是 Wp
                self.model.load_state_dict(aggregated)

                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)

                self.optimizer.zero_grad()
                self.agg_optimizer.zero_grad()

                loss.backward()

                agg_grad = torch.autograd.grad(
                    outputs=aggregated.values(),
                    inputs=[self.agg_model.dr] + list(self.agg_model.client_models[self.client_id].values()),
                    grad_outputs=[param.grad for param in self.model.parameters()],
                )

                for grad, param in zip(
                        agg_grad,
                        [self.agg_model.dr] + list(self.agg_model.client_models[self.client_id].values())
                ):
                    param.grad = grad

                self.optimizer.step()
                self.agg_optimizer.step()

                local_iter.set_description(f"client {self.client_id} epoch: {epoch} loss: {loss.item():.4f}")


class AggregateModel(nn.Module):
    def __init__(self, args, client_id, num_clients, device):
        super().__init__()

        self.args = args
        self.client_id = client_id
        self.num_clients = num_clients
        self.device = device

        self.dr = nn.Parameter(torch.tensor([1 / num_clients for _ in range(num_clients)]))

        initial_param = self.args.model.state_dict()
        self.client_models = [copy.deepcopy(initial_param) for _ in range(self.num_clients)]

        for param in self.client_models[self.client_id].values():
            param.requires_grad_(True)

    def forward(self):
        aggregated = {}
        for layer in self.args.model.state_dict().keys():
            param = [client_model[layer] for client_model in self.client_models]
            param = torch.stack(param, dim=0)
            shape = [-1] + [1] * (len(param.shape) - 1)
            param_aggregated = torch.sum(param * self.dr.reshape(shape), dim=0)
            aggregated[layer] = param_aggregated
        return aggregated
