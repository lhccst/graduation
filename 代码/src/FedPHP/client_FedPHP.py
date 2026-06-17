import copy

import torch
from torch import nn

from src.client_base import ClientBase
from tqdm import tqdm


class ClientFedPHP(ClientBase):
    def __init__(self, args, client_id, trainset, testset, statistic):
        super().__init__(args, client_id, trainset, testset, statistic)

        self.mu = args.client["mu"]
        self.lamb_da = args.client["lambda"]
        self.selected_times = 0

        self.server_model = None  # state_dict
        self.previous_model = copy.deepcopy(args.model)
        self.model = copy.deepcopy(args.model)
        
        self.proto_loss = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                         lr=self.learning_rate, 
                                         momentum=0.9)

    def update_parameter(self, join_rate, total_global_rounds):
        mu = self.mu * self.selected_times / (join_rate * total_global_rounds)  # mu 越来约大，越信任全局模型

        local_model_dict = self.model.base.state_dict()

        aggregated = {}
        for key in self.server_model:
            aggregated[key] = (1 - mu) * local_model_dict[key] + mu * self.server_model[key]

        self.previous_model.base.load_state_dict(local_model_dict)
        self.model.base.load_state_dict(aggregated)

    def train(self):
        self.selected_times += 1

        train_loader = self.get_train_loader()
        self.model.train()

        for epoch in range(self.local_epochs):
            local_iter = tqdm(train_loader, desc=f"client {self.client_id} epoch: {epoch}")
            for x, y in local_iter:
                x, y = x.to(self.device), y.to(self.device)

                feature = self.model.base(x)
                output = self.model.head(feature)

                loss = ((1 - self.lamb_da) * self.loss(output, y) +
                        self.lamb_da * self.proto_loss(feature, self.previous_model.base(x)))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                local_iter.set_description(f"client {self.client_id} epoch: {epoch} loss: {loss.item():.4f}")
