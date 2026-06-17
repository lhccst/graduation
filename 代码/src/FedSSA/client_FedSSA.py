# client_fedssa.py
import copy
import torch
from tqdm import tqdm
from src.client_base import ClientBase


class ClientFedSSA(ClientBase):
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
        self.alpha = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(0.5)
        self.globalHead =copy.deepcopy(args.head).to(self.device)
        self.decay_rounds=2
        self.miu_0 = 0.5 # 0.1,0.3,0.5,0.7,0.9,1.0


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

    def train(self):
        """本地训练过程"""
        self.model.train()
        train_loader = self.get_train_loader()

        # 动态调整alpha
        self.adjust_alpha()

        # 融合全局参数
        self.fuse_global_params()

        for epoch in range(self.local_epochs):
            local_iter = tqdm(train_loader, desc=f"client {self.client_id} epoch: {epoch}")
            for x, y in local_iter:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                loss = self.loss(self.model(x), y)
                loss.backward()
                self.optimizer.step()

                local_iter.set_description(f"client {self.client_id} epoch: {epoch} loss: {loss.item():.4f}")

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 50)
                self.optimizer.step()

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