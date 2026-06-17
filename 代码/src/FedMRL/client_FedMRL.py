import copy
import copy
import torch
from torch import nn
from tqdm import tqdm
from src.client_base import ClientBase
from models.models import BaseHeadSplit

class ClientFedMRL(ClientBase):
    def __init__(self, args, client_id, trainset, testset, statistic):
        super().__init__(args, client_id, trainset, testset, statistic)

        self.sub_feature_dim = args.sub_feature_dim
        # self.model = copy.deepcopy(args.model)
        self.global_model = BaseHeadSplit(args, 0, self.sub_feature_dim).to(args.device)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)



        self.proj = nn.Linear(args.feature_dim + self.sub_feature_dim, args.feature_dim).to(self.device)

    def train(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        optimizer_p = torch.optim.SGD(self.proj.parameters(), lr=self.learning_rate)
        optimizer_g = torch.optim.SGD(self.global_model.parameters(), lr=self.learning_rate)

        train_loader = self.get_train_loader()
        self.model.train()

        for epoch in range(self.local_epochs):
            local_iter = tqdm(train_loader, desc=f"client {self.client_id} epoch: {epoch}")
            for x, y in local_iter:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                rep_g = self.global_model.base(x)
                rep = self.model.base(x)
                rep_concat = torch.concat((rep_g, rep), dim=1)
                rep_new = self.proj(rep_concat)
                output_g = self.global_model.head(rep_new[:, :self.sub_feature_dim])
                output = self.model.head(rep_new)
                loss = self.loss(output, y) + self.loss(output_g, y)

                optimizer.zero_grad()
                optimizer_p.zero_grad()
                optimizer_g.zero_grad()
                loss.backward()
                # # prevent divergency on specifical tasks
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                # torch.nn.utils.clip_grad_norm_(global_model.parameters(), 10)
                optimizer.step()
                optimizer_p.step()
                optimizer_g.step()

                local_iter.set_description(f"client {self.client_id} epoch: {epoch} loss: {loss.item():.4f}")


    def evaluate_test(self):
        test_loader = self.get_test_loader()

        self.model.eval()

        class_correct = [0] * self.num_classes
        class_total = [0] * self.num_classes

        with torch.inference_mode():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)

                rep_g = self.global_model.base(x)
                rep = self.model.base(x)
                rep_concat = torch.concat((rep_g, rep), dim=1)
                rep_new = self.proj(rep_concat)
                output = self.model.head(rep_new)
                preds = torch.argmax(output, dim=1)

                # 统计每个类别的正确预测数和总样本数
                for i in range(self.num_classes):
                    class_mask = (y == i)
                    class_total[i] += torch.sum(class_mask).item()
                    class_correct[i] += torch.sum((preds == y) & class_mask).item()

        # 计算并打印每个类别的准确率
        # for i in range(self.num_classes):
        #     if class_total[i] > 0:
        #         accuracy = 100 * class_correct[i] / class_total[i]
        #         print(f'Class {i} Accuracy: {accuracy:.2f}%')
        #     else:
        #         print(f'Class {i} Accuracy: N/A (no samples)')

        # 返回总的正确预测数和总样本数
        total_correct = sum(class_correct)
        total_samples = sum(class_total)
        return total_correct, total_samples,class_correct,class_total
