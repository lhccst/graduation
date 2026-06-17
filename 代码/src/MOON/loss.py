import torch
import torch.nn as nn


class GlobalConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()

        self.temperature = temperature

    def forward(self, features, global_protos):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if len(features.shape) < 3:
            features = features.unsqueeze(dim=1)

        global_similarity = torch.exp(torch.matmul(features, global_protos[0].T) / self.temperature)
        previous_similarity = torch.exp(torch.matmul(features, global_protos[1].T) / self.temperature)

        loss = - torch.log(global_similarity / (global_similarity + previous_similarity))
        loss = torch.mean(loss)

        return loss
