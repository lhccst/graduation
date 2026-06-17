import copy
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from src.client_base import ClientBase


class ClientFPL(ClientBase):
    def __init__(self, args, client_id, trainset, testset, statistic):
        super().__init__(args, client_id, trainset, testset, statistic)

        # self.model = copy.deepcopy(args.model)
        self.global_protos = []
        self.infoNCET = args.client["infoNCET"]
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)

    def hierarchical_info_loss(self, f_now, label, all_f, mean_f, all_global_protos_keys):
        all_f = [f.unsqueeze(0) if f.shape == torch.Size([84]) else f for f in all_f]
        f_idx = np.where(all_global_protos_keys == label.item())[0][0]
        f_pos = all_f[f_idx].to(self.device)
        # for i, f in enumerate(all_f):
        #     print("f!!!!", f.shape)

        f_neg = torch.cat([f for i, f in enumerate(all_f) if i != f_idx]).to(self.device)
        # f_neg = torch.cat([f.flatten() for i, f in enumerate(all_f) if i != f_idx]).to(self.device)

        mean_f_pos = mean_f[f_idx].to(self.device)

        # mask = all_global_protos_keys == label.item()
        # f_pos = torch.cat([f for f, m in zip(all_f, mask) if m]).to(self.device)
        # f_neg = torch.cat([f for f, m in zip(all_f, mask) if not m]).to(self.device)
        xi_info_loss = self.calculate_infonce(f_now, f_pos, f_neg)
        # f_pos = np.array(all_f)[all_global_protos_keys == label.item()][0].to(self.device)
        # f_neg = torch.cat(list(np.array(all_f)[all_global_protos_keys != label.item()])).to(self.device)

        # mean_f_pos = np.array(mean_f)[all_global_protos_keys == label.item()][0].to(self.device)
        # mean_f_pos = torch.cat([f for f, m in zip(mean_f, mask) if m]).to(self.device)
        mean_f_pos = mean_f_pos.view(1, -1)
        # mean_f_neg = torch.cat(list(np.array(mean_f)[all_global_protos_keys != label.item()]), dim=0).to(self.device)
        # mean_f_neg = mean_f_neg.view(9, -1)

        loss_mse = nn.MSELoss()
        cu_info_loss = loss_mse(f_now, mean_f_pos)

        hierar_info_loss = xi_info_loss + cu_info_loss
        return hierar_info_loss

    # CPCL Loss
    def calculate_infonce(self, f_now, f_pos, f_neg):
        # print("f_pos",f_pos.shape)
        # print("f_neg",f_neg.shape)
        f_proto = torch.cat((f_pos, f_neg), dim=0)
        l = torch.cosine_similarity(f_now, f_proto, dim=1)
        l = l / self.infoNCET

        exp_l = torch.exp(l)
        exp_l = exp_l.view(1, -1)
        pos_mask = [1 for _ in range(f_pos.shape[0])] + [0 for _ in range(f_neg.shape[0])]
        pos_mask = torch.tensor(pos_mask, dtype=torch.float).to(self.device)
        pos_mask = pos_mask.view(1, -1)
        # pos_l = torch.einsum('nc,ck->nk', [exp_l, pos_mask])
        pos_l = exp_l * pos_mask
        sum_pos_l = pos_l.sum(1)
        sum_exp_l = exp_l.sum(1)
        infonce_loss = -torch.log(sum_pos_l / sum_exp_l)
        return infonce_loss

    def agg_func(self, protos):
        """
        Returns the average of the weights.
        """

        for [label, proto_list] in protos.items():
            # proto_list = F.normalize(torch.stack(proto_list, dim=0))

            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                protos[label] = proto / len(proto_list)
            else:
                protos[label] = proto_list[0]

        return protos

    def train(self):
        train_loader = self.get_train_loader()
        self.model.train()

        if len(self.global_protos) != 0:
            all_global_protos_keys = np.array(list(self.global_protos.keys()))
            all_f = []
            mean_f = []
            for protos_key in all_global_protos_keys:
                temp_f = self.global_protos[protos_key]
                temp_f = torch.cat(temp_f, dim=0).to(self.device)
                all_f.append(temp_f.cpu())
                mean_f.append(torch.mean(temp_f, dim=0).cpu())
            all_f = [item.detach() for item in all_f]
            mean_f = [item.detach() for item in mean_f]

        for epoch in range(self.local_epochs):
            local_iter = tqdm(train_loader, desc=f"client {self.client_id} epoch: {epoch}")
            agg_protos_label = {}
            for x, y in local_iter:
                x, y = x.to(self.device), y.to(self.device)

                feature = self.model.base(x)
                output = self.model.head(feature)

                lossCE = self.loss(output, y)  # crossentropy loss

                if len(self.global_protos) == 0:
                    loss_InfoNCE = 0 * lossCE
                else:
                    i = 0
                    loss_InfoNCE = None

                    for label in y:
                        if label.item() in self.global_protos.keys():
                            f_now = feature[i].unsqueeze(0)
                            loss_instance = self.hierarchical_info_loss(f_now, label, all_f, mean_f,
                                                                        all_global_protos_keys)

                            if loss_InfoNCE is None:
                                loss_InfoNCE = loss_instance
                            else:
                                loss_InfoNCE += loss_instance
                        i += 1
                    loss_InfoNCE = loss_InfoNCE / i
                loss_InfoNCE = loss_InfoNCE

                loss = lossCE + loss_InfoNCE
                # loss = lossCE
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if epoch == self.local_epochs - 1:
                    for i in range(len(y)):
                        if y[i].item() in agg_protos_label:
                            agg_protos_label[y[i].item()].append(feature[i, :])
                        else:
                            agg_protos_label[y[i].item()] = [feature[i, :]]

                local_iter.set_description(f"client {self.client_id} epoch: {epoch} loss: {loss.item():.4f}")

        agg_protos = self.agg_func(agg_protos_label)
        self.local_protos = agg_protos
