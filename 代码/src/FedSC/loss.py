import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super().__init__()

        """ 虽然是模型，但是实际上没有参数，相同于一个简单的函数 """
        self.temperature = temperature  # 温度
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature  # 基础温度

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric 可以不对称.
                如果样本 i 和 j 属于同一个类，那么 mask_{i,j}=1，否则 mask_{i,j}=0
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        ## features(64,1,84) label(64,1)
        ## contrast_feature(64,84)
        if len(features.shape) < 3:
            # raise ValueError('`features` needs to be [bsz, n_views, ...],'
            #                  'at least 3 dimensions are required')
            features = features.unsqueeze(dim=1)
        if len(features.shape) > 3:
            """ 如果维度大于3，把后面的维度合并到第3维 """
            features = features.reshape(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            """ 要么 labels 为 None，要么 mask 为 None，不能同时定义 labels 和 mask """
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            """ 如果 labels 和 mask 都为 None，那坍塌为 unsupervised 的对比学习 """
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            """ 如果 labels 不为 None，通过 labels 得到 masks """
            labels = labels.reshape(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            """ 提供了 labels，如果 labels 计算出 masks，如果样本 i 和 j 的标签相同，则 mask_{i,j}=1，否则 mask_{i,j}=0 """
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            """ 如果提供了 mask，稍微处理一下就行 """
            mask = mask.float().to(device)

        """ contrast_count 也就是 n_views """
        contrast_count = features.shape[1]
        """ 将不同视角得到的 feature 和 anchor 并在一起，成为额外的 anchor """
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            """ contrast_mode 为 one，则只使用原来的 feature 作为 anchor """
            """ 所谓 anchor 就是需要拉近和推远的特征，因此叫锚点 """
            anchor_feature = features[:, 0]
            anchor_count = 1  # anchor_count 为采用的视角数，为 1
        elif self.contrast_mode == 'all':
            """ contrast_mode 为 all，则其他视角得到的 feature 也为 anchor """
            anchor_feature = contrast_feature
            anchor_count = contrast_count  # anchor_count 为采用的视角数
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        """ 通过点积的方式计算所有样本 feature 之间的相似度，得到相似度矩阵 """
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        """ 减去最大值，避免数值不稳定 """
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        """ 原来的 mask 为 [bsz, bsz]，现在扩展为 [bsz * anchor_count, bsz * contrast_count] """
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        """ 将对角线上的元素置为 0，即不考虑自己和自己的相似度 """
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).reshape(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        """ exp_logits 为相似度 logits 取 e 指数 """
        exp_logits = torch.exp(logits) * logits_mask
        """ log 中，真数的相除为对数的相减
            logits 被减数为公式中的分子
            torch.log(exp_logits.sum(1, keepdim=True)) 为公式中的分母
        """
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        """ 得到每个样本的正样本数 """
        mask_pos_pairs = mask.sum(dim=1)
        """ 如果正样本数为 0，将其置为 1，避免除 0 出现 nan。出现 0 的原因是过滤了对角线自成正样本的情况 """
        one = torch.tensor(1.0).float().to(device)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, one, mask_pos_pairs)
        #########
        # sim_mat = torch.div( === anchor_dot_contrast
        #     torch.matmul(anchor_feature, contrast_feature.T),
        #     self.temperature)

        s_dist = F.softmax(anchor_dot_contrast, dim=1)
        sim_mat_noDIV = torch.matmul(anchor_feature, contrast_feature.T)

        beta = 0.1
        sim_mat_exp = torch.exp(-beta * sim_mat_noDIV)
        # ##欧几里得
        # cost_mat = self.EuclideanDistances(anchor_feature, contrast_feature)
        # mean_log_prob_pos = (cost_mat * mask * log_prob).sum(dim=1) / mask_pos_pairs
        ##指数
        # loss = (sim_mat_exp * s_dist).sum(1).mean()
        # mean_log_prob_pos = (sim_mat_exp * mask * log_prob).sum(dim=1) / mask_pos_pairs
        #########
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_pos_pairs
        ############

        # loss
        """ 取反 """
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        """ 对 batch 中的所有样本的损失求均值 """
        loss = loss.reshape(anchor_count, batch_size).mean()

        return loss

    def EuclideanDistances(self, a, b):
        sq_a = a ** 2
        sum_sq_a = torch.sum(sq_a, dim=1).unsqueeze(1)  # m->[m, 1]
        sq_b = b ** 2
        sum_sq_b = torch.sum(sq_b, dim=1).unsqueeze(0)  # n->[1, n]
        bt = b.t()
        dist_sq = sum_sq_a + sum_sq_b - 2 * a.mm(bt)
        dist_sq = torch.clamp(dist_sq, min=0)
        dist = torch.sqrt(dist_sq + 1e-12)
        return dist


class GlobalConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super().__init__()

        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, global_protos=None, mask=None):
        """
        Compute contrastive loss between feature and global prototype
        """
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(features, list):
            # 如果是空列表，返回0损失
            if len(features) == 0:
                return torch.tensor(0.0,device=global_protos.device if global_protos is not None else torch.device('cpu'))
            # 将列表转换为张量
            features = torch.stack(features)

        if len(features.shape) < 3:
            # raise ValueError('`features` needs to be [bsz, n_views, ...],'
            #                  'at least 3 dimensions are required')
            features = features.unsqueeze(dim=1)
        if len(features.shape) > 3:
            features = features.reshape(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.reshape(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            # anchor_feature = contrast_feature
            anchor_count = contrast_count
            anchor_feature = torch.zeros_like(contrast_feature)
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # generate anchor_feature
        for i in range(batch_size * anchor_count):
            anchor_feature[i, :] = global_protos[labels[i % batch_size].item()]

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob 特征矩阵
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mask_pos_pairs = mask.sum(dim=1)
        # mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.tensor(1.0).to(mask_pos_pairs.device), mask_pos_pairs)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.reshape(anchor_count, batch_size).mean()

        return loss


class SINCERELoss(nn.Module):
    def __init__(self, temperature=0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, embeds: torch.Tensor, labels: torch.tensor):
        """Supervised InfoNCE REvisited loss with cosine distance

        Args:
            embeds (torch.Tensor): (B, D) embeddings of B images normalized over D dimension.
            labels (torch.tensor): (B,) integer class labels.

        Returns:
            torch.Tensor: Scalar loss.
        """
        # calculate logits (activations) for each embeddings pair (B, B)
        # using matrix multiply instead of cosine distance function for ~10x cost reduction
        logits = embeds @ embeds.T
        logits /= self.temperature
        # determine which logits are between embeds of the same label (B, B)
        same_label = labels.unsqueeze(0) == labels.unsqueeze(1)

        # masking with -inf to get zeros in the summation for the softmax denominator
        denom_activations = torch.full_like(logits, float("-inf"))
        denom_activations[~same_label] = logits[~same_label]
        # get logsumexp of the logits between embeds of different labels for each row (B,)
        base_denom_row = torch.logsumexp(denom_activations, dim=0)
        # reshape to be (B, B) with row values equivalent, to be masked later
        base_denom = base_denom_row.unsqueeze(1).repeat((1, len(base_denom_row)))

        # get mask for numerator terms by removing comparisons between an image and itself (B, B)
        in_numer = same_label
        in_numer[torch.eye(in_numer.shape[0], dtype=bool)] = False
        # delete same_label so don't need to copy for in_numer
        del same_label
        # count numerator terms for averaging (B,)
        numer_count = in_numer.sum(dim=0)
        numer_count = torch.where(numer_count == 0, torch.tensor(1, device=numer_count.device), numer_count)

        # numerator activations with others zeroed (B, B)
        numer_logits = torch.zeros_like(logits)
        numer_logits[in_numer] = logits[in_numer]

        # construct denominator term for each numerator via logsumexp over a stack (B, B)
        log_denom = torch.zeros_like(logits)
        log_denom[in_numer] = torch.stack(
            (numer_logits[in_numer], base_denom[in_numer]), dim=0).logsumexp(dim=0)

        # cross entropy loss of each positive pair with the logsumexp of the negative classes (B, B)
        # entries not in numerator set to 0
        ce = -1 * (numer_logits - log_denom)
        # take average over rows with entry count then average over batch
        loss = torch.sum(ce / numer_count) / ce.shape[0]
        return loss


class MultiviewSINCERELoss(SINCERELoss):
    def __init__(self, temperature=0.07) -> None:
        super().__init__(temperature)

    def forward(self, features, labels: torch.tensor):
        """Supervised InfoNCE with cosine distance and multiple image views

        Args:
            embeds (torch.Tensor): (B, V, D) embeddings of V augmented views of B images,
                                   normalized over D dimension.
            labels (torch.tensor): (B,) integer class labels.

        Returns:
            torch.Tensor: Scalar loss.
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if len(features.shape) < 3:
            # raise ValueError('`features` needs to be [bsz, n_views, ...],'
            #                  'at least 3 dimensions are required')
            features = features.unsqueeze(dim=1)
        if len(features.shape) > 3:
            features = features.reshape(features.shape[0], features.shape[1], -1)

        # collapse view dimension, leaving views next to each other
        reshaped_embeds = torch.reshape(features, (-1, features.shape[2]))
        # repeat labels to account for views
        labels = labels.repeat_interleave(features.shape[1])
        return super().forward(reshaped_embeds, labels)


class ClusterLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super().__init__()

        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features1, global_protos=None, mask=None):
        """
        Compute contrastive loss between feature and global prototype
        """
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 先把global_protos从list转为tensor
        global_protos_tensor = torch.stack(global_protos)
        features = torch.stack(features1)

        # 求sim(zi,g)/t
        sim_mat = torch.div(
            torch.matmul(features, global_protos_tensor.T),
            self.temperature)

        s_dist = F.softmax(sim_mat, dim=1)
        cost_mat = self.EuclideanDistances(features, global_protos_tensor)
        loss = (cost_mat * s_dist).sum(1).mean()

        return loss

    # def EuclideanDistances(self, a, b):
    #     sq_a = a**2
    #     sum_sq_a = torch.sum(sq_a,dim=1).unsqueeze(1)  # m->[m, 1]
    #     sq_b = b**2
    #     sum_sq_b = torch.sum(sq_b,dim=1).unsqueeze(0)  # n->[1, n]
    #     bt = b.t()
    #     return torch.sqrt(sum_sq_a+sum_sq_b-2*a.mm(bt))

    def EuclideanDistances(self, a, b):
        sq_a = a ** 2
        sum_sq_a = torch.sum(sq_a, dim=1).unsqueeze(1)  # m->[m, 1]
        sq_b = b ** 2
        sum_sq_b = torch.sum(sq_b, dim=1).unsqueeze(0)  # n->[1, n]
        bt = b.t()
        dist = sum_sq_a + sum_sq_b - 2 * a.mm(bt)

        # 检查 dist 是否有 NaN
        if torch.isnan(dist).any():
            raise ValueError("Distance calculation resulted in NaN values")

        return torch.sqrt(dist.clamp(min=1e-12))  # 加上clamp来避免负数开方
        # return torch.sqrt(sum_sq_a+sum_sq_b-2*a.mm(bt))


class PrototypeLoss(nn.Module):
    def __init__(self):
        super(PrototypeLoss, self).__init__()

    def forward(self, prototypes):
        """
        计算不同类别原型之间的欧氏距离损失

        参数:
        - prototypes: dict, {class_id: prototype_vector}
          各类别的原型向量字典

        返回:
        - loss: Tensor, scalar
          欧氏距离损失
        """
        class_ids = list(prototypes.keys())
        num_classes = len(class_ids)
        loss = 0.0

        # 遍历所有原型对
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                proto_i = prototypes[class_ids[i]]
                proto_j = prototypes[class_ids[j]]
                distance = torch.norm(proto_i - proto_j, p=2)
                loss -= distance

        return loss / num_classes


class DPNLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super().__init__()

        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, global_protos=None, mask=None):
        """
        Compute contrastive loss between feature and global prototype
        """
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 先把global_protos从list转为tensor
        if isinstance(global_protos, list):
            global_protos = torch.stack(global_protos)
        if isinstance(features, list):
            features_tensor = torch.stack(features)
        else:
            features_tensor = features

            # 求sim(zi,g)/t
            ##features(64,84) globalproto(10,84)
            ##sim_mat(64*10) sim_mat 中的每一行代表一个样本与所有全局原型之间的相似度。
        sim_mat = torch.div(
            torch.matmul(features_tensor, global_protos.T),
            self.temperature)

        s_dist = F.softmax(sim_mat, dim=1)
        # cost_mat = self.EuclideanDistances(features_tensor, global_protos)
        # loss = (cost_mat * s_dist).sum(1).mean()

        # return loss

        # 求sim(zi,g)/t

        #         sim_mat = torch.div(
        #             torch.matmul(features_tensor, global_protos.T),
        #             self.temperature)

        #         # 计算 features_tensor 和 global_protos 的 L2 范数
        #         # features_norm = features_tensor / features_tensor.norm(p=2, dim=1, keepdim=True)
        #         # protos_norm = global_protos / global_protos.norm(p=2, dim=1, keepdim=True)
        #         #
        #         # # 计算余弦相似度
        #         # sim_mat_cosine = torch.matmul(features_norm, protos_norm.T)

        sim_mat_noDIV = torch.matmul(features_tensor, global_protos.T)

        beta = 0.1
        sim_mat_exp = torch.exp(-beta * sim_mat_noDIV)

        #         s_dist = F.softmax(sim_mat, dim=1)
        #         # cost_mat = self.EuclideanDistances(features_tensor, global_protos)
        #         # print("马氏")
        #         # dist_mashi = self.mahalanobis_distances(features_tensor, global_protos)
        #         # fuduishu =  -torch.log(s_dist)
        #         # dist_manhadun = self.manhattan_distances(features_tensor, global_protos)
        #         # loss= s_dist.sum(1).mean()

        #         # loss= (fuduishu * s_dist).sum(1).mean()
        #         # loss= (dist_mashi * s_dist).sum(1).mean()
        #         # loss= (dist_manhadun * s_dist).sum(1).mean()
        #         # loss= (cost_mat * s_dist).sum(1).mean()
        #         # loss= (torch.pow(cost_mat,2) * s_dist).sum(1).mean()
        #         # print("1-Sim")
        #         # loss= ((1-sim_mat_noDIV) * s_dist).sum(1).mean()
        #         # loss= ((1-sim_mat) * s_dist).sum(1).mean()
        #         # loss= ((1-sim_mat_cosine) * s_dist).sum(1).mean()
        loss = (sim_mat_exp * s_dist).sum(1).mean()
        #         # loss= (torch.pow((1-sim_mat_noDIV),2) * s_dist).sum(1).mean()

        return loss

    def EuclideanDistances(self, a, b):
        sq_a = a ** 2
        sum_sq_a = torch.sum(sq_a, dim=1).unsqueeze(1)  # m->[m, 1]
        sq_b = b ** 2
        sum_sq_b = torch.sum(sq_b, dim=1).unsqueeze(0)  # n->[1, n]
        bt = b.t()
        dist_sq = sum_sq_a + sum_sq_b - 2 * a.mm(bt)
        dist_sq = torch.clamp(dist_sq, min=0)
        dist = torch.sqrt(dist_sq + 1e-12)
        return dist

    def mahalanobis_distances(self, features, prototypes):
        """
        计算马氏距离。
        features: 输入样本特征
        prototypes: 全局原型
        """
        # 获取协方差矩阵的逆
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cov_matrix = torch.eye(84).to(device)  # 假设特征维度为512
        inv_cov_matrix = torch.inverse(cov_matrix)

        # 计算每个样本与每个原型之间的马氏距离
        diff = features[:, None] - prototypes  # (N, 1, D) - (K, D)
        mahalanobis_dist = torch.sqrt(torch.sum(torch.matmul(diff, inv_cov_matrix) * diff, dim=-1))

        return mahalanobis_dist

    def manhattan_distances(self, features, prototypes):
        """
        计算曼哈顿距离。
        features: 输入样本特征
        prototypes: 全局原型
        """
        # 计算每个样本和每个原型的曼哈顿距离
        distances = torch.sum(torch.abs(features[:, None] - prototypes), dim=-1)
        return distances


class LabeledGlobalConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super().__init__()

        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature



    def forward(self, local_protos=None, global_protos=None, mask=None):
        # 用于存储匹配的键和值
        matching_keys = []
        local_values = []
        global_values = []

        global_protos_dict = {label: proto for label, proto in global_protos}
        # 遍历 local_protos 的键，检查是否存在于 global_protos 中
        for key in local_protos.keys():
            matching_keys.append(key)
            local_values.append(local_protos[key])
            global_values.append(global_protos_dict[key])

        local_values_tensor = torch.stack(local_values)
        global_values_tensor = torch.stack(global_values)

        cost_mat = self.EuclideanDistances(local_values_tensor, global_values_tensor)
        loss = (cost_mat).sum(1).mean()

        return loss

    def EuclideanDistances(self, a, b):
        sq_a = a ** 2
        sum_sq_a = torch.sum(sq_a, dim=1).unsqueeze(1)  # m->[m, 1]
        sq_b = b ** 2
        sum_sq_b = torch.sum(sq_b, dim=1).unsqueeze(0)  # n->[1, n]
        bt = b.t()
        return torch.sqrt(sum_sq_a + sum_sq_b - 2 * a.mm(bt))


class DPNembAndProLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super().__init__()

        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels, local_protos=None, global_protos=None, mask=None):
        """
        Compute contrastive loss between feature and global prototype
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        global_protos_tensor = torch.stack(global_protos)
        feature_local_protos = features
        batch_size = features.shape[0]
        for i in range(batch_size):
            feature_local_protos[i, :] = local_protos.get(labels[i % batch_size].item())

        sim_mat = torch.div(
            torch.matmul(feature_local_protos, global_protos_tensor.T),
            self.temperature)

        s_dist = F.softmax(sim_mat, dim=1)
        cost_mat = self.EuclideanDistances(features, global_protos_tensor)
        loss = (cost_mat * s_dist).sum(1).mean()

        return loss

    def EuclideanDistances(self, a, b):
        sq_a = a ** 2
        sum_sq_a = torch.sum(sq_a, dim=1).unsqueeze(1)  # m->[m, 1]
        sq_b = b ** 2
        sum_sq_b = torch.sum(sq_b, dim=1).unsqueeze(0)  # n->[1, n]
        bt = b.t()
        return torch.sqrt(sum_sq_a + sum_sq_b - 2 * a.mm(bt))


class FedTGPLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super().__init__()

        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, local_protos=None, global_protos=None, mask=None):
        """
        Compute contrastive loss between feature and global prototype
        """
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 先把global_protos从list转为tensor
        global_protos_tensor = torch.stack(global_protos)

        if isinstance(local_protos, list):
            local_protos_tensor = torch.stack(local_protos)
        else:
            local_protos_tensor = local_protos

        # 求sim(zi,g)/t
        sim_mat = torch.div(
            torch.matmul(local_protos_tensor, global_protos_tensor.T),
            self.temperature)

        cost_mat = self.EuclideanDistances(local_protos_tensor, global_protos_tensor)

        numerator = torch.exp(sim_mat - cost_mat)

        # 计算不带 margin 的分母
        denominator = torch.sum(torch.exp(sim_mat), dim=1, keepdim=True)

        # 计算 softmax
        s_dist = numerator / denominator
        loss = s_dist.sum(1).mean()

        return loss

    def EuclideanDistances(self, a, b):
        sq_a = a ** 2
        sum_sq_a = torch.sum(sq_a, dim=1).unsqueeze(1)  # m->[m, 1]
        sq_b = b ** 2
        sum_sq_b = torch.sum(sq_b, dim=1).unsqueeze(0)  # n->[1, n]
        bt = b.t()
        return torch.sqrt(sum_sq_a + sum_sq_b - 2 * a.mm(bt))


class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        # self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss