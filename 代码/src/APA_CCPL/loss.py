import torch
import torch.nn as nn


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
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")

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
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        """ 损失公式中的最晚层求和，然后除自身的正样本数 """
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_pos_pairs

        # loss
        """ 取反 """
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        """ 对 batch 中的所有样本的损失求均值 """
        loss = loss.reshape(anchor_count, batch_size).mean()

        return loss


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
        device = torch.device("cpu")

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

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mask_pos_pairs = mask.sum(dim=1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.reshape(anchor_count, batch_size).mean()

        return loss
