import torch


def EuclideanDistances(a, b):
    sq_a = a ** 2
    sum_sq_a = torch.sum(sq_a, dim=1).unsqueeze(1)  # m->[m, 1]
    print(sum_sq_a.size())
    print(sum_sq_a)
    sq_b = b ** 2
    sum_sq_b = torch.sum(sq_b, dim=1).unsqueeze(0)  # n->[1, n]
    print(sum_sq_b.size())
    print(sum_sq_b)
    bt = b.t()
    print(bt.size())
    print(bt)
    dist_sq = sum_sq_a + sum_sq_b - 2 * a.mm(bt)
    print(dist_sq.size())
    print(dist_sq)
    dist_sq = torch.clamp(dist_sq, min=0)
    dist = torch.sqrt(dist_sq + 1e-12)
    return dist


# 测试
a = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
], dtype=torch.float32)
2*3

64*84
84*64
b = torch.tensor([
    [1.0, 1.0, 1.0],
    [2.0, 2.0, 2.0],
    [3.0, 3.0, 3.0]
], dtype=torch.float32)

dist_matrix = EuclideanDistances(a, b)
print(dist_matrix)
