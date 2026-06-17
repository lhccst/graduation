import torch.nn as nn

class simpleMLP(nn.Module):
    def __init__(self, input_size, dropout_rate=0.5):
        super(simpleMLP, self).__init__()
        hidden_size = input_size
        output_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)  # 第一层全连接层
        self.fc2 = nn.Linear(hidden_size, output_size)  # 第二层全连接层
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)  # Dropout 层

    def forward(self, x):
        x = self.fc1(x)  # 通过第一层
        x = self.relu(x)  # 通过 ReLU 激活函数
        x = self.dropout(x)  # Dropout
        x = self.fc2(x)  # 通过第二层
        return x
