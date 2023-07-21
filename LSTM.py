import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 读取数据
# 假设你的DataFrame名为stock_returns_df
# 请确保stock_returns_df中每一列是一只股票的每日收益率，每行代表一个时间点

# 假设这里是你的读取数据的代码
stock_df = pd.read_csv('/Users/huanggm/Desktop/Quant/data/astocks_market.csv')[['td', 'codenum', 'close']]
stock_df = stock_df[stock_df['td'] >= 20101231]
stock_df['return'] = stock_df.groupby('codenum')['close'].pct_change()
stock_returns_df = pd.pivot_table(stock_df, index='td', columns='codenum', values='return').fillna(0)

# 转换为NumPy数组
stock_returns = stock_returns_df.values

# 划分训练集和测试集
# 假设你希望使用前80%的数据作为训练集，后20%的数据作为测试集
train_size = int(len(stock_returns) * 0.8)
train_data = stock_returns[:train_size]
test_data = stock_returns[train_size:]

# 数据标准化
mean = np.mean(train_data, axis=0)
std = np.std(train_data, axis=0)
# 删除std = 0的列
std = np.where(std == 0, 1, std)
print(np.where(std == 0))
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

# 将数据转换为PyTorch张量
train_data = torch.FloatTensor(train_data)
test_data = torch.FloatTensor(test_data)


# 创建输入序列和目标序列
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data[i + seq_length:i + seq_length + 1]
        sequences.append((seq, target))
    return sequences


# 设置序列长度
sequence_length = 30  # 可根据需要调整

# 创建训练集和测试集的序列
train_sequences = create_sequences(train_data, sequence_length)
test_sequences = create_sequences(test_data, sequence_length)


# 创建LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)  # 输出维度与输入维度一致

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# 设置模型参数
input_size = stock_returns.shape[1]  # 输入维度等于股票数量
hidden_size = 64
num_layers = 1

# 创建LSTM模型实例
model = LSTMModel(input_size, hidden_size, num_layers)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
'''
# 训练模型
num_epochs = 100  # 可根据需要调整
train_losses = []
for epoch in range(num_epochs):
    for seq, target in train_sequences:
        optimizer.zero_grad()
        output = model(seq.unsqueeze(0))
        loss = criterion(output, target)

        # 正则化
        l2_reg = torch.tensor(0.)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss += 0.001 * l2_reg

        loss.backward()
        optimizer.step()
        # 存储损失值
        train_losses.append(loss.item())

#     if (epoch + 1) % 10 == 0:
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    # 保存模型
    torch.save(model.state_dict(), f'lstm_{epoch+1}.pth')

print('Finished Training')
# 保存模型
torch.save(model.state_dict(), 'lstm.pth')

# 可视化损失值
plt.figure(figsize=(12, 6))
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
'''
model.load_state_dict(torch.load('lstm_60.pth'))
# 在测试集上进行预测
model.eval()
with torch.no_grad():
    test_predictions = []
    for seq, target in test_sequences:
        output = model(seq.unsqueeze(0))
        test_predictions.append(output)

# 将预测结果转换为NumPy数组，并还原标准化
test_predictions = torch.cat(test_predictions, dim=0).numpy()
test_predictions = (test_predictions * std) + mean

# 可视化预测结果
plt.figure(figsize=(12, 6))
plt.plot(test_data[:, 0], label='True')
plt.plot(range(sequence_length, len(test_data)), test_predictions[:, 0], label='Predicted')
plt.xlabel('Time')
plt.ylabel('Returns')
plt.legend()
plt.show()
