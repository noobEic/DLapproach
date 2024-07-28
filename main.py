import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
# 定义三层线性神经网络
class ThreeLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ThreeLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义ODE：dx/dt = f(t, x)
def ode_function(t, x):
    # 假设的一个函数
    return t * torch.exp(-x)

# 定义损失函数
def loss_function(predicted_x, t, net):
    # 计算dx/dt
    predicted_x_t = net(t)
    dx_dt = torch.autograd.grad(predicted_x, t, torch.ones_like(predicted_x), create_graph=True)[0]
    # 计算损失
    loss = torch.mean((dx_dt - ode_function(t, predicted_x_t)) ** 2)
    return loss

# 初始化网络和优化器
input_size = 1  # 输入大小（时间维度）
hidden_size = 10  # 隐藏层大小
output_size = 1  # 输出大小（x的维度）
net = ThreeLayerNet(input_size, hidden_size, output_size)
optimizer = optim.Adam(net.parameters(), lr=0.01)

# 训练网络
n_epochs = 1000
t_values = torch.linspace(0, 1, 100).view(-1, 1).requires_grad_(True)  # 时间点
initial_x = torch.zeros(1, 1)                    # 初始条件

for epoch in range(n_epochs):
    optimizer.zero_grad()
    x_values = net(t_values)
    loss = loss_function(x_values, t_values, net)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# 测试网络
with torch.no_grad():
    t_test = torch.linspace(0, 1, 100).view(-1, 1)
    x_test = net(t_test)
    print(x_test)

# 可视化结果

figure = plt.figure
plt.plot(t_test.numpy(), x_test.numpy(), label='Neural Network Solution')
plt.xlabel('t')
plt.ylabel('x')
plt.legend()
plt.show()
