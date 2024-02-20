# 导入必要的库
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 读取数据并进行预处理
df = pd.read_csv("jyc48-16.csv")
df = df.dropna()
df.iloc[:, 0] = range(len(df))
y = df.loc[:, 'TMIN']
X = df.drop('TMIN', axis=1)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
y_train = y_train. to_numpy()
y_test = y_test. to_numpy()
y_train = y_train . reshape((-1, 1))
y_test = y_test . reshape((-1, 1))
# 数据预处理
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), ['DATE', 'PRCP', 'TMAX']),
    ('cat', OneHotEncoder(), ['RAIN'])
])
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train.astype(np.float32))
y_train_tensor = torch.tensor(y_train.astype(np.float32))
X_test_tensor = torch.tensor(X_test.astype(np.float32))
y_test_tensor = torch.tensor(y_test.astype(np.float32))

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 定义模型
class WeatherModel(nn.Module):
    def __init__(self, input_dim):
        super(WeatherModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x


model = WeatherModel(X_train.shape[1])

# 训练模型
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
epochs = 20

for epoch in range(epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.view(-1, 1))
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# 评估模型
model.eval()
total_loss = 0
actual = []
y_pred_list = []
with torch.no_grad():
    for inputs, targets in test_loader:

        outputs = model(inputs)
        loss = criterion(outputs, targets.view(-1, 1))
        total_loss += loss.item()

        actual.append(targets.tolist())
        y_pred_list.append(outputs.tolist())

print(f'Test Loss: {total_loss / len(test_loader)}')


y_pred_list1 = [item for sublist in y_pred_list for item in sublist]
actual1 = [item for sublist in actual for item in sublist]

plt.figure(figsize=(16, 6))
plt.scatter(range(len(y_pred_list1)), y_pred_list1, color='blue', label='predict')
plt.scatter(range(len(actual1)),actual1, color='red', label='Actual')

plt.xlabel('Index')
plt.ylabel('Gold rate')
plt.legend()

plt.show()