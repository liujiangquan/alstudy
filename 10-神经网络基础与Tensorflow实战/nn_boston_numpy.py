import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 读取数据
def load_data(path):
    data = np.loadtxt(path)
    X = data[:, :-1]
    y = data[:, -1:]
    # 归一化
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y_mean = y.mean()
    y_std = y.std()
    y_norm = (y - y_mean) / y_std
    return X, y, y_norm, y_mean, y_std

# 激活函数和其导数
def relu(x):
    return np.maximum(0, x)
def relu_deriv(x):
    return (x > 0).astype(float)

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mse_loss_grad(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.shape[0]

class SimpleNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def backward(self, X, y, y_pred, lr=0.01):
        m = X.shape[0]
        dz2 = mse_loss_grad(y_pred, y)
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_deriv(self.z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        # 更新参数
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def fit(self, X, y, epochs=500, lr=0.01):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = mse_loss(y_pred, y)
            self.backward(X, y, y_pred, lr)
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    def predict(self, X):
        return self.forward(X)

if __name__ == "__main__":
    X, y, y_norm, y_mean, y_std = load_data("/home/breeze/Documents/workspace/zhihui/code/data/housing.csv")
    nn = SimpleNN(input_dim=X.shape[1], hidden_dim=16, output_dim=1)
    nn.fit(X, y_norm, epochs=500, lr=0.01)
    y_pred_norm = nn.predict(X)
    # 反归一化
    y_pred = y_pred_norm * y_std + y_mean
    # 可视化
    plt.figure(figsize=(8,5))
    plt.scatter(range(len(y)), y, label='实际房价', alpha=0.7)
    plt.scatter(range(len(y_pred)), y_pred, label='预测房价', alpha=0.7)
    plt.xlabel('样本编号')
    plt.ylabel('房价')
    plt.legend()
    plt.title('实际房价 vs 预测房价')
    plt.tight_layout()
    plt.savefig('boston_pred_vs_actual.png')
    print("图像已保存为 boston_pred_vs_actual.png")
    # plt.show()  # 如在支持GUI的环境下可取消注释
    print("Final MSE:", mse_loss(y_pred, y))
