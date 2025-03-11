import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import os
import matplotlib.pyplot as plt
# 参数设置
n_steps = 60        # 输入序列长度（时间步长）
forecast_window = 20  # 预测波动率的计算窗口
n_features = 1       # 输入特征数（仅使用收益率）
epochs = 100         # 训练轮数
batch_size = 32      # 批大小

# 读取所有股票数据并预处理
all_returns = []
stock_symbols = ['AAPL', 'AMGN', 'AMZN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'GS', 'HD', 'HON', 'IBM', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'NVDA', 'PG', 'SHW', 'TRV', 'UNH', 'V', 'VZ', 'WMT']
stock_paths = [os.path.join('道琼斯30', symbol, f'{symbol}.csv') for symbol in stock_symbols]

# 第一阶段：收集所有收益率数据用于计算相关系数矩阵
for path in stock_paths:
    data = pd.read_csv(path)
    returns = np.log(data['Close']).diff().dropna()  # 对数收益率
    all_returns.append(returns)
    
# 创建全市场收益率DataFrame并计算相关系数矩阵
df_returns = pd.concat(all_returns, axis=1)
df_returns.columns = [f'stock_{i}' for i in range(30)]
corr_matrix = df_returns.corr().values

# 第二阶段：逐个股票训练LSTM模型
pred_vols = []  # 存储各股票预测波动率
GT_vols = []  # 存储各股票预测波动率

for stock_idx in range(30):
    # 读取数据
    data = pd.read_csv(stock_paths[stock_idx])
    prices = data['Close'].values
    
    # 计算收益率和波动率
    returns = pd.Series(np.log(prices)).diff().dropna()

    vol = returns.rolling(forecast_window).std().shift(1).dropna()  # 历史波动率
    
    # 创建序列数据
    X, y = [], []
    for i in range(n_steps, len(returns)-forecast_window+1):
        X.append(returns[i-n_steps:i])
        y.append(vol[i+forecast_window-1])  # 对齐未来波动率
    
    X = np.array(X)
    y = np.array(y)
    
    # 划分训练测试集
    split = int(0.8*len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 归一化
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train = scaler_X.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    X_test = scaler_X.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
    
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test = scaler_y.transform(y_test.reshape(-1, 1))
    
    # 调整输入形状
    X_train = X_train.reshape(*X_train.shape, n_features)
    X_test = X_test.reshape(*X_test.shape, n_features)
    
    # 创建LSTM模型
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(32))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # 预测并反归一化
    y_pred = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred).flatten()
    pred_vols.append(y_pred)
    GT_vols.append(scaler_y.inverse_transform(y_test).flatten())

# 第三阶段：计算组合波动率
min_length = min(len(p) for p in pred_vols)  # 统一预测长度
# weights = np.ones(30)/30  # 等权重配置
weights = np.array([
    0.0842, 0.0000, 0.0900, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0557,
    0.0171, 0.0000, 0.0000, 0.0000, 0.0000, 0.3110, 0.0000, 0.0000, 0.1626, 0.0199,
    0.0037, 0.0000, 0.0000, 0.0000, 0.0102, 0.0000, 0.0784, 0.0000, 0.0000, 0.1671
])
portfolio_vols = []
GT_vols_all=[]
for t in range(min_length):
    # 获取当前时刻各股票波动率
    vols = np.array([pred_vols[i][t] for i in range(30)])
    
    # 构建协方差矩阵
    cov_matrix = np.outer(vols, vols) * corr_matrix
    
    # 计算组合方差
    port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_vols.append(np.sqrt(port_var))

# 计算真实组合波动率
GT_vols_all = []
min_length_GT = min(len(g) for g in GT_vols)  # 统一真实波动率长度

for t in range(min_length_GT):
    # 获取当前时刻各股票的真实波动率
    vols_GT = np.array([GT_vols[i][t] for i in range(30)])

    # 构建协方差矩阵（真实波动率）
    cov_matrix_GT = np.outer(vols_GT, vols_GT) * corr_matrix

    # 计算组合方差
    port_var_GT = np.dot(weights.T, np.dot(cov_matrix_GT, weights))
    GT_vols_all.append(np.sqrt(port_var_GT))

# 确保两者长度对齐（取最短长度）
min_length_final = min(len(portfolio_vols), len(GT_vols_all))
portfolio_vols = portfolio_vols[:min_length_final]
GT_vols_all = GT_vols_all[:min_length_final]

# 可视化真实 vs 预测组合波动率
plt.figure(figsize=(12,6))
plt.plot(GT_vols_all, label='True Portfolio Volatility', color='blue')
plt.plot(portfolio_vols, label='Predicted Portfolio Volatility', color='red', linestyle='dashed')
plt.title('True vs Predicted Portfolio Volatility')
plt.xlabel('Time Steps')
plt.ylabel('Volatility')
plt.legend()
plt.show()