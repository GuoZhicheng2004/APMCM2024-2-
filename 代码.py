import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 读取用户上传的 Excel 文件
file_path = 'Q1.xlsx'
data = pd.read_excel(file_path)

# 检查数据结构
data.head()

# 设置显示中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 计算猫和狗的年增长率
data['Cat Growth Rate (%)'] = data['Cat'].pct_change() * 100
data['Dog Growth Rate (%)'] = data['Dog'].pct_change() * 100
data['Bird Growth Rate (%)']= data['Bird'].pct_change() * 100
data['Fish Growth Rate (%)']= data['Fish'].pct_change() * 100

# 创建增长率的可视化
plt.figure(figsize=(12, 6))
plt.plot(data['Year'], data['Cat Growth Rate (%)'], marker='o', label='Cat Growth Rate (%)')
plt.plot(data['Year'], data['Dog Growth Rate (%)'], marker='o', label='Dog Growth Rate (%)')
plt.plot(data['Year'], data['Bird Growth Rate (%)'], marker='o', label='Bird Growth Rate (%)')
plt.plot(data['Year'], data['Fish Growth Rate (%)'], marker='o', label='Fish Growth Rate (%)')

# 设置图表标题和标签
plt.title('The annual growth rate of pet cats,dogs,birds and fish in China from 2019 to 2023', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Growth Rate (%)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.xticks(data['Year'])

# 显示图表
plt.tight_layout()
plt.show()

# 设置显示中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 创建包含猫和狗数量随时间变化的可视化
plt.figure(figsize=(12, 6))
plt.plot(data['Year'], data['Cat'], marker='o', label='Number of pet cats')
plt.plot(data['Year'], data['Dog'], marker='o', label='Number of pet dogs')
plt.plot(data['Year'], data['Bird'], marker='o', label='Number of pet birds')
plt.plot(data['Year'], data['Fish'], marker='o', label='Number of pet fish')

# 设置图表标题和标签
plt.title('Trends in the number of pet cats,dogs,birds and fish in China from 2019 to 2023', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Quantity (10 000)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.xticks(data['Year'])

# 显示图表
plt.tight_layout()
plt.show()

# 创建包含猫、狗、鸟、鱼年增长率的可视化
plt.figure(figsize=(12, 6))
plt.bar(data['Year'], data['Cat Growth Rate (%)'], alpha=0.6, label='Cat Growth Rate (%)', color='blue', edgecolor='black')
plt.bar(data['Year'], data['Dog Growth Rate (%)'], alpha=0.6, label='Dog Growth Rate (%)', color='orange', edgecolor='black')
plt.bar(data['Year'], data['Bird Growth Rate (%)'], alpha=0.6, label='Bird Growth Rate (%)', color='green', edgecolor='black')
plt.bar(data['Year'], data['Fish Growth Rate (%)'], alpha=0.6, label='Fish Growth Rate (%)', color='yellow', edgecolor='black')

# 设置图表标题和标签
plt.title('The annual growth rate of pet cats,dogs,birds and fish in China from 2019 to 2023', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Growth Rate (%)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.xticks(data['Year'])

# 显示图表
plt.tight_layout()
plt.show()

from scipy.stats import shapiro, spearmanr, pearsonr
import seaborn as sns

# 数据准备
variables = ['Economic Growth Rate (%)', 'GDP per capita (RMB)', 'Urbanization Rate (%)', 'Disposable income per capita', 'Cat', 'Dog', 'Bird', 'Fish']
data_subset = data[variables]

# 正态性检验（Shapiro-Wilk检验）
normality_results = {var: shapiro(data_subset[var]) for var in variables}

# 显示正态性检验结果
normality_summary = pd.DataFrame({
    'variable': variables,
    'W-value': [normality_results[var].statistic for var in variables],
    'P-value': [normality_results[var].pvalue for var in variables],
    'Whether it is normally distributed or not': ['yes' if normality_results[var].pvalue > 0.05 else '否' for var in variables]
})

# 根据正态性检验结果，选择相关性分析方法
# 如果所有变量正态分布，使用皮尔逊；否则使用斯皮尔曼
correlation_method = 'pearson' if all(normality_summary['Whether it is normally distributed or not'] == 'yes') else 'spearman'

# 计算相关性矩阵
correlation_matrix = data_subset.corr(method=correlation_method)

# 可视化相关性矩阵
plt.figure(figsize=(6,6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title(f'{correlation_method.capitalize()} Correlation analysis heat map', fontsize=16)
plt.show()

import numpy as np

# 数据提取
years = data['Year'].values
cat_data = data['Cat'].values
dog_data = data['Dog'].values
bird_data = data['Bird'].values
fish_data = data['Fish'].values

# 定义 GM(1,1) 模型
def gm11(x, n_pred):
    x1 = np.cumsum(x)  # 累加序列
    z1 = (x1[:-1] + x1[1:]) / 2.0  # 均值生成序列
    B = np.vstack([-z1, np.ones_like(z1)]).T
    Y = x[1:]
    u = np.linalg.inv(B.T @ B) @ B.T @ Y
    a, b = u[0], u[1]

    # 预测值生成
    def f(k):
        return (x[0] - b / a) * np.exp(-a * k) + b / a

    pred = [f(k) for k in range(len(x) + n_pred)]
    return np.array(pred[:len(x)]), np.array(pred[len(x):])

# 猫和狗的预测
n_pred = 3
cat_actual, cat_pred = gm11(cat_data, n_pred)
dog_actual, dog_pred = gm11(dog_data, n_pred)
bird_actual, bird_pred = gm11(bird_data, n_pred)
fish_actual, fish_pred = gm11(fish_data, n_pred)

# 未来年份
future_years = np.arange(years[-1] + 1, years[-1] + n_pred + 1)

# 可视化
plt.figure(figsize=(12, 6))
plt.plot(years, cat_actual, marker='o', label='Cat actuals', color='blue')
plt.plot(future_years, cat_pred, marker='o', linestyle='--', label='Cat predictions', color='blue', alpha=0.6)
plt.plot(years, dog_actual, marker='o', label='Dog actuals', color='orange')
plt.plot(future_years, dog_pred, marker='o', linestyle='--', label='Dog predictions', color='orange', alpha=0.6)
plt.plot(years, bird_actual, marker='o', label='Bird actuals', color='green')
plt.plot(future_years, bird_pred, marker='o', linestyle='--', label='Bird predictions', color='green', alpha=0.6)
plt.plot(years, fish_actual, marker='o', label='Fish actuals', color='yellow')
plt.plot(future_years, fish_pred, marker='o', linestyle='--', label='Fish predictions', color='yellow', alpha=0.6)
plt.title('Forecast of the number of pet cats, dogs, birds, and fish in China from 2019 to 2026', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Quantity (10 000)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(list(years) + list(future_years))
plt.tight_layout()
plt.show()