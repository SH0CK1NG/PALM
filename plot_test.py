import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
data1 = np.random.normal(0, 1, 200)
data2 = np.random.normal(2, 1, 200)

sns.set_style("white")        # 设置白色网格背景
sns.set_palette("Set2")           # 使用深色调色板

sns.kdeplot(data1, fill=True, label="Group 1")
sns.kdeplot(data2, fill=True, label="Group 2")
plt.legend()
plt.savefig('test.png', dpi=200)
