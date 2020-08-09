"""产生随机点"""

import random
import numpy as np
from matplotlib import pyplot as pt

# 在0-10的区间上生成100个点作为输入数据
X = np.linspace(0, 10, 1000, endpoint=True)
Y = 2.0 * X + 5.0
# 定义gauss噪声的均值和方差
mu = 0.00
sigma = 0.12
# 对输入数据加入gauss噪声
for i in range(X.size):
    X[i] += random.gauss(mu, sigma)
    Y[i] += random.gauss(mu, sigma)
# 读取label.txt文件，没有则创建，‘w’表示再次写入时覆盖之前的内容
file = open("Label.txt", 'w')
for i in range(len(X)):
    print(i, X[i], Y[i])
    file.write(str(X[i]) + "," + str(Y[i]) + '\n')
# 画出这些点
pt.plot(X, Y, linestyle='', marker='.')
pt.show()
