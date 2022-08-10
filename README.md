
plt.plot()   # 绘图函数
plt.show()   # 显示图像
在jupyter notebook中不执行这条语句也是可以将图形展示出来

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

np.random.seed(1000)
y = np.random.standard_normal(20)  # 生成正态分布的随机数

x = range(len(y))
plt.plot(x,y)
执行结果：

![image](https://user-images.githubusercontent.com/44520334/183868721-9f004f41-736c-4f1b-8ade-b0c8bb41eee9.png)
