# !/usr/bin/env python
# coding:utf-8
# Author: Caojian
# p33
import numpy as np

x = np.array([[1, 2, 3],
              [4, 5, 6]],
             dtype=np.float32)
y = np.zeros((3, 4))

print("x:\n", x, "\ny:\n", y)

###################################
# p34
x1 = np.linspace(1, 6, 12) #从1开始 每排6个 共12个
x2 = np.arange(11, 17) #前闭后开
x3 = x1.reshape(2, 6) #2行6列
x4 = np.argsort(x3, axis=0)  #axis为0 纵向
x5 = np.argsort(x3, axis=1)  #axis为1 横向
xx, yy = np.mgrid[-3:3:.01, -3:3:.01]  #从-3到3每次加0.01
print("x1:\n", x1, "\nx2:\n", x2, "\nx3:\n", x3, "\nx4:\n", x4, "\nx5:\n", x5, "\nxx\n", xx, "\nyy:\n", yy)
###################################
# p35
x6 = x * x  #
x7 = np.multiply(x, x)
x8 = x.T
x9 = np.dot(x, x.T)
x10 = np.power(x, 2)
print("x6:\n", x6, "\nx7:\n", x7, "\nx8:\n", x8, "\nx9:\n", x9, "\nx10:\n", x10)
###################################
# p36
x11 = np.array([[1, 2, 3], [4, 5, 6]])
x12 = np.array([[11, 12, 13], [14, 15, 16]])
y1 = np.r_[x11, x12]
y11 = np.vstack((x11, x12))
y2 = np.c_[x11, x12]
y22 = np.hstack((x11, x12))
print("x11:\n", x11, "\nx12:\n", x12, "\ny1:\n", y1, "\ny11:\n", y11, "\ny2:\n", y2, "\ny22:\n", y22)
###################################
# p37
np.save("outfile.npy", x1)
z1 = np.load("outfile.npy")
print("x1:\n", x1, "\nz1:\n", z1)
