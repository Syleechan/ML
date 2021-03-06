# softmax与交叉熵损失函数的结合
import tensorflow as tf
import numpy as np

y_ = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]])
y = np.array([[12, 3, 2], [3, 10, 1], [1, 2, 5], [4, 6.5, 1.2], [3, 6, 1]])
y_pred = tf.nn.softmax(y)
E1=tf.losses.categorical_crossentropy(y_,y_pred)
E2 = tf.nn.softmax_cross_entropy_with_logits(y_, y)

print('分步计算的结果:\n', E1)
print('结合计算的结果:\n', E2)
# 输出的E1，E2结果相同