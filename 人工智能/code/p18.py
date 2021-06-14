import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示error，不显示其他信息

a = tf.zeros([2, 3])
b = tf.ones(1)
c = tf.fill([2, 2], 9)
print("a:", a)
print("b:", b)
print("c:", c)
