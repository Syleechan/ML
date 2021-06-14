import tensorflow as tf
import numpy as np

a = np.arange(0, 5)
aa = tf.convert_to_tensor(a, dtype=tf.int32)
print("a:", a)
print("aa:", aa)
