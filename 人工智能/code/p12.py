import tensorflow as tf
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示error，不显示其他信息

w = tf.Variable(tf.constant(5, dtype=tf.float32))
epoch = 100
lr = 0.05

for epoch in range(epoch):
    with tf.GradientTape() as tape:
        loss = tf.square(w + 1)
    grads = tape.gradient(loss, w)

    w.assign_sub(lr * grads)
    print("After %s steps,w is %f,loss is %f" % (epoch, w.numpy(), loss))
