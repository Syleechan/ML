import tensorflow as tf
import os
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

cifar10=tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)  # 给数据增加一个维度，使数据和网络结构匹配
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)

x_train, x_test = x_train / 255.0, x_test / 255.0

image_gen_train = ImageDataGenerator(
    rescale=1,
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    horizontal_flip=True,
    zoom_range=1
)
image_gen_train.fit(x_train)


class Model_VGG(Model):
    def __init__(self):
        super(Model_VGG, self).__init__()

        self.c1 = Conv2D(input_shape=(32, 32, 3), kernel_size=(3, 3), filters=64,
                         strides=1, padding='same')

        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.c2 = Conv2D(64, kernel_size=(3, 3), strides=1, padding='same')
        self.b2 = BatchNormalization()
        self.a2 = Activation('relu')
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d1 = Dropout(0.2)

        self.c3 = Conv2D(128, kernel_size=(3, 3), strides=1, padding='same')
        self.b3 = BatchNormalization()
        self.a3 = Activation('relu')
        self.c4 = Conv2D(128, kernel_size=(3, 3), strides=1, padding='same')
        self.b4 = BatchNormalization()
        self.a4 = Activation('relu')
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d2 = Dropout(0.2)

        self.c5 = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same')
        self.b5 = BatchNormalization()
        self.a5 = Activation('relu')
        self.c6 = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same')
        self.b6 = BatchNormalization()
        self.a6 = Activation('relu')
        self.c7 = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same')
        self.b7 = BatchNormalization()
        self.a7 = Activation('relu')
        self.p3 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d3 = Dropout(0.2)

        self.c8 = Conv2D(512, kernel_size=(3, 3), strides=1, padding='same')
        self.b8 = BatchNormalization()
        self.a8 = Activation('relu')
        self.c9 = Conv2D(512, kernel_size=(3, 3), strides=1, padding='same')
        self.b9 = BatchNormalization()
        self.a9 = Activation('relu')
        self.c10 = Conv2D(512, kernel_size=(3, 3), strides=1, padding='same')
        self.b10 = BatchNormalization()
        self.a10 = Activation('relu')
        self.p4 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d4 = Dropout(0.2)

        self.c11 = Conv2D(512, kernel_size=(3, 3), strides=1, padding='same')
        self.b11 = BatchNormalization()
        self.a11 = Activation('relu')
        self.c12 = Conv2D(512, kernel_size=(3, 3), strides=1, padding='same')
        self.b12 = BatchNormalization()
        self.a12 = Activation('relu')
        self.c13 = Conv2D(512, kernel_size=(3, 3), strides=1, padding='same')
        self.b13 = BatchNormalization()
        self.a13 = Activation('relu')
        self.p5 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d5 = Dropout(0.2)

        self.flatten = Flatten()
        self.f1 = Dense(4096, activation='relu')
        self.d6 = Dropout(0.2)
        self.f2 = Dense(512, activation='relu')
        self.d7 = Dropout(0.2)
        self.f3 = Dense(10, activation='softmax')

    def call(self, x):
        a = self.c1(x)
        b = self.b1(a)
        c = self.a1(b)
        d = self.c2(c)
        e = self.b2(d)
        f = self.a2(e)
        g = self.p1(f)
        h = self.d1(g)

        i = self.c3(h)
        j = self.b3(i)
        k = self.a3(j)
        l = self.c4(k)
        m = self.b4(l)
        n = self.a4(m)
        o = self.p2(n)
        p = self.d2(o)

        q = self.c5(p)
        r = self.b5(q)
        s = self.a5(r)
        t = self.c6(s)
        u = self.b6(t)
        v = self.a6(u)
        w = self.c7(v)
        x = self.b7(w)
        y = self.a7(x)
        z = self.p3(y)
        a = self.d3(z)

        b = self.c8(a)
        c = self.b8(b)
        d = self.a8(c)
        e = self.c9(d)
        f = self.b9(e)
        g = self.a9(f)
        h = self.c10(g)
        i = self.b10(h)
        j = self.a10(i)
        k = self.p4(j)
        l = self.d4(k)

        m = self.c11(l)
        n = self.b11(m)
        o = self.a11(n)
        p = self.c12(o)
        q = self.b12(p)
        r = self.a12(q)
        s = self.c13(r)
        t = self.b13(s)
        u = self.a13(t)
        v = self.p5(u)
        w = self.d5(v)

        x = self.flatten(w)
        y = self.f1(x)
        z = self.d6(y)
        a = self.f2(z)
        b = self.d7(a)
        res = self.f3(b)

        return res


model = Model_VGG()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "C:/Users/47826/Desktop/rengong/checkpoint/VGG.ckpt"

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 # monitor='loss',
                                                 # save_best_only=True,
                                                 verbose=2)
#
history = model.fit(x_train, y_train, batch_size=64, epochs=50,
                    validation_data=(x_test, y_test),
                    validation_freq=1, callbacks=[cp_callback], verbose=1)

model.summary()

file = open('C:/Users/47826/Desktop/rengong/weights.txt', 'w')  # 参数提取
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

###############################################    show   ###############################################

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()