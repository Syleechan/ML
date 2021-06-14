import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from glob import glob
from keras.preprocessing.image import ImageDataGenerator
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.set_printoptions(threshold=float('inf'))
model_save_path = './checkpoint/mnist.tf'
load_pretrain_model = False

train_path = './cifar-10/train/'
test_path = './cifar-10/test/'
text_label = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
def generateds(path):
    x_, y_ = [], []
    img_label = 0
    for content in text_label:
        filename = glob(path + content + '/*jpg')
        for name in filename:
            img = Image.open(name)
            img = np.array(img)
            img = img/255
            x_.append(img)
            y_.append(img_label)
        img_label = img_label + 1
    x_ = np.array(x_)
    y_ = np.array(y_)
    y_ = y_.astype(np.int64)
    return x_, y_

x_test, y_test = generateds(test_path)
x_train, y_train = generateds(train_path)

np.random.seed(57)
np.random.shuffle(x_train)
np.random.seed(57)
np.random.shuffle(y_train)

np.random.seed(57)
np.random.shuffle(x_test)
np.random.seed(57)
np.random.shuffle(y_test)

def Polt(loss, val_loss, val_accuracy, accuracy):
    plt.subplot(121)
    plt.title('loss curve')
    plt.xlabel('epoch')
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.subplot(122)
    plt.title('accuracy curve')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.plot(val_accuracy, label="Validation Acccuracy")
    plt.plot(accuracy, label="Training Acccuracy")
    plt.legend()
    plt.show()

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.c1 = tf.keras.layers.Conv2D(input_shape=(32, 32, 3), filters=32, kernel_size=[5, 5], padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.ac1 = tf.keras.layers.Activation('relu')
        self.s1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.drop1 = tf.keras.layers.Dropout(0.2)

        self.c2 = tf.keras.layers.Conv2D(64, kernel_size=[5, 5], padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.ac2 = tf.keras.layers.Activation('relu')
        self.s2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.drop2 = tf.keras.layers.Dropout(0.2)

        self.flatten = tf.keras.layers.Flatten()
        self.f1 = tf.keras.layers.Dense(512, activation='relu')
        self.drop3 = tf.keras.layers.Dropout(0.2)
        self.d1 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, l):
        x = self.c1(l)
        x = self.bn1(x)
        x = self.ac1(x)
        x = self.s1(x)
        x = self.drop1(x)

        x = self.c2(x)
        x = self.bn2(x)
        x = self.ac2(x)
        x = self.s2(x)
        x = self.drop2(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.drop3(x)
        y = self.d1(x)
        return y

print('-------load the data---------')

image_gen_train = ImageDataGenerator(
                                      rotation_range=55,
                                      width_shift_range=.15,
                                      height_shift_range=.15,
                                      horizontal_flip=True,
                                      zoom_range=0.5
)
image_gen_train.fit(x_train)
# model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(input_shape=(28, 28, 1), filters=32, kernel_size=(5, 5), padding='same'),
#                                     tf.keras.layers.BatchNormalization(),
#                                     tf.keras.layers.Activation('relu'),
#                                     tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
#                                     tf.keras.layers.Dropout(0.2),
#
#                                     tf.keras.layers.Conv2D(64, kernel_size=(5, 5), padding='same'),
#                                     tf.keras.layers.BatchNormalization(),
#                                     tf.keras.layers.Activation('relu'),
#                                     tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
#                                     tf.keras.layers.Dropout(0.2),
#
#                                     tf.keras.layers.Flatten(),
#                                     tf.keras.layers.Dense(512, activation='relu'),
#                                     tf.keras.layers.Dropout(0.2),
#                                     tf.keras.layers.Dense(10, activation='softmax')
# ])
model = MyModel()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

if load_pretrain_model:
    print('-----------load the model ----------')
    model.load_weights(model_save_path)

for i in range(1):
    history = model.fit(image_gen_train.flow(x_train, y_train,  batch_size=32), epochs=100, validation_data=(x_test, y_test), validation_freq=1)
    model.save_weights(model_save_path, save_format='tf')

model.summary()

#Polt(history.history['loss'], history.history['val_loss'], history.history['val_sparse_categorical_accuracy'], history.history['sparse_categorical_accuracy'])
loss = history.history['loss']
val_loss = history.history['val_loss']

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']

plt.subplot(121)
plt.title('loss curve')
plt.xlabel('epoch')
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label='Validation Loss')
plt.legend()

plt.subplot(122)
plt.title('accuracy curve')
plt.xlabel('epoch')
plt.plot(acc, label="Training Acccuracy")
plt.plot(val_acc, label="Validation Acccuracy")
plt.legend()
plt.show()