import tensorflow as tf

#comments 123

mnist_dataset = tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test) = mnist_dataset.load_data()

import matplotlib.pyplot as plt

#plt.imshow(x_train[2])
#plt.show()

mymodel = tf.keras.models.Sequential()

mymodel.add(tf.keras.layers.Flatten())
mymodel.add(tf.keras.layers.Dense(20,activation=tf.nn.relu))
mymodel.add(tf.keras.layers.Dense(20,activation=tf.nn.relu))
mymodel.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

mymodel.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

mymodel.fit(x_train,y_train,epochs=3)

pred = mymodel.predict(x_test)

import numpy as np

print(np.argmax(pred[0]))

#comments to follow 12345

