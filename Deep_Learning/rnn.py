import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Dropout, CuDNNLSTM
from tensorflow.python.keras.models import Sequential

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train= x_train/255.0
x_test = x_test/255.0
print(x_train.shape)
print(x_train[0].shape)

model = Sequential()

model.add(CuDNNLSTM(128, input_shape=(
    x_train.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))

model.add(CuDNNLSTM(128))
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

optimizer = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
