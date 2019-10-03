import tensorflow as tf
import numpy as np
import random

def isEven(x):
    return tf.keras.utils.to_categorical(x % 2, 2)

def lprint(x):
    print(list(x))

def convert_to_binary(x):
  a = format(x, '032b')
  l = list(str(a))
  l = np.array(list(map(int, l)))
  return l

def generate_data(size, train_rate):
    test_rate = 1 - train_rate
    train_size = round(size * train_rate)
    test_size = round(size * test_rate)

    train_data = range(0, train_size)
    train_label = np.array(list(map(isEven, train_data)))
    train_data = np.array(list(map(convert_to_binary, train_data)))

    test_data = range(train_size, size)
    test_label = np.array(list(map(isEven, test_data)))
    test_data = np.array(list(map(convert_to_binary, test_data)))
    return train_data, train_label, test_data, test_label

x_train, y_train, x_test, y_test = generate_data(100000, 0.7)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, activation=tf.nn.relu, input_dim=32),
    tf.keras.layers.Dense(units=2, activation=tf.nn.sigmoid)
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=100)
model.evaluate(x_test, y_test)
x_raw_predictions = [random.randint(0, 10000) for i in range(10)]
x_predictions = np.array(list(map(convert_to_binary, x_raw_predictions)))
predictions = model.predict(x_predictions)
print(predictions)
y_predictions = map(np.argmax, predictions)
lprint(zip(x_raw_predictions, y_predictions))
