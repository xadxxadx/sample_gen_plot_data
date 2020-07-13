import gen_data as gd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random

class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()    # Flatten层将除第一维（batch_size）以外的维度展平
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=1, activation=None)

    def call(self, inputs):         # [batch_size, 28, 28, 1]
        x = self.flatten(inputs)    # [batch_size, 784]
        x = self.dense1(x)          # [batch_size, 100]
        x = self.dense2(x)          # [batch_size, 1]
        return x

if __name__ == '__main__':
    params = [0.5, 1.2, -0.7, 1]
    x_min = -3
    x_max = 2
    data = gd.gen_2d_3exp_data(params[0], params[1], params[2], params[3], x_min, x_max, 100, 3)
    plt.plot(data[:,0], data[:, 1], label='point')
    plt.plot(data[:, 0], data[:, 2], label='point')
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(data[:,0], data[:, 1])
    fig.suptitle('A Simple Scatter Plot')
    plt.show()

    data_num = 10000
    x = np.zeros((data_num, 1))
    y = np.zeros((data_num, 1))
    for i in range(data_num):
        x_ = x_min + (x_max - x_min) * random.random()
        y_gt_ = params[0]*x_**3+params[1]*x_**2+params[2]*x_+params[3]
        y_ = y_gt_ + (0.5-random.random()) * 3
        x[i, 0] = x_
        y[i, 0] = y_

    num_epochs = 500
    batch_size = 50
    learning_rate = 0.001
    model = MLP()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for epoch in range(num_epochs):
        num_batches = int(data_num // batch_size)
        for batch_index in range(num_batches):
            X = x[batch_index*batch_size:batch_index*batch_size+batch_size, :]
            Y = y[batch_index*batch_size:batch_index*batch_size+batch_size, :]

            with tf.GradientTape() as tape:
                y_pred = model(X)
                loss = tf.keras.losses.mean_absolute_error(y_true=Y, y_pred=y_pred)
                loss = tf.reduce_mean(loss)
                print("batch %d: loss %f" % (batch_index, loss.numpy()))
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    x = np.linspace(x_min, x_max,50)
    y = model(x).numpy()
    x = x.reshape((50,1))
    y = y.reshape((50,1))

    plt.plot(x, y, label='point')
    plt.show()

    pass