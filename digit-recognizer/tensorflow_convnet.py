import numpy as np
import tensorflow as tf
from mlxtend.preprocessing import one_hot
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from util import load_train_data, load_test_data, save_predictions


def weight_variable(shape):
    initial = tf.truncated_normal(shape, mean=0.0, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def main():
    # Placeholders

    images = tf.placeholder(tf.float32, [None, 28, 28])
    targets = tf.placeholder(tf.int32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

    # Weights

    W_conv1 = weight_variable([3, 3, 1, 16])
    b_conv1 = bias_variable([16])

    W_conv2 = weight_variable([3, 3, 16, 32])
    b_conv2 = bias_variable([32])

    hidden_units = (7 * 7 * 32 + 10) // 2
    W_hidden = weight_variable([7 * 7 * 32, hidden_units])
    b_hidden = bias_variable([hidden_units])

    W_output = weight_variable([hidden_units, 10])
    b_output = bias_variable([10])

    weights = [
        W_conv1, b_conv1,
        W_conv2, b_conv2,
        W_hidden, b_hidden,
        W_output, b_output,
    ]

    # Forward

    x = tf.reshape(images, [-1, 28, 28, 1])

    x = max_pool(tf.nn.relu(conv2d(x, W_conv1) + b_conv1))
    x = max_pool(tf.nn.relu(conv2d(x, W_conv2) + b_conv2))
    x = tf.reshape(x, [-1, 7 * 7 * 32])

    x = tf.nn.dropout(x, keep_prob)
    x = tf.nn.relu(tf.matmul(x, W_hidden) + b_hidden)

    x = tf.nn.dropout(x, keep_prob)
    outputs = tf.matmul(x, W_output) + b_output

    # Loss

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=targets))
    optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)

    # Accuracy

    correct = tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    with tf.Session() as sess:
        batch_size = 64

        # Training
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(weights, max_to_keep=1)

        X, y = load_train_data()
        y = one_hot(y)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

        best_val_acc = -1
        patience_count = 0

        for epoch in range(1, 1001):
            X_train, y_train = shuffle(X_train, y_train)
            X_batches = np.array_split(X_train, X_train.shape[0] // batch_size)
            y_batches = np.array_split(y_train, y_train.shape[0] // batch_size)
            loss_sum = acc_sum = 0.0
            for X_batch, y_batch in zip(X_batches, y_batches):
                loss_batch, acc_batch, _ = sess.run(
                    [loss, accuracy, optimizer],
                    feed_dict={images: X_batch, targets: y_batch, keep_prob: 0.5})
                loss_sum += loss_batch * X_batch.shape[0]
                acc_sum += acc_batch * X_batch.shape[0]
            acc = acc_sum / X.shape[0]

            X_batches = np.array_split(X_val, X_val.shape[0] // batch_size)
            y_batches = np.array_split(y_val, y_val.shape[0] // batch_size)
            acc_sum = 0.0
            for X_batch, y_batch in zip(X_batches, y_batches):
                acc_batch = sess.run(accuracy, feed_dict={images: X_batch, targets: y_batch, keep_prob: 1.0})
                acc_sum += acc_batch * X_batch.shape[0]
            val_acc = acc_sum / X_val.shape[0]
            patience_count += 1
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_count = 0
                saver.save(sess, 'tensorflow_convnet')

            msg = 'Epoch {:04d} - loss: {:.6g} - acc: {:.6g} - val_acc: {:.6g}'
            print(msg.format(epoch, loss_sum / X.shape[0], acc, val_acc))
            if patience_count > 3:
                break

        # Prediction
        saver.restore(sess, 'tensorflow_convnet')
        X = load_test_data()
        X_batches = np.array_split(X, X.shape[0] // batch_size)
        labels = []
        for X_batch in X_batches:
            y = sess.run(outputs, feed_dict={images: X_batch, keep_prob: 1.0})
            labels.extend(np.argmax(y, 1))
        save_predictions(np.array(labels), 'tensorflow_convnet.csv')


if __name__ == '__main__':
    main()
