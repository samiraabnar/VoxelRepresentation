import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

def forward_step(X,w_in,w_out):
    hidden_state = tf.nn.sigmoid(tf.matmul(X,w_in))
    output = tf.matmul(hidden_state,w_out)
    return output, hidden_state

def main():
    inputs = np.load("inputs")
    outputs = np.load("outputs")

    number_of_samples = len(inputs)
    input_dim = 21764
    hidden_dim = 500

    all_inputs = np.ones((number_of_samples,input_dim+1))
    all_inputs[:,1:] = inputs
    all_outputs = np.eye(input_dim)[outputs]

    train_X, train_y, test_X, test_Y = train_test_split(all_inputs, all_outputs, test_size=0, random_state=RANDOM_SEED)

    X = tf.placeholder("float", shape=[None, input_dim])
    y = tf.placeholder("float", shape=[None, input_dim])

    W_in = tf.random_normal((input_dim+1,hidden_dim), stddev=0.1)
    W_out = tf.random_normal((hidden_dim,input_dim), stddev=0.1)


    p_output    = forward_step(X, W_in, W_out)
    predict = tf.argmax(p_output, axis=1)

    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=p_output))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(100):
        # Train with each example
        for i in range(len(train_X)):
            sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: train_X, y: train_y}))


        print("Epoch = %d, train accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy))

    sess.close()

if __name__ == '__main__':
    main()






