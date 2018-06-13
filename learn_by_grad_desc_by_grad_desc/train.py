import tensorflow as tf



def train_optimizer(N = 10, n_unroll = 20, num_of_coordinates = 3, hidden_size = 5, n_features = 3, num_layers = 2, max_epoch = 20):
    """Implementation of Learning to learn by gradient descent by gradient descent https://arxiv.org/abs/1606.04474

    Args:
        N: number of datapoints i.e samples of loss functions, total loss will be averaged over N.
        n_unroll: number of gradient optimizee steps
        num_of_coordinates: the dimension of optimizee params
        hidden_size: number of hidden units of LSTM
        n_features: number of dimensions in the data space
        num_layers: number of LSTM cells per coordinate


    """
    g = tf.Graph()
    with g.as_default():
        cell_list = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(num_of_coordinates):
                cell_list.append(tf.contrib.rnn.MultiRNNCell
                                 ([tf.contrib.rnn.BasicLSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse)
                                   for _ in range(num_layers)])) #
            loss = 0
            for t in range(N):
                # generate a datapoint random i.e instance of the quadratic function
                W = tf.truncated_normal([n_features, n_features])
                y = tf.truncated_normal([n_features, 1])
                theta = tf.truncated_normal([num_of_coordinates, 1])
                # the optimizee loss function
                f = tf.reduce_sum(tf.square(tf.matmul(W, theta) - y))
                batch_size = 1
                state_list = [cell_list[i].zero_state(batch_size, tf.float32) for i in range(num_of_coordinates)]
                g_new_list = []
                grad_f = tf.gradients(f, theta)[0]
                for i in range(num_of_coordinates):
                    cell = cell_list[i]
                    state = state_list[i]
                    grad_h_t = tf.slice(grad_f, begin=[i, 0], size=[1, 1])
                    for k in range(n_unroll):
                        if k > 0: tf.get_variable_scope().reuse_variables()
                        cell_output, state = cell(grad_h_t, state)  # g_new should be a scalar b/c grad_h_t is a scalar
                        softmax_w = tf.get_variable("softmax_w", [hidden_size, 1])
                        softmax_b = tf.get_variable("softmax_b", [1])
                        g_new_i = tf.matmul(cell_output, softmax_w) + softmax_b
                    g_new_list.append(g_new_i)
                    state_list[i] = state # update LSTM state

                # update parameter
                g_new = tf.reshape(tf.squeeze(tf.stack(g_new_list)),
                                   [n_features, 1])  # should be a [n_features, 1] tensor
                theta = tf.add(theta, g_new)
                f_at_theta_t = tf.reduce_sum(tf.square(tf.matmul(W, theta) - y))
                loss += f_at_theta_t

        loss = loss / N
        tvars = tf.trainable_variables()  # should be just the variable inside the RNN
        grads = tf.gradients(loss, tvars)
        lr = 0.001
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.apply_gradients(zip(grads, tvars))


        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            max_epoch = 100
            for epoch in range(max_epoch):
                cost, _ = sess.run([loss, train_op])
                print("Epoch %d : loss %f" % (epoch, cost))


if __name__ == "__main__":
    train_optimizer()