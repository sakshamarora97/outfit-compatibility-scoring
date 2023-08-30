import tensorflow as tf
import numpy as np
from tensorflow.compat.v1.nn.rnn_cell import GRUCell

tf.to_float = lambda x: tf.cast(x, tf.float32)

######################model##########################
def weights(name, hidden_size, i):
    hidden_stdv = np.sqrt(1. / (hidden_size))
    if name == 'in_image':
        w = tf.compat.v1.get_variable(name='w/in_image_' + str(i),
                            shape=[2048, hidden_size],
                            initializer=tf.random_normal_initializer(stddev=hidden_stdv))
    if name == 'out_image':
        w = tf.compat.v1.get_variable(name='w/out_image_' + str(i),
                            shape=[hidden_size, hidden_size],
                            initializer=tf.random_normal_initializer(stddev=hidden_stdv))
    if name == 'out_in_image':
        w = tf.compat.v1.get_variable(name='w/out_in_image_' + str(i),
                            shape=[hidden_size, hidden_size],
                            initializer=tf.random_normal_initializer(stddev=hidden_stdv))
    if name == 'image_hidden_state_out':
        w = tf.compat.v1.get_variable(name='w/image_hidden_state_out' + str(i),
                            shape=[2 * hidden_size, 2 * hidden_size],
                            initializer=tf.random_normal_initializer(stddev=hidden_stdv))
    if name == 'image_hidden_state_in':
        w = tf.compat.v1.get_variable(name='w/image_hidden_state_in_' + str(i),
                            shape=[2 * hidden_size, 2 * hidden_size],
                            initializer=tf.random_normal_initializer(stddev=hidden_stdv))

    return w


def biases(name, hidden_size, i):
    hidden_stdv = np.sqrt(1. / (hidden_size))
    if name == 'image_hidden_state_out':
        b = tf.compat.v1.get_variable(name='b/image_hidden_state_out' + str(i), shape=[2 * hidden_size],
                            initializer=tf.random_normal_initializer(stddev=hidden_stdv))
    if name == 'image_hidden_state_in':
        b = tf.compat.v1.get_variable(name='b/image_hidden_state_in' + str(i), shape=[2 * hidden_size],
                            initializer=tf.random_normal_initializer(stddev=hidden_stdv))
    if name == 'out_image':
        b = tf.compat.v1.get_variable(name='b/out_image_' + str(i), shape=[hidden_size],
                            initializer=tf.random_normal_initializer(stddev=hidden_stdv))
    if name == 'out_in_image':
        b = tf.compat.v1.get_variable(name='b/out_in_image_' + str(i), shape=[hidden_size],
                            initializer=tf.random_normal_initializer(stddev=hidden_stdv))
    if name == 'in_image':
        b = tf.compat.v1.get_variable(name='b/in_image_' + str(i), shape=[hidden_size],
                            initializer=tf.random_normal_initializer(stddev=hidden_stdv))

    return b


def Laplacian(feature, num_category, category_w, hidden_size):
    index = tf.where(tf.not_equal(tf.reduce_sum(feature, 1), 0))
    category_w = tf.reshape(tf.gather_nd(category_w, index), [-1, hidden_size])
    feature = tf.reshape(tf.gather_nd(feature, index), [-1, hidden_size])
    outfit_feature = tf.concat([category_w, feature], 1)
    outfit_feature_t = tf.transpose(outfit_feature, (1, 0))

    outfit_sim = tf.matmul(outfit_feature, outfit_feature_t)
    outfit_upper_trg = tf.linalg.band_part(outfit_sim, -1, 0)
    outfit_diag = tf.compat.v1.diag_part(outfit_sim)
    outfit_diag = tf.compat.v1.matrix_diag(outfit_diag)
    outfit_similar = outfit_upper_trg - outfit_diag

    outfit_index = tf.where(tf.not_equal(outfit_similar, 0))
    sim_min = tf.reduce_min(tf.gather_nd(outfit_similar, outfit_index))
    indices = tf.reshape(tf.where(tf.equal(outfit_similar, sim_min)), [-1, 1])

    key_A_indices = tf.reshape(indices[0], [1])
    key_B_indices = tf.reshape(indices[1], [1])

    E = tf.reshape(index, [-1])
    key_A = tf.slice(E, key_A_indices, [1])
    key_B = tf.slice(E, key_B_indices, [1])

    E_key_pair = tf.reshape(tf.concat([key_A, key_B], 0), [1, 2])
    E_key_pair_ = tf.reshape(tf.concat([key_B, key_A], 0), [1, 2])

    E1_index = tf.where(tf.not_equal(E, key_A))
    E1 = tf.gather_nd(E, E1_index)
    E2_index = tf.where(tf.not_equal(E1, key_B))
    E_meditor = tf.gather_nd(E1, E2_index)

    index1, index2 = tf.meshgrid(E_key_pair, E_meditor)
    index1 = tf.reshape(index1, [1, -1])
    index2 = tf.reshape(index2, [1, -1])
    indices1 = tf.transpose(tf.concat([index1, index2], 0), (1, 0))
    indices2 = tf.transpose(tf.concat([index2, index1], 0), (1, 0))
    indices = tf.concat([indices1, indices2], 0)
    indices = tf.concat([indices, E_key_pair], 0)
    indices = tf.concat([indices, E_key_pair_], 0)
    indices = tf.cast(indices, tf.int64) #tf.to_int64(indices)
    values = tf.ones(tf.shape(indices)[0], dtype=tf.int64)
    shape = tf.constant([num_category, num_category], dtype=tf.int64)
    graph = tf.scatter_nd(indices, values, shape)
    adjacency = graph

    return adjacency, key_A, key_B


def message_pass(label, x, hidden_size, batch_size, num_category, graph):
    w_hidden_state = weights(label + '_hidden_state_out', hidden_size, 0)
    b_hidden_state = biases(label + '_hidden_state_out', hidden_size, 0)
    x_all = tf.reshape(tf.matmul(
        tf.reshape(x[:, 0, :], [batch_size, 2 * hidden_size]),
        w_hidden_state) + b_hidden_state,
                       [batch_size, 2 * hidden_size])
    for i in range(1, num_category):
        w_hidden_state = weights(label + '_hidden_state_out', hidden_size, i)
        b_hidden_state = biases(label + '_hidden_state_out', hidden_size, i)
        x_all_ = tf.reshape(tf.matmul(
            tf.reshape(x[:, i, :], [batch_size, 2 * hidden_size]),
            w_hidden_state) + b_hidden_state,
                            [batch_size, 2 * hidden_size])
        x_all = tf.concat([x_all, x_all_], 1)
    x_all = tf.reshape(x_all, [batch_size, num_category, 2 * hidden_size])
    x_all = tf.transpose(x_all, (0, 2, 1))

    x_ = x_all[0]
    graph = tf.to_float(graph)
    graph_ = graph[0]

    x = tf.matmul(x_, graph_)
    for i in range(1, batch_size):
        x_ = x_all[i]
        graph_ = graph[i]
        x_ = tf.matmul(x_, graph_)
        x = tf.concat([x, x_], 0)
    x = tf.reshape(x, [batch_size, 2 * hidden_size, num_category])
    x = tf.transpose(x, (0, 2, 1))

    w_hidden_state_in = weights(label + '_hidden_state_in', hidden_size, 0)
    b_hidden_state_in = biases(label + '_hidden_state_in', hidden_size, 0)
    x_ = tf.reshape(tf.matmul(x[:, 0, :], w_hidden_state_in) + b_hidden_state_in,
                    [batch_size, 2 * hidden_size])
    for j in range(1, num_category):
        w_hidden_state_in = weights(label + '_hidden_state_in', hidden_size, j)
        b_hidden_state_in = biases(label + '_hidden_state_in', hidden_size, j)
        _x = tf.reshape(tf.matmul(x[:, j, :], w_hidden_state_in) + b_hidden_state_in,
                        [batch_size, 2 * hidden_size])
        x_ = tf.concat([x_, _x], 1)
    x = tf.reshape(x_, [batch_size, num_category, 2 * hidden_size])

    return x


def GNN(label, data, batch_size, hidden_size, n_steps, num_category, category_w):
    gru_cell = GRUCell(2 * hidden_size)
    w_mlp1 = weights('in_' + label, hidden_size, 0)
    b_mlp1 = biases('in_' + label, hidden_size, 0)
    h0 = tf.reshape(tf.matmul(data[:, 0, :], w_mlp1) + b_mlp1,
                    [batch_size, hidden_size])  # initialize h0 [batchsize, hidden_state]
    category0 = tf.reshape(tf.matmul(tf.reshape(category_w[0], [1, 2048]), w_mlp1) + b_mlp1, [-1, hidden_size])
    for i in range(1, num_category):
        w_mlp1 = weights('in_' + label, hidden_size, i)
        b_mlp1 = biases('in_' + label, hidden_size, i)
        h0 = tf.concat([h0, tf.reshape(
            tf.matmul(data[:, i, :], w_mlp1) + b_mlp1, [batch_size, hidden_size])], 1)
        category0 = tf.concat([category0, tf.reshape(
            tf.matmul(tf.reshape(category_w[i], [1, 2048]), w_mlp1) + b_mlp1, [-1, hidden_size])], 1)
    h0 = tf.reshape(h0, [batch_size, num_category, hidden_size])
    category0 = tf.reshape(category0, [num_category, hidden_size])
    h0 = tf.nn.leaky_relu(h0)
    category0 = tf.nn.leaky_relu(category0)

    w_mlp2 = weights('out_' + label, hidden_size, 0)
    b_mlp2 = biases('out_' + label, hidden_size, 0)
    h1 = tf.reshape(tf.matmul(h0[:, 0, :], w_mlp2) + b_mlp2,
                    [batch_size, hidden_size])
    category1 = tf.reshape(tf.matmul(tf.reshape(category0[0], [1, hidden_size]), w_mlp2) + b_mlp2, [-1, hidden_size])
    for i in range(1, num_category):
        w_mlp2 = weights('out_' + label, hidden_size, i)
        b_mlp2 = biases('out_' + label, hidden_size, i)
        h1 = tf.concat([h1, tf.reshape(
            tf.matmul(h0[:, i, :], w_mlp2) + b_mlp2, [batch_size, hidden_size])], 1)
        category1 = tf.concat([category1, tf.reshape(
            tf.matmul(tf.reshape(category0[i], [1, hidden_size]), w_mlp2) + b_mlp2, [-1, hidden_size])], 1)
    h1 = tf.reshape(h1, [batch_size, num_category, hidden_size])  # h0: [batchsize, num_category, hidden_state]
    category1 = tf.reshape(category1, [num_category, hidden_size])
    h1 = tf.nn.tanh(h1)
    category1 = tf.nn.tanh(category1)

    w_mlp3 = weights('out_in_' + label, hidden_size, 0)
    b_mlp3 = biases('out_in_' + label, hidden_size, 0)
    h2 = tf.reshape(tf.matmul(h1[:, 0, :], w_mlp3) + b_mlp3,
                    [batch_size, hidden_size])
    category2 = tf.reshape(tf.matmul(tf.reshape(category1[0], [1, hidden_size]), w_mlp2) + b_mlp2, [-1, hidden_size])
    for i in range(1, num_category):
        w_mlp3 = weights('out_in_' + label, hidden_size, i)
        b_mlp3 = biases('out_in_' + label, hidden_size, i)
        h2 = tf.concat([h2, tf.reshape(
            tf.matmul(h1[:, i, :], w_mlp3) + b_mlp3, [batch_size, hidden_size])], 1)
        category2 = tf.concat([category2, tf.reshape(
            tf.matmul(tf.reshape(category1[i], [1, hidden_size]), w_mlp3) + b_mlp3, [-1, hidden_size])], 1)
    h2 = tf.reshape(h2, [batch_size, num_category, hidden_size])  # h0: [batchsize, num_category, hidden_state]
    category2 = tf.reshape(category2, [num_category, hidden_size])
    h2 = tf.nn.tanh(h2)
    category2 = tf.nn.tanh(category2)


    visual_state = h2
    cate_state = category2

    for i in range(batch_size):
        graphm, Sem, Iem = Laplacian(visual_state[i], num_category, cate_state, hidden_size)
        if i == 0:
            graph = graphm
            Se = Sem
            Ie = Iem
            dataw = tf.concat([visual_state[i], cate_state], 1)
        else:
            graph_ = graphm
            Se_ = Sem
            Ie_ = Iem
            dataw_ = tf.concat([visual_state[i], cate_state], 1)
            graph = tf.concat([graph, graph_], 0)
            Se = tf.concat([Se, Se_], 0)
            Ie = tf.concat([Ie, Ie_], 0)
            dataw = tf.concat([dataw, dataw_], 0)

    graph = tf.reshape(graph, [-1, num_category, num_category])
    Se = tf.reshape(Se, [-1, 1])
    Ie = tf.reshape(Ie, [-1, 1])
    dataw = tf.reshape(dataw, [batch_size, num_category, 2 * hidden_size])
    state = dataw

    sum_graph = tf.compat.v1.reduce_sum(graph, reduction_indices=1)
    enable_node = tf.cast(tf.cast(sum_graph, dtype=bool), dtype=tf.float32)

    with tf.compat.v1.variable_scope("gnn"):
        for step in range(n_steps):
            if step > 0: tf.compat.v1.get_variable_scope().reuse_variables()
            x = message_pass(label, state, hidden_size, batch_size, num_category, graph)
            (x_new, state_new) = gru_cell(x[0], state[0])
            state_new = tf.transpose(state_new, (1, 0))
            state_new = tf.multiply(state_new, enable_node[0])
            state_new = tf.transpose(state_new, (1, 0))
            for i in range(1, batch_size):
                (x_, state_) = gru_cell(x[i], state[i])  # #input of GRUCell must be 2 rank, not 3 rank
                state_ = tf.transpose(state_, (1, 0))
                state_ = tf.multiply(state_, enable_node[i])
                state_ = tf.transpose(state_, (1, 0))
                state_new = tf.concat([state_new, state_], 0)
            state = tf.reshape(state_new, [batch_size, num_category, -1])  # #restore: 2 rank to 3 rank

 
    return state, Se, Ie


