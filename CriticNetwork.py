import numpy as np
import math
import tensorflow as tf

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600


class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        self.label = tf.placeholder(tf.float32, [None, action_size])

        #Now create the model
        self.state, self.outputs, self.action, self.weights = self.create_critic_network(state_size, action_size, 'behavior')
        self.target_state, self.target_outputs, self.target_action, self.target_weights = self.create_critic_network(state_size, action_size, 'target')
        # self.target_state, self.target_outputs, self.target_action, self.target_weights, self.target_update = self.create_target_critic_network(state_size, action_size)
        self.action_grads = tf.gradients(self.outputs, self.action)  #GRADIENTS for policy update
        self.mse_loss = tf.reduce_mean(tf.square(self.label - self.outputs))
        # weight_decay = tf.add_n([0.01*tf.nn.l2_loss(var) for var in self.weights if "kernel" in var.name])
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.mse_loss)
        self.update_ops = self.build_update_operation()
        sess.run(tf.global_variables_initializer())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        # critic_weights = self.weights
        # critic_target_weights = self.target_weights
        # for i in xrange(len(critic_weights)):
        #     critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU) * critic_target_weights[i]
        self.sess.run(self.update_ops)
        # self.sess.run(self.target_update)

    # def build_update_operation(self, new_weights):  # Define parameter update operation in TF graph
    #     update_ops = []
    #     for var, var_old in zip(new_weights, self.target_weights):  # Update Target Network's Parameter with Prediction Network
    #         update_ops.append(var_old.assign(var))
    #     return update_ops

    def build_update_operation(self):  # Define parameter update operation in TF graph
        update_ops = []
        for var, var_old in zip(self.weights, self.target_weights):  # Update Target Network's Parameter with Prediction Network
            var = self.TAU * var + (1 - self.TAU) * var_old
            update_ops.append(var_old.assign(var))
        return update_ops

    def create_critic_network(self, state_size, action_dim, name):
        S = tf.placeholder(tf.float32, [None, state_size])
        A = tf.placeholder(tf.float32, [None, action_dim])
        with tf.variable_scope('critic'):
            with tf.variable_scope(name):
                w1 = tf.layers.dense(S, HIDDEN1_UNITS, activation=tf.nn.relu, kernel_initializer=tf.initializers.random_normal(), name="w1")
                a1 = tf.layers.dense(A, HIDDEN1_UNITS, activation=tf.nn.relu,  kernel_initializer=tf.initializers.random_normal(), name="a1")
                h1 = tf.layers.dense(w1, HIDDEN1_UNITS, activation=tf.nn.relu, kernel_initializer=tf.initializers.random_normal(), name="h1")
                h2 = tf.concat([h1, a1], axis=-1)
                h3 = tf.layers.dense(h2, HIDDEN1_UNITS, activation=tf.nn.relu, kernel_initializer=tf.initializers.random_normal(), name="h3")
                V = tf.layers.dense(h3, action_dim)
        weights = [var for var in tf.trainable_variables() if "critic" in var.name and name in var.name]
        return S, V, A, weights

    # def create_target_critic_network(self, state_size, action_dim):
    #     S = tf.placeholder(tf.float32, [None, state_size])
    #     A = tf.placeholder(tf.float32, [None, action_dim])
    #     ema = tf.train.ExponentialMovingAverage(decay=1 - self.TAU)
    #     target_update = ema.apply(self.weights)
    #     target_net = [ema.average(x) for x in self.weights]
    #     # with tf.variable_scope('critic'):
    #     #     with tf.variable_scope(name):
    #     w1 = tf.nn.relu(tf.matmul(S, target_net[0]) + target_net[1])
    #     a1 = tf.matmul(A, target_net[2]) + target_net[3]
    #     h1 = tf.matmul(w1, target_net[4]) + target_net[5]
    #     h2 = tf.concat([h1, a1], axis=-1)
    #     h3 = tf.nn.relu(tf.matmul(h2, target_net[6]) + target_net[7])
    #     V = tf.matmul(h3, target_net[8]) + target_net[9]
    #     # weights = [var for var in tf.trainable_variables() if "critic" in var.name and name in var.name]
    #     return S, V, A, target_net, target_update

    def predict(self, state_batch, action_batch):
        return self.sess.run(self.outputs, feed_dict={
            self.state: state_batch,
            self.action: action_batch
            })

    def target_predict(self, state_batch, action_batch):
        return self.sess.run(self.target_outputs, feed_dict={
            self.target_state: state_batch,
            self.target_action: action_batch
            })

    def train_on_batch(self, state_batch, action_batch, y_t):
        _, loss = self.sess.run([self.optimize, self.mse_loss], feed_dict={
            self.state: state_batch,
            self.action: action_batch,
            self.label: y_t
            })
        return loss