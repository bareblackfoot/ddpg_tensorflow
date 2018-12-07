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
        self.target_state, self.target_outputs, self.target_action, self.target_weights = self.create_critic_network(state_size, action_size, 'target', trainable=False)
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
        self.sess.run(self.update_ops)

    def build_update_operation(self):  # Define parameter update operation in TF graph
        update_ops = []
        for var, var_old in zip(self.weights, self.target_weights):  # Update Target Network's Parameter with Prediction Network
            aa = self.TAU * var + (1 - self.TAU) * var_old
            update_ops.append(var_old.assign(aa))
        return update_ops

    def create_critic_network(self, state_size, action_dim, name, trainable=True):
        S = tf.placeholder(tf.float32, [None, state_size])
        A = tf.placeholder(tf.float32, [None, action_dim])
        with tf.variable_scope('critic'):
            with tf.variable_scope(name):
                w1 = tf.layers.dense(S, HIDDEN1_UNITS, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="w1", trainable=trainable)
                a1 = tf.layers.dense(A, HIDDEN1_UNITS,  kernel_initializer=tf.contrib.layers.xavier_initializer(), name="a1", trainable=trainable)
                h1 = tf.layers.dense(w1, HIDDEN1_UNITS, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="h1", trainable=trainable)
                h2 = tf.concat([h1, a1], axis=-1)
                h3 = tf.layers.dense(h2, HIDDEN1_UNITS, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="h3", trainable=trainable)
                V = tf.layers.dense(h3, action_dim, trainable=trainable)
        weights = [var for var in tf.global_variables() if "critic" in var.name and name in var.name]
        return S, V, A, weights

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