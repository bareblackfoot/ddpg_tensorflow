import numpy as np
import math
import tensorflow as tf

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        #Now create the model
        self.state, self.outputs, self.weights = self.create_actor_network(state_size, 'behavior')
        self.target_state, self.target_outputs, self.target_weights = self.create_actor_network(state_size, 'target', trainable=False)
        ema = tf.train.ExponentialMovingAverage(decay=1-self.TAU)
        self.target_update = ema.apply(self.target_weights)
        self.action_gradient = tf.placeholder(tf.float32, [None, action_size])
        self.params_grad = tf.gradients(self.outputs, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.update_ops = self.build_update_operation()
        sess.run(tf.global_variables_initializer())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        self.sess.run(self.update_ops)

    def build_update_operation(self):  # Define parameter update operation in TF graph
        update_ops = []
        for var, var_old in zip(self.weights, self.target_weights):  # Update Target Network's Parameter with Prediction Network
            aa = self.TAU * var + (1 - self.TAU) * var_old
            update_ops.append(var_old.assign(aa))
        return update_ops

    def create_actor_network(self, state_size, name, trainable=True):
        S = tf.placeholder(tf.float32, [None, state_size])
        with tf.variable_scope('actor'):
            with tf.variable_scope(name):
                h0 = tf.layers.dense(S, units=HIDDEN1_UNITS, activation=tf.nn.relu, kernel_initializer= tf.contrib.layers.xavier_initializer(),name="h0", trainable=trainable)
                h1 = tf.layers.dense(h0, units=HIDDEN1_UNITS, activation=tf.nn.relu, kernel_initializer= tf.contrib.layers.xavier_initializer(), name="h1", trainable=trainable)
                Steering = tf.layers.dense(h1, 1, activation=tf.tanh, kernel_initializer=tf.initializers.random_normal(), name="Steering", trainable=trainable)
                Acceleration = tf.layers.dense(h1, 1, activation=tf.sigmoid, kernel_initializer=tf.initializers.random_normal(), name="Acceleration", trainable=trainable)
                Brake = tf.layers.dense(h1, 1, activation=tf.sigmoid, kernel_initializer=tf.initializers.random_normal(), name="Brake", trainable=trainable)
                V = tf.concat([Steering, Acceleration, Brake], axis=-1)
        weights = [var for var in tf.global_variables() if "actor" in var.name and name in var.name]
        return S, V, weights

    def predict(self, state_batch):
        return self.sess.run(self.outputs, feed_dict={
            self.state: state_batch
            })

    def target_predict(self, state_batch):
        return self.sess.run(self.target_outputs, feed_dict={
            self.target_state: state_batch
            })
