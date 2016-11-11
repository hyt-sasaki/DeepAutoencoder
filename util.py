# -*- coding:utf-8 -*-
import tflearn
import tensorflow as tf
import numpy as np
import os


def cross_entropy(incoming, placeholder):
    with tf.name_scope('cross_entropy'):
        cliped_out1 = tf.clip_by_value(
            incoming, 1e-10, float('inf')
        )
        cliped_out2 = tf.clip_by_value(
            1 - incoming, 1e-10, float('inf')
        )

        cost = - tf.reduce_mean(
            placeholder *
            tf.log(cliped_out1) +
            (1 - placeholder) *
            tf.log(cliped_out2)
        )
    return cost


def create_cost(c, reg):
    if isinstance(c, str):
        c = tflearn.optimizers.get(c)
    def _cost(incoming, placeholder):
        return c(incoming, placeholder) + reg

    return _cost


def fully_connected_with_tied_weight(incoming, encoder_weight, activation='linear', bias=True,
                    bias_init='zeros',
                    trainable=True,
                    restore=True, reuse=False, scope=None,
                    name="FullyConnected"):

    n_units = encoder_weight.get_shape()[0]
    input_shape = tflearn.utils.get_incoming_shape(incoming)
    assert len(input_shape) > 1, "Incoming Tensor shape must be at least 2-D"
    n_inputs = int(np.prod(input_shape[1:]))

    # Build variables and inference.
    with tf.variable_scope(scope, name, values=[incoming], reuse=reuse) as scope:
        name = scope.name

        W = tf.transpose(encoder_weight)
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, W)

        b = None
        if bias:
            if isinstance(bias_init, str):
                bias_init = tflearn.initializations.get(bias_init)()
            b = tflearn.variables.variable('b', shape=[n_units], initializer=bias_init,
                            trainable=trainable, restore=restore)
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, b)

        inference = incoming
        # If input is not 2d, flatten it.
        if len(input_shape) > 2:
            inference = tf.reshape(inference, [-1, n_inputs])

        inference = tf.matmul(inference, W)
        if b: inference = tf.nn.bias_add(inference, b)

        if isinstance(activation, str):
            inference = tflearn.activations.get(activation)(inference)
        elif hasattr(activation, '__call__'):
            inference = activation(inference)
        else:
            raise ValueError("Invalid Activation.")

        # Track activations.
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inference)

    # Add attributes to Tensor to easy access weights.
    inference.scope = scope
    inference.W = W
    inference.b = b

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference

def create_id_decorator(func):
    import util
    import functools
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        id_name = self.name + '_lr=%s_bs=%s' % (self.learning_rate, self.batch_size)
        id_name += func(self, *args, **kwargs)
        tensorboard_dir = self.model.trainer.tensorboard_dir
        id_name = util.create_id(id_name, tensorboard_dir, args[0])
        return id_name
    return wrapper


def create_id(id_name, tensorboard_dir, restore):
    run_times = 0
    if os.path.exists(tensorboard_dir):
        existence = False
        for e in os.listdir(tensorboard_dir):
            if e[:len(id_name)] == id_name:
                existence = True
                r = int(e[len(id_name):])
                if r > run_times:
                    run_times = r
        if existence and (not restore):
            run_times += 1
    id_name += str(run_times)
    return id_name
