# -*- coding:utf-8 -*-
import tflearn
import tensorflow as tf
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import util
import os


class Model(BaseEstimator, RegressorMixin):
    def __init__(
        self, optimizer='adam',
        learning_rate=1e-3, batch_size=20,
        loss='mean_square', n_epoch=1,
        graph=None, name='Model', logdir='logs'
    ):
        self.n_epoch = n_epoch
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        self.batch_size = batch_size
        self.graph = graph
        self.name = name
        self.logdir = logdir

    def create_network(self):
        self.input_ph = tflearn.input_data(shape=[None, 1])
        self.output_layer = self.input_ph

        return self

    @util.create_id_decorator
    def _create_id(self, restore):
        id_suffix = ''
        return id_suffix

    def _create_model(self):
        if self.graph is None:
            tf.reset_default_graph()
            graph = tf.get_default_graph()
        else:
            graph = self.graph
        with graph.as_default():
            with tf.name_scope(self.name):
                self.create_network()

            net = tflearn.regression(
                self.output_layer, optimizer=self.optimizer,
                learning_rate=self.learning_rate, loss=self.loss
            )

            self.model = tflearn.DNN(net, tensorboard_verbose=3, tensorboard_dir=self.logdir)

        self.has_model = True

        return self

    def fit(self, X, y, restore=False):
        if not hasattr(self, 'has_model'):
            self._create_model()
        else:
            if not self.has_model:
                self._create_model()

        id_name = self._create_id(restore)

        tensorboard_dir = os.path.abspath(self.model.trainer.tensorboard_dir)
        run_dir = os.path.join(tensorboard_dir, id_name)
        model_file = os.path.join(run_dir, self.name+'.tflearn')

        if restore and os.path.exists(model_file):
            self.model.load(model_file)

        self.model.fit(
            X, y, n_epoch=self.n_epoch,
            run_id=id_name, batch_size=self.batch_size
        )

        self.model.save(model_file)

        return self

    def predict(self, X):
        if X.ndim == 1:
            X_= X.reshape(1, X.shape[0])
        else:
            X_ = X
        predY = self.model.predict(X_)
        if X.ndim == 1:
            predY = predY[0]

        return predY

    def get_output_layer(self):
        return self.output_layer
