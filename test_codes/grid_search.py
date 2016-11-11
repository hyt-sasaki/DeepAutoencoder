# -*- coding:utf-8 -*-
import sys, os
sys.path.append(os.pardir)
from deep_autoencoder import DeepAutoEncoder
from sklearn.model_selection import GridSearchCV
import numpy as np
import tflearn
import tensorflow as tf


if __name__ == '__main__':
    import tflearn.datasets.mnist as mnist
    dataset = mnist.load_data(data_dir="./mnist/", one_hot=True)
    X = dataset[0]
    y = X

    n_input = X.shape[1]
    layers = [64, 10]
    n_epoch = 1
    graph = None
    name = 'DeepAutoencoder'
    logdir = 'logs'

    base_parameters = {
        'n_input': [n_input],
        'layers': [layers],
        'n_epoch': [n_epoch],
        'graph': [graph],
        'name': [name],
        'logdir': [logdir],
    }

    parameters = {
        'learning_rate': [1e-1],
        'batch_size': [400, 500],
        'optimizer': ['adam'],
        'loss': ['mean_square'],
        'batch_normalization': [True],
        'tied_weight': [True],
        'encoder_act': ['relu'],
        'decoder_act': ['tanh']
    }

    parameters.update(base_parameters)


    reg = GridSearchCV(DeepAutoEncoder(), parameters, cv=2)
    reg.fit(X, y)
    best = reg.best_estimator_
    print 'best parameter'
    print best
