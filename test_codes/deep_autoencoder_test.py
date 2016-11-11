import sys, os
sys.path.append(os.pardir)
from deep_autoencoder import DeepAutoEncoder
import tflearn.datasets.mnist as mnist
import tensorflow as tf


if __name__ == '__main__':
    X, Y, testX, testY = mnist.load_data(one_hot=True)
    y = X

    n_input = X.shape[1]
    layers = [64, 10]
    optimizer = 'adam'
    learning_rate = 1e-2
    batch_size = 1000
    loss = 'mean_square'
    n_epoch = 1
    batch_normalization = True
    tied_weight = True
    encoder_act = 'relu'
    decoder_act = 'sigmoid'
    graph = tf.get_default_graph()
    name = 'DeepAutoencoder'
    logdir = 'logs'

    model = DeepAutoEncoder(
        n_input=n_input,
        layers=layers,
        optimizer=optimizer,
        learning_rate=learning_rate,
        batch_size=batch_size,
        loss=loss,
        n_epoch=n_epoch,
        batch_normalization=batch_normalization,
        tied_weight=tied_weight,
        encoder_act=encoder_act,
        decoder_act=decoder_act,
        graph=graph,
        name=name,
        logdir=logdir
    )

    model.fit(X, y, True)
