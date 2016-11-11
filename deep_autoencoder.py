# -*- coding:utf-8 -*-
import tflearn
import tensorflow as tf
import numpy as np
from model import Model
import util


class DeepAutoEncoder(Model):
    def __init__(
        self, n_input=None, layers=[], optimizer='adam',
        learning_rate=1e-3, batch_size=20,
        loss='mean_square', n_epoch=1,
        batch_normalization=True, tied_weight=True,
        encoder_act='relu', decoder_act='sigmoid',
        graph=None, name='DeepAutoEncoder', logdir='logs'
    ):
        super(DeepAutoEncoder, self).__init__(
            optimizer=optimizer,
            learning_rate=learning_rate,
            batch_size=batch_size,
            loss=loss, n_epoch=n_epoch,
            graph=graph, name=name, logdir=logdir
        )
        self.n_input = n_input
        self.layers = layers
        self.batch_normalization = batch_normalization
        self.tied_weight = tied_weight
        self.encoder_act = encoder_act
        self.decoder_act = decoder_act

    def create_network(self):
        self.input_ph = tflearn.input_data(shape=[None, self.n_input])

        bn_name = 'BN'
        encoder = create_encoder(
            self.input_ph, layers=self.layers,
            activation=self.encoder_act,
            batch_normalization=self.batch_normalization,
            bn_name=bn_name
        )

        self.feature_layer = encoder
        if self.tied_weight:
            layers = self.layers
        else:
            n_in =self.input_ph.get_shape()[1]
            layers = [n_in, ] + self.layers

        decoder = create_decoder(
            incoming=encoder, layers=layers,
            activation=self.decoder_act,
            batch_normalization=self.batch_normalization,
            tied_weight=self.tied_weight,
            bn_name=bn_name
        )
        self.output_layer = decoder

        return self

    @util.create_id_decorator
    def _create_id(self, restore):
        id_suffix = '_tied=%s' % self.tied_weight
        id_suffix += '_batch_normalized=%s_' % self.batch_normalization

        return id_suffix

    def encode(self, X):
        if X.ndim == 1:
            X_= X.reshape(1, X.shape[0])
        else:
            X_ = X
        encoder_net = tflearn.Evaluator(self.feature_layer, session=self.model.session)

        feed_dict = {self.input_ph: X_}
        feature = encoder_net.predict(feed_dict)
        if X.ndim == 1:
            feature = feature[0]

        return feature


def create_encoder(
    incoming, layers,
    activation='relu', batch_normalization=True,
    base_name='Encoder', bn_name='BN'
):
    encoder_base_name = base_name + '%s'

    encoder = incoming
    for l, layer in enumerate(layers):
        encoder_name = encoder_base_name % (l + 1)
        encoder = tflearn.fully_connected(
            incoming=encoder, n_units=layer,
            bias=not(batch_normalization),
            name=encoder_name
        )
        if batch_normalization:
            encoder = tflearn.batch_normalization(encoder, name=encoder_name+'/'+bn_name)
        if isinstance(activation, str):
            encoder = tflearn.activations.get(activation)(encoder)
        else:
            try:
                encoder = activation(encoder)
            except Exception as e:
                print e.message

    return encoder


def create_decoder(
    incoming, layers,
    activation='sigmoid', batch_normalization=True, tied_weight=True,
    encoder_base_name='Encoder', decoder_base_name='Decoder', bn_name='BN'
):
    decoder = incoming
    decoder_base_name = decoder_base_name + '%s'
    encoder_base_name = encoder_base_name + '%s'
    for l, layer in reversed(list(enumerate(layers))):
        decoder_name = decoder_base_name % (l + 1)
        if tied_weight:
            encoder_name = encoder_base_name % (l + 1)
            W = tflearn.get_layer_variables_by_name(encoder_name)[0]
            decoder = util.fully_connected_with_tied_weight(
                incoming=decoder, encoder_weight=W,
                bias=not(batch_normalization),
                name=decoder_name
            )
        else:
            decoder = tflearn.fully_connected(
                incoming=decoder, n_units=layer,
                bias=not(batch_normalization),
                name=decoder_name
            )
        if batch_normalization:
            decoder = tflearn.batch_normalization(decoder, name=decoder_name+'/'+bn_name)
        if isinstance(activation, str):
            decoder = tflearn.activations.get(activation)(decoder)
        else:
            try:
                decoder = activation(decoder)
            except Exception as e:
                print e.message

    return decoder
