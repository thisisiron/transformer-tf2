# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf #TF2

class Encoder(tf.keras.Model):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer,N)
        self.norm = LayerNorm()

    def call(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(tf.keras.layers.Layer):
    def __init__(self, axis=-1, eps=1e-6):
       super(LayerNorm, self).__init__()
       self.axis = axis

    def build(self, input_shape):
        dim = input_shape[-1]
        self.a_2 = self.add_weight(
            name='a_2',
            shape=(dim,),
            initializer='ones',
            trainable=True)
        self.b_2 = self.add_weight(
            name='b_2',
            shape=(dim,),
            initializer='zeros',
            trainable=True)
        return super(LayerNorm, self).build(input_shape)
    
    def call(self, inputs, **kwargs):
        mean = tf.reduce_mean(inputs, axis=self.axis, keepdims=True)
        #variance = tf.reduce_mean(tf.square(inputs - mean), axis=self.axis, keepdims=True)
        variance = tf.math.reduce_std(inputs, axis=self.axis, keepdims=True)
        epsilon = tf.constant(1e-5)
        normalized_inputs = (inputs - mean) / tf.sqrt(variance + epsilon)
        result = self.a_2 * normalized_inputs + self.b_2
        return result


class EncdoerLayer(tf.keras.Model):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncdoerLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def call(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class MultiHeadedAttention(tf.keras.Model):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.W_q = tf.keras.layers.Dense(d_model)
        self.W_k = tf.keras.layers.Dense(d_model)
        self.W_v = tf.keras.layers.Dense(d_model)
        self.linear = tf.keras.layers.Dense(d_model)

        self.attn = None
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, query, key, value, mask=None):
        if mask is not None:
            mask = tf.expand_dims(mask, axis=1)

        batch_size = tf.shape(query)[0]

        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)

        for x in [query, key, value]:
            x = tf.transpose(tf.reshape(x, (batch_size, -1, self.h, self.d_k)), [0, 2, 1, 3])

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = tf.reshape(tf.transpose(x, [0, 2, 1, 3]), (batch_size, -1, self.h * self.d_k))

        return self.linear(x)
    
    def attention(self, query, key, value, mask=None, dropout=None):
        scores = tf.matmul(query, key, transpose_b=True) / tf.sqrt(self.d_k)
        if mask is None:
            scores 
        p_attn = tf.nn.softmax(scores, axis=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return tf.matmul(p_attn, value), p_attn
        

class PositionwiseFeedForward(tf.keras.Model):
    """FFN(x) = max(0, xW_1+b_1)W_2 + b_2
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.W_1 = tf.keras.layers.Dense(d_ff)
        self.W_2 = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x):
        return self.W_2(self.dropout(tf.nn.relu(self.W_1(x))))

class Embedder(tf.keras.Model):
    def __init__(self, d_model, vocab):
        super(Embedder, self).__init__()
        self.emb = tf.keras.layers.Embedding(vocab, d_model)
        self.d_model = d_model
        self.vocab = vocab

    def call(self, x):

        seq_len = tf.shape(x)[1]
        print('seq_len:', seq_len)
        x = self.emb(x) * tf.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.get_positional_encoding()[:, :seq_len, :]
        return x 

    def get_positional_encoding(self):

        pos = np.expand_dims(np.arange(0, self.vocab), axis=1)
        div_term = np.array([[1 / np.power(10000, (2 * (i//2) / self.d_model)) for i in range(self.d_model)]])       
        pos = pos * div_term

        pe = np.zeros((self.vocab, self.d_model))
        pe[:, 0::2] = np.sin(pos[:, 0::2])
        pe[:, 1::2] = np.cos(pos[:, 0::2])

        pe = np.expand_dims(pe, 0)

        return tf.cast(pe, dtype=tf.float32) 


def main():
    d_model = 512
    N = 6
    d_ff = 2048
    h = 8
    dropout=0.1


if __name__=='__main__':
    main()
