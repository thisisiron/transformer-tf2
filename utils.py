# -*- coding: utf-8 -*-

import io
from tqdm import tqdm
import tensorflow as tf

FILE_PATH = './data/'

def create_dataset(path, limit_size=None):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    lines = ['<s> ' + line + ' </s>' for line in tqdm(lines[:limit_size])]

    # Print examples
    for line in lines[:5]:
        print(line)

    return lines 

def create_dataset_test(path, lang=['en', 'de']):
    dataset_train_input_path = 'train.{}'.format(lang[0]) 
    dataset_train_target_path = 'train.{}'.format(lang[1])
    in_sent = create_dataset(path + dataset_train_en_path, 50000)
    print(in_sent)

def tokenize(text, vocab, max_len):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<unk>')

    lang_tokenizer.word_index = vocab

    tensor = lang_tokenizer.texts_to_sequences(text)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=max_len, padding='post')

    return tensor, lang_tokenizer

def load_dataset(path, max_len, limit_size=None, lang=['en', 'de']):
    dataset_train_input_path = 'train.{}'.format(lang[0]) 
    dataset_train_target_path = 'train.{}'.format(lang[1])

    print('Loading...')
    vocab_input = load_vocab(path, lang[0])
    vocab_target = load_vocab(path, lang[1])
    
    input_text = create_dataset(path + dataset_train_input_path, limit_size)
    target_text = create_dataset(path + dataset_train_target_path,limit_size)

    input_tensor, input_lang_tokenizer = tokenize(input_text, vocab_input, max_len)
    target_tensor, target_lang_tokenizer = tokenize(target_text, vocab_target, max_len)

    return input_tensor, target_tensor, input_lang_tokenizer, target_lang_tokenizer

def max_length(tensor):
    return max(len(t) for t in tensor)
    
def load_dataset_test(path):

    it, tt, ilt, tlt =  load_dataset(path, 90, 5000)
    print(tt[0].shape)
    print(it.shape, tt.shape)
    max_it, max_tt = max_length(it), max_length(tt)
    print(max_it, max_tt)

def load_vocab(path, lang):
    lines = io.open(path + 'vocab.50K.{}'.format(lang), encoding='UTF-8').read().strip().split('\n')
    vocab = {}
    
    # 0 is padding
    for idx, word in enumerate(lines):
        vocab[word] = idx + 1

    return vocab

def convert_vocab(tokenizer, vocab):
    for key, val in vocab.items():
        tokenizer.index_word[val] = key

def loss_function(loss_object, y_true, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    loss = loss_object(y_true, y_pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    return tf.reduce_mean(loss)

class Mask():
    """ref: https://www.tensorflow.org/alpha/tutorials/text/transformer#masking
    """
    @staticmethod
    def create_padding_mask(seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions so that we can add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    @staticmethod
    def create_look_ahead_mask(size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)

    @staticmethod
    def create_masks(inp, tar):
        # Encoder padding mask
        enc_padding_mask = Mask.create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        last_dec_padding_mask = Mask.create_padding_mask(inp)

        dec_padding_mask = Mask.create_padding_mask(tar)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by 
        # the decoder.
        look_ahead_mask = Mask.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = Mask.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        
        return enc_padding_mask, combined_mask, dec_padding_mask, last_dec_padding_mask

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """ref: https://www.tensorflow.org/alpha/tutorials/text/transformer#optimizer
    """
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def main():
    #load_dataset_test(FILE_PATH)
    pass

