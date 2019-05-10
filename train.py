# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import time

from argparse import ArgumentParser, Namespace
from sklearn.model_selection import train_test_split

import tensorflow as tf

from utils import load_dataset, load_vocab, convert_vocab, loss_function
from utils import CustomSchedule, Mask
from model import Transformer 


def train(args: Namespace):
    input_tensor, target_tensor, input_lang_tokenizer, target_lang_tokenizer = load_dataset('./data/', args.max_len)

    max_len_input = len(input_tensor[0])
    max_len_target = len(target_tensor[0])

    print('max len of each seq:', max_len_input, ',', max_len_target)

    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=args.dev_split)

    # init hyperparameter
    EPOCHS = args.epoch
    BATCH_SIZE = args.batch_size 
    steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
    d_model = args.d_model
    d_ff = args.d_ff
    num_layers = args.layers
    num_heads = args.heads
    input_vocab_size = len(input_lang_tokenizer.word_index) + 1
    target_vocab_size = len(target_lang_tokenizer.word_index) + 1
    BUFFER_SIZE = len(input_tensor_train)

    setattr(args, 'max_len_input', max_len_input)
    setattr(args, 'max_len_target', max_len_target)

    setattr(args, 'steps_per_epoch', steps_per_epoch)
    setattr(args, 'input_vocab_size', input_vocab_size)
    setattr(args, 'target_vocab_size', target_vocab_size)
    setattr(args, 'BUFFER_SIZE', BUFFER_SIZE)

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)

    print('dataset shape (batch_size, max_len):', dataset)

    # Creating Transformer
    transformer = Transformer(num_layers, d_model, num_heads, 
                              d_ff, input_vocab_size, target_vocab_size, args.dropout_prob)

    # Creating Loss and Optimizer
    learning_rate = CustomSchedule(d_model) 
    optimizer = tf.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    # Setting checkpoint
    now = time.localtime(time.time())
    now_time = '/{}{}{}{}'.format(now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)
    checkpoint_dir = './training_checkpoints' + now_time
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     transformer=transformer)
    setattr(args, 'checkpoint_dir', checkpoint_dir) 

    os.makedirs(checkpoint_dir, exist_ok=True)

    # saving information of the model
    with open('{}/config.json'.format(checkpoint_dir), 'w', encoding='UTF-8') as fout:
        json.dump(vars(args), fout, indent=2, sort_keys=True)


    @tf.function
    def train_step(_input, _target):
        tar_inp = _target[:, :-1]
        tar_real = _target[:, 1:]
        
        enc_padding_mask, combined_mask, dec_padding_mask = \
            Mask.create_masks(_input, tar_inp)

        with tf.GradientTape() as tape:

            predictions = transformer(_input, tar_inp,
                                      enc_padding_mask,
                                      combined_mask,
                                      dec_padding_mask)
            loss = loss_function(loss_object, tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)

        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        return tf.reduce_mean(loss) 

    min_total_loss = 1000

    for epoch in range(EPOCHS):
        start = time.time()

        total_loss = 0

        for(batch, (_input, _target)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(_input, _target)
            total_loss += batch_loss

            if batch % 10 == 0:
                print('Epoch {}/{} Batch {}/{} Loss {:.4f}'.format(epoch + 1,
                                                             EPOCHS,
                                                             batch + 10,
                                                             steps_per_epoch,
                                                             batch_loss.numpy()))


        print('Epoch {}/{} Total Loss per epoch {:.4f} - {} sec'.format(epoch + 1, 
                                                             EPOCHS,
                                                             total_loss / steps_per_epoch,
                                                             time.time() - start))

        # saving checkpoint 
        if min_total_loss > total_loss / steps_per_epoch:
            print('Saving checkpoint...')
            min_total_loss = total_loss / steps_per_epoch
            checkpoint.save(file_prefix= checkpoint_prefix)

        print('\n')



def main():
    pass



if __name__=='__main__':
    main()
