import numpy as np  # linear algebra
import pandas as pd
import re
import os
from time import time
import spacy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
import gensim
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings


def text_strip(column):
    for row in column:

        # ORDER OF REGEX IS VERY VERY IMPORTANT!!!!!!

        row = re.sub("(\\t)", ' ', str(row)).lower()  # remove escape charecters
        row = re.sub("(\\r)", ' ', str(row)).lower()
        row = re.sub("(\\n)", ' ', str(row)).lower()

        row = re.sub("(__+)", ' ', str(row)).lower()  # remove _ if it occors more than one time consecutively
        row = re.sub("(--+)", ' ', str(row)).lower()  # remove - if it occors more than one time consecutively
        row = re.sub("(~~+)", ' ', str(row)).lower()  # remove ~ if it occors more than one time consecutively
        row = re.sub("(\+\++)", ' ', str(row)).lower()  # remove + if it occors more than one time consecutively
        row = re.sub("(\.\.+)", ' ', str(row)).lower()  # remove . if it occors more than one time consecutively

        row = re.sub(r"[<>()|&©ø\[\]\'\",;?~*!]", ' ', str(row)).lower()  # remove <>()|&©ø"',;?~*!

        row = re.sub("(mailto:)", ' ', str(row)).lower()  # remove mailto:
        row = re.sub(r"(\\x9\d)", ' ', str(row)).lower()  # remove \x9* in text
        row = re.sub("([iI][nN][cC]\d+)", 'INC_NUM', str(row)).lower()  # replace INC nums to INC_NUM
        row = re.sub("([cC][mM]\d+)|([cC][hH][gG]\d+)", 'CM_NUM', str(row)).lower()  # replace CM# and CHG# to CM_NUM

        row = re.sub("(\.\s+)", ' ', str(row)).lower()  # remove full stop at end of words(not between)
        row = re.sub("(\-\s+)", ' ', str(row)).lower()  # remove - at end of words(not between)
        row = re.sub("(\:\s+)", ' ', str(row)).lower()  # remove : at end of words(not between)

        row = re.sub("(\s+.\s+)", ' ', str(row)).lower()  # remove any single charecters hanging between 2 spaces

        # Replace any url as such https://abc.xyz.net/browse/sdf-5327 ====> abc.xyz.net
        try:
            url = re.search(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', str(row))
            repl_url = url.group(3)
            row = re.sub(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', repl_url, str(row))
        except:
            pass  # there might be emails with no url in them

        row = re.sub("(\s+)", ' ', str(row)).lower()  # remove multiple spaces

        # Should always be last
        row = re.sub("(\s+.\s+)", ' ', str(row)).lower()  # remove any single charecters hanging between 2 spaces

        yield row


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))

    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:

        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        if (sampled_token != 'eostok'):
            decoded_sentence += ' ' + sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'eostok' or len(decoded_sentence.split()) >= (max_summary_len - 1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

        return decoded_sentence


def seq2summary(input_seq):
    newString = ''
    for i in input_seq:
        if ((i != 0 and i != target_word_index['sostok']) and i != target_word_index['eostok']):
            newString = newString + reverse_target_word_index[i] + ' '
    return newString


def seq2text(input_seq):
    newString = ''
    for i in input_seq:
        if (i != 0):
            newString = newString + reverse_source_word_index[i] + ' '
    return newString
