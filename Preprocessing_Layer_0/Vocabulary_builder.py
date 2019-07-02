
import os.path
import operator
import pickle
from nltk.tokenize import WhitespaceTokenizer
from gensim.models import Word2Vec, KeyedVectors
from collections import defaultdict
from math import sqrt
import numpy as np
import codecs
import re
import string
import sys
import tqdm as tqdm
import os
from collections import Counter
# import Squad_processor
import spacy

import nltk
from argparse import ArgumentParser


class Vocabulary():

    def __init__(self, vocab_input_files,vocab_output_filename):
        """
        This function works the same as contructors and is used to initilaize the parameters used in the making the model

        """
        self.vocab = {}
        self.vocab_output_filename = vocab_output_filename
        self.vocab_input_files = vocab_input_files
        self.word_list = []
        self.word_to_index = {} # dictionary with keys as words and values as their corresponding index number
        self.char_to_index = {} # dictionary with keys as characters and values as their corresponding index number
        self.word_to_index["<pad>"] = 0
        self.word_to_index["<sos>"] = 1
        self.word_to_index["<unk>"] = 2
#         self.word_to_index["<SOS>"]
        ## self.index_to_word = # dictionary with values as words and keys as their corresponding index number
        ## self.index_to_char = # dictionary with values as characters and keys as their corresponding index number


    def normalize_answer(self,s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))



    def create_vocabulary(self,vocab_freq, vocab_size , data_path):
        """
        This function creates dictionaries namely:
        1) word_to_index
        2) char_to_index
        3) index_to_word
        4) index_to_char

        and dumps them into pickle file namely: "dictionaries.pkl"
        """


        for filename in self.vocab_input_files:
            with open(filename,'r', encoding = 'utf-8') as file_input:

                for line in file_input:
                    words = self.normalize_answer(line).strip().split()
#                     print(words)S
                    for word in words:
                        if not (word in self.vocab):
                            self.vocab[word] = 1
                        else:
                            self.vocab[word] +=1

        if vocab_freq == 0:
            vocab_words = sorted(self.vocab,key=self.vocab.get,reverse=True)


#         print(vocab_words)

        temp_index = 3
        for word in vocab_words:
            if temp_index < vocab_size and word not in self.word_to_index:
                self.word_to_index[word] = temp_index
                temp_index += 1

#         print(len(self.word_to_index))

        self.vocab_size = len(self.word_to_index)
        self.index_to_word =  {v: k for k, v in self.word_to_index.items()}


        characters = list(string.printable.lower())
        characters.remove(' ')

        char_ind = 1
        for c in characters:
            if c not in self.char_to_index:
                self.char_to_index[c] = char_ind
                char_ind += 1


        self.index_to_char = {v: k for k,v in self.char_to_index.items()}

        dict_all = {"word_to_index" : self.word_to_index, "char_to_index" : self.char_to_index,"index_to_word": self.index_to_word, "index_to_char": self.index_to_char}

        pickle.dump(dict_all, open(os.path.join(data_path, "dictionaries.pkl"), "wb")) ## creates dictionaries and stores in memory as pickle files
